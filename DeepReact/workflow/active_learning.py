"""Active-learning loop.

After the initial DP model is trained this loop:

1. Runs LAMMPS with the Deep Potential model (→ trajectory).
2. Extracts structures via mddatasetbuilder (→ .gjf).
3. Runs / waits for Gaussian (→ .log).
4. Converts .log to DeepMD datasets via dpdata.
5. Evaluates the model with ``dp test`` and selects the *top-N*
   worst-predicted structures by force error.
6. Moves those structures into the training set and continues
   training from the last checkpoint (→ improved model).

Each iteration is checkpointed so that interrupted loops can resume.
"""

import shutil
from pathlib import Path

from deepreact.config.parser import (
    GaussianConfig,
    MddatasetbuilderConfig,
)
from deepreact.utils.checkpoint import CheckpointManager
from deepreact.utils.logger import WorkflowLogger
from deepreact.workflow.active_lammps import ActiveLammpsStage
from deepreact.workflow.check import CheckStage
from deepreact.workflow.dataset_builder import DatasetBuilderStage
from deepreact.workflow.dpdata_converter import DpdataStage
from deepreact.workflow.dptest import DptestStage
from deepreact.workflow.gaussian import GaussianStage


# Lightweight config-like objects used to re-run existing stages with
# the active-learning paths.
def _al_mdb_cfg(orig: MddatasetbuilderConfig, al, iteration: int, cwd: Path) -> MddatasetbuilderConfig:
    """mddatasetbuilder config for the AL loop — gjf stay in
    ``dataset_loopN_gjf/`` (renamed to ``dataset_loopN/`` afterwards)."""
    name = f"loop{iteration}"
    gjf_dir = (cwd / f"dataset_{name}_gjf").resolve()
    return MddatasetbuilderConfig(
        trajectory=(cwd / al.trajectory_output).resolve(),
        bonds=None,
        output=gjf_dir,
        atoms=orig.atoms,
        name=name,
        cutoff=orig.cutoff,
        stride=orig.stride,
        nprocjob=orig.nprocjob,
        keywords=orig.keywords,
        gjf_inject=orig.gjf_inject,
        command=orig.command,
    )


def _al_gaussian_cfg(orig: GaussianConfig, al, iteration: int, cwd: Path) -> GaussianConfig:
    """Gaussian reads from ``dataset_loopN/`` (after rename)."""
    name = f"loop{iteration}"
    gjf_dir = (cwd / f"dataset_{name}").resolve()
    return GaussianConfig(
        mode=orig.mode,
        executable=orig.executable,
        input_dir=gjf_dir,
        output_dir=gjf_dir / "log",
        template=orig.template,
    )


class ActiveLearningLoop:
    """Orchestrate one active-learning iteration."""

    def __init__(self, config, logger: WorkflowLogger):
        self.config = config
        self.logger = logger
        self.cwd = config.config_dir
        self.checkpoint = CheckpointManager(
            config.project.workdir / "al_checkpoints"
        )

    def run(self) -> None:
        al = self.config.active_learning
        if not al.enabled:
            return

        self.logger.section("ACTIVE LEARNING")

        for iteration in range(1, al.iterations + 1):
            tag = f"al_iter{iteration}"
            if self.checkpoint.is_done(tag):
                self.logger.stage_skip(f"Active Learning iteration {iteration}")
                continue

            self.logger.section(f"AL Iteration {iteration}")
            self._run_iteration(iteration)
            self.checkpoint.mark_done(tag)
            break  # single iteration for now; remove to loop

        self.logger.section("ACTIVE LEARNING COMPLETE")

    # ------------------------------------------------------------------
    def _run_iteration(self, iteration: int = 1) -> None:
        self._iteration = iteration
        al = self.config.active_learning
        cwd = self.cwd

        # 1. LAMMPS with Deep Potential
        lam = ActiveLammpsStage(self.config, self.logger, self.checkpoint)
        lam.run()

        # 2. mddatasetbuilder — reuse existing stage with AL paths
        al_mdb_cfg = _al_mdb_cfg(self.config.mddatasetbuilder, al, iteration, cwd)
        mdb = DatasetBuilderStage(
            _patch_config(self.config, mddatasetbuilder=al_mdb_cfg),
            self.logger,
            self.checkpoint,
        )
        mdb.stage_name = "al_mddatasetbuilder"
        mdb.run()

        # Rename  dataset_loopN_gjf/ → dataset_loopN/
        name = f"loop{iteration}"
        gjf_src = cwd / f"dataset_{name}_gjf"
        gjf_dst = cwd / f"dataset_{name}"
        if gjf_src.is_dir():
            if gjf_dst.exists():
                shutil.rmtree(str(gjf_dst))
            gjf_src.rename(gjf_dst)
            self.logger.info(f"Renamed {gjf_src.name} → {gjf_dst.name}")

        # 3. Gaussian
        al_gau_cfg = _al_gaussian_cfg(self.config.gaussian, al, iteration, cwd)
        gau = GaussianStage(
            _patch_config(self.config, gaussian=al_gau_cfg),
            self.logger,
            self.checkpoint,
        )
        gau.stage_name = "al_gaussian"
        gau.run()

        # 4. Convergence check
        chk = CheckStage(
            _patch_config(self.config, gaussian=al_gau_cfg),
            self.logger,
            self.checkpoint,
        )
        chk.stage_name = "al_check"
        chk.run()

        # 5. dpdata — read logs from dataset_loopN/log, write to deepmd_loopN
        dpdata_out = f"deepmd_{name}"
        dp_cfg = _patch_config(self.config, gaussian=al_gau_cfg)
        dp_cfg._dpdata_output = dpdata_out                         # bypass hasattr
        dp = DpdataStage(dp_cfg, self.logger, self.checkpoint)
        dp.stage_name = "al_dpdata"
        dp.run()

        # 6. dptest — evaluate model (for reporting, not filtering)
        al.dptest_system = str(cwd / dpdata_out)
        dt = DptestStage(self.config, self.logger, self.checkpoint)
        dt.run()

        # 7. Retrain — merge AL data, re-split, train from scratch
        self._retrain(dpdata_out)

    # ------------------------------------------------------------------
    def _retrain(self, dpdata_dir_name: str) -> None:
        al = self.config.active_learning
        cwd = self.cwd
        al_data_dir = cwd / dpdata_dir_name
        data_dir = cwd / "deepmd_data"

        # 1. Find the highest set number already in deepmd_data/
        max_n = _max_set_number(data_dir)

        # 2. Move sets from deepmd_loopN/ → deepmd_data/ with renamed indices
        added = 0
        for s in sorted(al_data_dir.iterdir()):
            if not s.is_dir() or not s.name.startswith("set."):
                continue
            max_n += 1
            dst = data_dir / f"set.{max_n:03d}"
            shutil.move(str(s), str(dst))
            added += 1
        self.logger.info(f"Moved {added} sets from {al_data_dir.name} → {data_dir.name}")

        # 3. Remove the now-empty deepmd_loopN/ directory
        if al_data_dir.is_dir():
            shutil.rmtree(str(al_data_dir))
            self.logger.info(f"Removed {al_data_dir.name}")

        # 4. Flatten: move set.xxx out of train/ val/ test/ back to deepmd_data/
        for sub in ["train", "val", "test"]:
            subdir = data_dir / sub
            if not subdir.is_dir():
                continue
            for s in sorted(subdir.iterdir()):
                if s.is_dir() and s.name.startswith("set."):
                    dst = data_dir / s.name
                    if dst.exists():
                        shutil.rmtree(str(dst))
                    shutil.move(str(s), str(dst))
        self.logger.info("Flattened existing train/val/test back into deepmd_data")

        # 5. Re-split using the same ratios
        from deepreact.workflow.split import SplitStage

        split = SplitStage(self.config, self.logger, self.checkpoint)
        split.stage_name = "al_split"
        split.run()

        # 4. Train with original train.json
        train_json = al.train_json or self.config.deepmd.train
        exe = self.config.deepmd.executable
        cmd = [exe, "train", str(train_json)]
        self.logger.info(f"Retraining: dp train {train_json.name}")
        from deepreact.utils.command import run_command

        run_command(cmd, logger=self.logger, cwd=cwd)

        # 5. Freeze to graph_alN.pb
        graph_name = f"graph_al{self._iteration}.pb"
        graph = cwd / graph_name
        if graph.exists():
            graph.unlink()
        freeze_cmd = [exe, "freeze", "-o", str(graph)]
        self.logger.info(f"Freezing → {graph_name} ...")
        run_command(freeze_cmd, logger=self.logger, cwd=cwd)

        if graph.exists():
            import deepreact

            dest = Path(deepreact.__file__).parent.parent / graph.name
            shutil.copy2(str(graph), str(dest))
            self.logger.info(f"Model: {graph}")
            self.logger.info(f"Copied → {dest}")
        else:
            self.logger.info(f"Freeze did not produce {graph_name} — check manually.")


def _max_set_number(data_dir: Path) -> int:
    """Return the highest set index found anywhere under *data_dir*."""
    max_n = -1
    for d in data_dir.rglob("set.[0-9][0-9][0-9]"):
        if d.is_dir():
            try:
                n = int(d.name.split(".")[-1])
                if n > max_n:
                    max_n = n
            except ValueError:
                continue
    return max_n


def _patch_config(original, **overrides):
    """Return a shallow copy of *original* with fields replaced."""
    import copy

    cfg = copy.copy(original)
    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg
