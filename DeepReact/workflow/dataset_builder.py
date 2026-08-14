"""MDDatasetBuilder stage — converts LAMMPS trajectories to Gaussian inputs."""

import shutil

from deepreact.workflow.base import BaseStage


class DatasetBuilderStage(BaseStage):
    """Run mddatasetbuilder to generate Gaussian input files.

    - If a raw ``command`` string is set in the config it is used verbatim.
    - Otherwise the command is built from the structured fields.
    - After the builder finishes, ``.gjf`` files are collected, the
      optional ``gjf_inject`` line is inserted, and the ``dataset_data``
      directory (xyz files) is removed.
    """

    stage_name = "mddatasetbuilder"

    def execute(self) -> None:
        cfg = self.config.mddatasetbuilder
        cwd = self.config.config_dir

        if cfg.command:
            cmd = cfg.command
        else:
            parts = [
                "python -m mddatasetbuilder",
                f"-d {cfg.trajectory}",
            ]
            if cfg.bonds is not None:
                parts.append(f"-b {cfg.bonds}")
            parts += [
                f"-a {cfg.atoms}",
                f"-n {cfg.name}",
                f"-c {cfg.cutoff}",
                f"-s {cfg.stride}",
                f"--nprocjob {cfg.nprocjob}",
                f'-k "{cfg.keywords}"',
            ]
            cmd = " ".join(parts)

        self._run_cmd(cmd, cwd=cwd)

        # --- collect .gjf files from the builder output directory ---
        # mddatasetbuilder creates  dataset_<name>_gjf/<n>/*.gjf  in cwd.
        src_dir = cwd / f"dataset_{cfg.name}_gjf"
        gjf_files: list = []
        if src_dir.is_dir():
            gjf_files = sorted(src_dir.rglob("*.gjf"))

        # --- inject user-specified line after the first line of each .gjf ---
        if cfg.gjf_inject:
            for gjf in gjf_files:
                lines = gjf.read_text(encoding="utf-8").splitlines()
                lines.insert(1, cfg.gjf_inject)          # becomes new line 2
                gjf.write_text("\n".join(lines) + "\n", encoding="utf-8")
            self.logger.info(
                f"Injected '{cfg.gjf_inject}' into {len(gjf_files)} .gjf files"
            )

        # --- move .gjf files to the configured output directory ---
        # Skip the move when the output directory is the same as the source.
        if cfg.output.resolve() != src_dir.resolve():
            cfg.output.mkdir(parents=True, exist_ok=True)
            for gjf in gjf_files:
                dest = cfg.output / gjf.name
                if gjf.parent != src_dir:
                    dest = cfg.output / gjf.relative_to(src_dir)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(gjf), str(dest))
            # remove the now-empty gjf source tree
            if src_dir.is_dir():
                shutil.rmtree(str(src_dir))

        # --- remove dataset_<name> (xyz files, not needed downstream) ---
        xyz_dir = cwd / f"dataset_{cfg.name}"
        if xyz_dir.is_dir():
            shutil.rmtree(str(xyz_dir))
            self.logger.info(f"Removed {xyz_dir.name} (xyz files)")

        gjf_final = sorted(cfg.output.rglob("*.gjf"))
        self.logger.info(f"Gaussian input files ({len(gjf_final)}):")
        for f in gjf_final:
            self.logger.info(f"  {f}")
