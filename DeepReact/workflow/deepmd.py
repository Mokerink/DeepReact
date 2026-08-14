"""DeepMD training stage."""

import shutil
from pathlib import Path

import deepreact

from deepreact.workflow.base import BaseStage


class DeepmdStage(BaseStage):
    """Run DeepMD-kit training, freeze the model, and copy graph.pb alongside
    the deepreact package directory."""

    stage_name = "deepmd"

    def execute(self) -> None:
        cfg = self.config.deepmd
        train_json = cfg.train

        if not train_json.exists():
            raise FileNotFoundError(
                f"Training config not found: {train_json}"
            )

        # --- train ---
        cmd = [cfg.executable, "train", str(train_json)]
        self._run_cmd(cmd, cwd=self.config.config_dir)

        cwd = self.config.config_dir

        # --- freeze (v3 checkpoints → graph.pb) ---
        graph = cwd / "graph.pb"
        frozen = cwd / "frozen_model.pb"

        if not graph.exists() and not frozen.exists():
            ckpt_files = list(cwd.rglob("model.ckpt*"))
            if ckpt_files:
                self.logger.info("Freezing model ...")
                freeze_cmd = [cfg.executable, "freeze", "-o", str(graph)]
                self._run_cmd(freeze_cmd, cwd=cwd)

        # --- report ---
        final = graph if graph.exists() else (frozen if frozen.exists() else None)
        if final:
            self.logger.info(f"Trained model: {final}")
        else:
            ckpt_files = list(cwd.rglob("model.ckpt*"))
            pt_files = list(cwd.rglob("*.pt"))
            if ckpt_files:
                self.logger.info("Checkpoint files:")
                for f in sorted(ckpt_files):
                    self.logger.info(f"  {f}")
                self.logger.info("dp freeze did not produce graph.pb — check manually.")
            elif pt_files:
                self.logger.info("PyTorch model files:")
                for f in pt_files:
                    self.logger.info(f"  {f}")
            else:
                self.logger.info("No model file found.")

        # --- copy graph.pb alongside the deepreact package ---
        if final:
            pkg_dir = Path(deepreact.__file__).parent   # .../deepreact/
            dest = pkg_dir.parent / final.name          # .../  (one level up)
            shutil.copy2(str(final), str(dest))
            self.logger.info(f"Copied {final.name} -> {dest}")
