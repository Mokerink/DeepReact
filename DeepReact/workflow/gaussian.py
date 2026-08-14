"""Gaussian calculation stage.

Two modes:
  - export :  Pause after ``.gjf`` generation — the user runs Gaussian
              manually, places ``.log`` files in *output_dir*, and re-runs
              the workflow to continue.
  - run    :  Execute Gaussian on each ``.gjf`` file automatically.
"""

import subprocess
import sys
from pathlib import Path

from deepreact.workflow.base import BaseStage


class GaussianStage(BaseStage):
    """Run Gaussian calculations or pause for manual execution."""

    stage_name = "gaussian"

    def execute(self) -> None:
        cfg = self.config.gaussian

        if cfg.mode == "export":
            self._handle_export(cfg)
            return

        if cfg.mode != "run":
            raise ValueError(
                f"Unknown Gaussian mode '{cfg.mode}'. Expected 'run' or 'export'."
            )

        self._run_gaussian(cfg)

    # ------------------------------------------------------------------
    def _handle_export(self, cfg) -> None:
        """Check whether .log files exist; if not, pause for the user."""
        log_files = list(cfg.output_dir.rglob("*.log"))
        if log_files:
            self.logger.info(
                f"Found {len(log_files)} Gaussian log files — continuing."
            )
            return

        gjf_files = list(cfg.input_dir.rglob("*.gjf"))
        self.logger.section("GAUSSIAN — MANUAL STEP REQUIRED")
        self.logger.info(f"GJF files ({len(gjf_files)}):")
        for f in gjf_files:
            self.logger.info(f"  {f}")
        self.logger.info("")
        self.logger.info("1. Run Gaussian on each .gjf file.")
        self.logger.info(f"2. Place the .log files in: {cfg.output_dir}")
        self.logger.info("3. Re-run:  deepreact run config.yaml")
        self.logger.info("")
        self.logger.info("(The workflow will resume from this point.)")
        self.logger.info("=" * 60)
        sys.exit(0)

    # ------------------------------------------------------------------
    def _run_gaussian(self, cfg) -> None:
        gjf_files = sorted(cfg.input_dir.rglob("*.gjf"))
        if not gjf_files:
            self.logger.info("No .gjf files found — nothing to run.")
            return

        cfg.output_dir.mkdir(parents=True, exist_ok=True)

        total = len(gjf_files)
        for i, gjf in enumerate(gjf_files, start=1):
            log_file = cfg.output_dir / f"{gjf.stem}.log"
            self.logger.info(f"[{i}/{total}] Running: {gjf.name}")

            cmd = [cfg.executable, str(gjf)]
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=str(cfg.output_dir),
                )
                if result.returncode != 0:
                    self.logger.error(
                        f"Gaussian failed for {gjf.name} "
                        f"(return code {result.returncode})"
                    )
                    log_file.write_text(
                        result.stdout + "\n" + result.stderr,
                        encoding="utf-8",
                    )
                    raise RuntimeError(
                        f"Gaussian calculation failed for {gjf.name}"
                    )

                default_log = Path.cwd() / f"{gjf.stem}.log"
                if default_log.exists() and default_log != log_file:
                    default_log.rename(log_file)

                self.logger.info(f"  -> {log_file.name}")

            except Exception:
                self.logger.error(
                    f"Gaussian stage failed on file {gjf.name}"
                )
                raise

        self.logger.info(f"Completed {total} Gaussian calculations.")
