"""Gaussian convergence checking stage.

If ``scripts.check`` is set in the config the external script is run.
Otherwise a built-in check scans *output_dir* for ``.log`` files and
verifies that every one contains ``Normal termination``.  Non-converged
files are moved to an ``error/`` sub-directory.
"""

import re
import shutil

from deepreact.workflow.base import BaseStage


class CheckStage(BaseStage):
    """Check Gaussian output for normal termination."""

    stage_name = "check"

    def execute(self) -> None:
        if self.config.scripts.check:
            self._run_external()
        else:
            self._run_builtin()

    # -- external script ------------------------------------------------
    def _run_external(self) -> None:
        import shlex
        cmd = shlex.split(self.config.scripts.check)
        self._run_cmd(cmd, cwd=self.config.config_dir)
        self.logger.info("Convergence check passed.")

    # -- built-in -------------------------------------------------------
    def _run_builtin(self) -> None:
        log_dir = self.config.gaussian.output_dir
        if not log_dir.is_dir():
            self.logger.info(f"Log directory not found: {log_dir}")
            return

        error_dir = log_dir / "error"
        error_dir.mkdir(exist_ok=True)

        digit_re = re.compile(r"^\d+$")
        total = 0
        errors = 0

        for subdir in sorted(log_dir.iterdir()):
            if not subdir.is_dir() or not digit_re.match(subdir.name):
                continue

            for log_file in sorted(subdir.glob("*.log")):
                total += 1
                try:
                    text = log_file.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    self.logger.info(f"[read error] {log_file.name}")
                    continue

                if "Normal termination" in text:
                    self.logger.info(f"  OK  {log_file.name}")
                else:
                    errors += 1
                    dest = error_dir / log_file.name
                    if dest.exists():
                        dest = error_dir / f"{log_file.stem}_dup{log_file.suffix}"
                    shutil.move(str(log_file), str(dest))
                    self.logger.info(f"  FAIL -> error/{dest.name}")

        self.logger.info(f"Checked {total} .log files, {errors} failed")
        if errors:
            self.logger.info(f"Non-converged files moved to: {error_dir}")
        else:
            self.logger.info("All files converged normally.")
