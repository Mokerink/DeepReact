"""dpdata conversion stage — builds DeepMD datasets from Gaussian outputs.

If ``scripts.dpdata`` is set the external script is used; otherwise
a built-in converter processes every ``.log`` and ``.fchk`` file found
under the numbered sub-directories of the Gaussian log directory.
"""

import re

from deepreact.workflow.base import BaseStage


class DpdataStage(BaseStage):
    """Convert Gaussian outputs to DeepMD training data."""

    stage_name = "dpdata"

    def execute(self) -> None:
        if self.config.scripts.dpdata:
            self._run_external()
        else:
            self._run_builtin()

    # -- external script ------------------------------------------------
    def _run_external(self) -> None:
        import shlex
        cmd = shlex.split(self.config.scripts.dpdata)
        self._run_cmd(cmd, cwd=self.config.config_dir)
        self.logger.info("dpdata conversion complete.")

    # -- built-in -------------------------------------------------------
    def _run_builtin(self) -> None:
        try:
            import dpdata
        except ImportError:
            raise RuntimeError(
                "dpdata is required for built-in conversion. "
                "Install it with:  pip install dpdata"
            )

        log_dir = self.config.gaussian.output_dir
        if not log_dir.is_dir():
            self.logger.info(f"Log directory not found: {log_dir}")
            return

        out_root = self.config.config_dir / getattr(
            self.config, "_dpdata_output", "deepmd_data"
        )
        out_root.mkdir(parents=True, exist_ok=True)

        digit_re = re.compile(r"^\d+$")
        supported_ext = (".log", ".fchk")
        global_idx = 0
        total = 0

        for subdir in sorted(log_dir.iterdir()):
            if not subdir.is_dir() or not digit_re.match(subdir.name):
                continue

            for fpath in sorted(subdir.iterdir()):
                if not fpath.suffix.lower() in supported_ext:
                    continue

                if fpath.suffix.lower() == ".log":
                    fmt = "gaussian/log"
                else:
                    fmt = "gaussian/fchk"

                try:
                    self.logger.info(f"[{global_idx:03d}] {fpath.name}")
                    system = dpdata.LabeledSystem(str(fpath), fmt=fmt)
                    out_path = out_root / f"set.{global_idx:03d}"
                    system.to("deepmd/npy", str(out_path))
                    global_idx += 1
                    total += 1
                except Exception as e:
                    self.logger.info(f"  Failed: {e}")

        self.logger.info(
            f"dpdata conversion complete — {total} systems in {out_root}"
        )
