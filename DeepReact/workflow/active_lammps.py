"""Active-learning LAMMPS stage — runs MD with a user-provided Deep Potential
input file (``example_md.lmp``).  No modification is performed — the file
is used as-is.
"""

from deepreact.workflow.base import BaseStage


class ActiveLammpsStage(BaseStage):
    """Run LAMMPS with a Deep Potential model for active learning."""

    stage_name = "active_lammps"

    def execute(self) -> None:
        al = self.config.active_learning
        cfg = self.config.lammps
        cwd = self.config.config_dir

        lmp_input = al.lammps_input or cfg.input
        lammps_cwd = cfg.cwd if cfg.cwd is not None else cwd

        cmd = [cfg.executable, "-in", lmp_input.name]
        self._run_cmd(cmd, cwd=lammps_cwd)

        self.logger.info(
            f"Active-learning trajectory: {cwd / al.trajectory_output}"
        )
