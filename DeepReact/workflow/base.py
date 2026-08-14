"""Base class for workflow stages."""

from pathlib import Path

from deepreact.config.parser import Config
from deepreact.utils.checkpoint import CheckpointManager
from deepreact.utils.command import run_command
from deepreact.utils.logger import WorkflowLogger


class BaseStage:
    """Common behaviour for all workflow stages."""

    stage_name: str = "base"

    def __init__(
        self,
        config: Config,
        logger: WorkflowLogger,
        checkpoint: CheckpointManager,
    ):
        self.config = config
        self.logger = logger
        self.checkpoint = checkpoint

    def run(self) -> None:
        """Run the stage, skipping if already completed."""
        if self.checkpoint.is_done(self.stage_name):
            self.logger.stage_skip(self.stage_name)
            return

        self.logger.stage_start(self.stage_name)
        try:
            self.execute()
        except Exception as e:
            self.logger.error(f"{self.stage_name} failed: {e}")
            raise
        self.checkpoint.mark_done(self.stage_name)
        self.logger.stage_end(self.stage_name)

    def execute(self) -> None:
        """Override in subclasses to implement the stage logic."""
        raise NotImplementedError

    def _run_cmd(
        self,
        command: str | list[str],
        cwd: str | Path | None = None,
    ) -> None:
        """Run a command with logging."""
        run_command(command, logger=self.logger, cwd=cwd)
