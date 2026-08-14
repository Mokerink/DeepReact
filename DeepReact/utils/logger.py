"""Logging system for DeepReact workflows."""

import logging
from datetime import datetime
from pathlib import Path


class WorkflowLogger:
    """Logger that writes to both console and a log file in the project directory."""

    def __init__(self, workdir: Path, project_name: str):
        workdir.mkdir(parents=True, exist_ok=True)

        self.log_path = workdir / "deepreact.log"
        self._logger = logging.getLogger(f"deepreact.{project_name}")
        self._logger.setLevel(logging.DEBUG)
        self._logger.handlers.clear()

        self._file_handler = logging.FileHandler(
            str(self.log_path), mode="a", encoding="utf-8"
        )
        self._file_handler.setLevel(logging.DEBUG)
        file_fmt = logging.Formatter("%(message)s")
        self._file_handler.setFormatter(file_fmt)

        self._console_handler = logging.StreamHandler()
        self._console_handler.setLevel(logging.INFO)
        console_fmt = logging.Formatter("%(message)s")
        self._console_handler.setFormatter(console_fmt)

        self._logger.addHandler(self._file_handler)
        self._logger.addHandler(self._console_handler)

    def stage_start(self, name: str) -> None:
        """Record the start of a workflow stage."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._logger.info(f"\n[{ts}]")
        self._logger.info(f"Starting {name} ...")

    def stage_end(self, name: str, files: list[str] | None = None) -> None:
        """Record the end of a workflow stage."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._logger.info(f"\n[{ts}]")
        self._logger.info(f"{name} finished")
        if files:
            self._logger.info("Output files:")
            for f in files:
                self._logger.info(f"  {f}")

    def stage_skip(self, name: str) -> None:
        """Record that a stage is being skipped (already completed)."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._logger.info(f"\n[{ts}]")
        self._logger.info(f"Skipping {name} (already completed)")

    def command(self, cmd: str) -> None:
        """Record a command being executed."""
        self._logger.info(f"Command: {cmd}")

    def error(self, message: str) -> None:
        """Record an error."""
        self._logger.error(f"ERROR: {message}")

    def info(self, message: str) -> None:
        """Record an informational message."""
        self._logger.info(message)

    def section(self, title: str) -> None:
        """Print a section header."""
        self._logger.info("")
        self._logger.info("=" * 60)
        self._logger.info(f"  {title}")
        self._logger.info("=" * 60)

    @property
    def file_path(self) -> Path:
        return self.log_path
