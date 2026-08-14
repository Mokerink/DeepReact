"""Subprocess execution utilities."""

import os
import shlex
import shutil
import subprocess
from pathlib import Path

from deepreact.utils.logger import WorkflowLogger


class CommandError(Exception):
    """Raised when an external command fails."""

    def __init__(self, command: str, returncode: int, stderr: str):
        self.command = command
        self.returncode = returncode
        self.stderr = stderr
        super().__init__(
            f"Command '{command}' failed with return code {returncode}"
        )


def run_command(
    command: str | list[str],
    logger: WorkflowLogger | None = None,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    """Run an external command and capture its output.

    For list commands the executable is resolved to its absolute path
    via ``shutil.which`` so that conda/WSL PATH issues are avoided.
    String commands use shell=True unchanged.
    """
    display_cmd = command if isinstance(command, str) else shlex.join(command)
    if logger:
        logger.command(display_cmd)

    is_shell = isinstance(command, str)

    if not is_shell:
        # Resolve the executable to an absolute path so the subprocess
        # does not depend on inheriting PATH correctly (WSL / conda).
        resolved = shutil.which(command[0])
        if resolved is None:
            raise CommandError(
                display_cmd, -1,
                f"'{command[0]}' not found on PATH. "
                f"Activate the conda environment and retry.",
            )
        command = [resolved] + list(command[1:])

    kwargs: dict = dict(
        capture_output=True,
        text=True,
        cwd=str(cwd) if cwd else None,
    )
    if env:
        kwargs["env"] = {**os.environ, **env}

    try:
        result = subprocess.run(command, shell=is_shell, **kwargs)
    except FileNotFoundError:
        raise CommandError(display_cmd, -1, "Executable not found on PATH")

    if logger:
        if result.stdout:
            logger.info(result.stdout.strip())
        if result.stderr:
            logger.info(f"[stderr]\n{result.stderr.strip()}")

    if result.returncode != 0:
        if logger:
            logger.error(
                f"Command failed with return code {result.returncode}\n"
                f"stderr: {result.stderr}"
            )
        raise CommandError(display_cmd, result.returncode, result.stderr)

    return result
