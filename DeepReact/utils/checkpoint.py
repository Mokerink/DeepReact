"""Checkpoint and resume system for workflow recovery."""

from pathlib import Path


class CheckpointManager:
    """Manages workflow checkpoints via .done marker files.

    Checkpoint files are stored in <workdir>/.deepreact/.
    Each completed stage writes a <stage>.done file.
    """

    def __init__(self, workdir: Path):
        self._checkpoint_dir = workdir / ".deepreact"
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def is_done(self, stage: str) -> bool:
        """Check whether a stage has already completed."""
        return (self._checkpoint_dir / f"{stage}.done").exists()

    def mark_done(self, stage: str) -> None:
        """Mark a stage as completed."""
        (self._checkpoint_dir / f"{stage}.done").touch()

    def reset(self, stage: str | None = None) -> None:
        """Remove checkpoint files.

        Args:
            stage: Specific stage to reset, or None to reset all.
        """
        if stage:
            marker = self._checkpoint_dir / f"{stage}.done"
            if marker.exists():
                marker.unlink()
        else:
            for f in self._checkpoint_dir.glob("*.done"):
                f.unlink()

    def list_completed(self) -> list[str]:
        """Return a list of completed stage names."""
        return [
            p.stem.replace(".done", "")
            for p in self._checkpoint_dir.glob("*.done")
        ]
