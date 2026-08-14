"""LAMMPS reactive molecular dynamics stage."""

import gzip
import shutil
import tarfile

from deepreact.workflow.base import BaseStage


class LammpsStage(BaseStage):
    """Run a LAMMPS reactive MD simulation.

    After LAMMPS finishes, if the bonds output is a ``.gz`` archive it is
    automatically extracted.  Both plain gzip and tar.gz are handled.
    When the archive contains a directory the inner bonds file is moved
    into the location the config expects, so downstream stages can use
    the same path.
    """

    stage_name = "lammps"

    def execute(self) -> None:
        cfg = self.config.lammps
        cwd = cfg.cwd if cfg.cwd is not None else cfg.input.parent
        cmd = [cfg.executable, "-in", cfg.input.name]
        self._run_cmd(cmd, cwd=cwd)

        self._extract_bonds(cfg)

        self.logger.info("Output files:")
        self.logger.info(f"  trajectory: {cfg.output}")
        self.logger.info(f"  bonds:      {cfg.bonds}")

    @staticmethod
    def _extract_bonds(cfg) -> None:
        """Decompress the gzipped bonds archive if present."""
        bonds_gz = cfg.bonds.parent / (cfg.bonds.name + ".gz")
        if not bonds_gz.exists() or cfg.bonds.exists():
            return

        # Try tar.gz first — LAMMPS may output a gzipped-tar directory.
        try:
            with tarfile.open(bonds_gz, "r:gz") as tar:
                tar.extractall(path=cfg.bonds.parent)
        except (tarfile.ReadError, EOFError):
            # Plain gzip → single file.
            with gzip.open(bonds_gz, "rb") as f_in, open(cfg.bonds, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            return

        # tar.gz extraction may have created a directory with the same
        # name as cfg.bonds, containing the real file inside.
        # Move the inner file to cfg.bonds so downstream stages work.
        if cfg.bonds.is_dir():
            inner = cfg.bonds / cfg.bonds.name
            if inner.is_file():
                tmp = cfg.bonds.parent / ("_" + cfg.bonds.name)
                shutil.move(str(inner), str(tmp))
                shutil.rmtree(str(cfg.bonds))
                shutil.move(str(tmp), str(cfg.bonds))
