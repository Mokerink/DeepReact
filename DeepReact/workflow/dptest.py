"""dptest stage — evaluate a DP model and select structures whose
energy or force error exceeds user-configured RMSE thresholds.

Runs ``dp test -m <model> -s <system> -d <output>``, which produces
``<output>.e.out`` and ``<output>.f.out``.  Each file groups per-frame
predictions by system (``set.xxx``).  Per-structure RMSE is computed and
structures exceeding *energy_rmse_threshold* or *force_rmse_threshold*
are selected.
"""

import re
from math import sqrt
from pathlib import Path

from deepreact.workflow.base import BaseStage


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_energy_file(path: Path) -> dict[int, list[float]]:
    """Parse ``dptest_out.e.out`` → ``{set_idx: [abs_error, …]}``.

    Format::

        # /path/set.022: data_e pred_e
        -1.040e+03 -3.148e+02
        # /path/set.060: data_e pred_e
        -4.080e+03 -4.324e+03
        …
    """
    system_errors: dict[int, list[float]] = {}
    current_set: int | None = None

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith("#"):
            # Extract set index from the system path
            m = re.search(r"set\.(\d+)", stripped)
            if m:
                current_set = int(m.group(1))
                if current_set not in system_errors:
                    system_errors[current_set] = []
            continue

        parts = stripped.split()
        if len(parts) >= 2 and current_set is not None:
            try:
                data_e = float(parts[0])
                pred_e = float(parts[1])
                system_errors[current_set].append(abs(data_e - pred_e))
            except ValueError:
                continue

    return system_errors


def _parse_force_file(path: Path) -> dict[int, list[float]]:
    """Parse ``dptest_out.f.out`` → ``{set_idx: [force_error_per_atom, …]}``.

    Format::

        # /path/set.022: data_fx data_fy data_fz pred_fx pred_fy pred_fz
        0.1 0.2 0.3 0.11 0.19 0.31
        …
    """
    system_errors: dict[int, list[float]] = {}
    current_set: int | None = None

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith("#"):
            m = re.search(r"set\.(\d+)", stripped)
            if m:
                current_set = int(m.group(1))
                if current_set not in system_errors:
                    system_errors[current_set] = []
            continue

        parts = stripped.split()
        if len(parts) >= 6 and current_set is not None:
            try:
                dfx = float(parts[0]) - float(parts[3])
                dfy = float(parts[1]) - float(parts[4])
                dfz = float(parts[2]) - float(parts[5])
                system_errors[current_set].append(sqrt(dfx**2 + dfy**2 + dfz**2))
            except ValueError:
                continue

    return system_errors


def _compute_rmse(errors: dict[int, list[float]]) -> dict[int, dict]:
    """Convert per-frame error lists into per-structure RMSE values."""
    result: dict[int, dict] = {}
    for sid, errs in errors.items():
        n = len(errs)
        if n == 0:
            continue
        rmse = sqrt(sum(e**2 for e in errs) / n)
        result[sid] = {"n_frames": n, "rmse": rmse}
    return result


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------

class DptestStage(BaseStage):
    """Run ``dp test`` and select structures exceeding RMSE thresholds."""

    stage_name = "dptest"

    def execute(self) -> None:
        al = self.config.active_learning
        cwd = self.config.config_dir
        exe = self.config.deepmd.executable

        system = cwd / al.dptest_system
        output = cwd / al.dptest_output

        # 1 — run dp test
        cmd = [
            exe, "test",
            "-m", str(cwd / al.dptest_model),
            "-s", str(system),
            "-d", str(output),
        ]
        self._run_cmd(cmd, cwd=cwd)

        # 2 — parse energy errors
        e_file = Path(str(output) + ".e.out")
        e_errs = _parse_energy_file(e_file) if e_file.exists() else {}
        e_rmse = _compute_rmse(e_errs)

        # 3 — parse force errors (optional)
        f_file = Path(str(output) + ".f.out")
        f_errs = _parse_force_file(f_file) if f_file.exists() else {}
        f_rmse = _compute_rmse(f_errs)

        # 4 — collect all structure indices
        all_sids = set(e_rmse.keys()) | set(f_rmse.keys())

        if not all_sids:
            self.logger.info("No structures found in dptest output.")
            return

        # 5 — threshold filter
        e_thr = al.energy_rmse_threshold
        f_thr = al.force_rmse_threshold
        selected: dict[int, dict] = {}

        for sid in sorted(all_sids):
            ev = e_rmse.get(sid, {}).get("rmse", 0.0)
            fv = f_rmse.get(sid, {}).get("rmse", 0.0)
            if ev > e_thr or fv > f_thr:
                selected[sid] = {"e_rmse": ev, "f_rmse": fv}

        # 6 — report
        self.logger.info(
            f"dptest: {len(all_sids)} structures evaluated"
        )
        self.logger.info(
            f"Energy RMSE threshold: {e_thr} eV  |  "
            f"Force RMSE threshold: {f_thr} eV/Å"
        )

        if selected:
            self.logger.info(
                f"Selected {len(selected)} structures above threshold:"
            )
            for sid in sorted(selected):
                v = selected[sid]
                self.logger.info(
                    f"  set.{sid:03d}  "
                    f"e_rmse={v['e_rmse']:.6f}  "
                    f"f_rmse={v['f_rmse']:.6f}"
                )
        else:
            self.logger.info("No structures exceed the RMSE thresholds.")

        self._selected_frames = sorted(selected.keys())

    @property
    def selected_frames(self) -> list[int]:
        return getattr(self, "_selected_frames", [])
