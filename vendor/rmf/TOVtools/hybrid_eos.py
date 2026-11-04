from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

# A BPS crust table ships next to this module (scripts/agentV1-tools/TOVtools/bps.dat).
DEFAULT_BPS_PATH = str(Path(__file__).with_name('bps.dat'))


@dataclass
class BPSEntry:
    rho_b: float
    pressure_mev_fm3: float
    energy_density_mev_fm3: float

    def to_dict(self):
        return asdict(self)


@dataclass
class HybridEOSResult:
    bps_table: list[BPSEntry]
    core_table: list[object]
    merged_table: list[dict]
    match_density: float | None
    match_pressure: float | None
    match_energy_density: float | None

    def to_dict(self):
        return {
            'bps_table': [x.to_dict() for x in self.bps_table],
            'core_table': [x.to_dict() if hasattr(x, 'to_dict') else dict(x) for x in self.core_table],
            'merged_table': self.merged_table,
            'match_density': self.match_density,
            'match_pressure': self.match_pressure,
            'match_energy_density': self.match_energy_density,
        }


def read_bps_table(path: str | Path) -> list[BPSEntry]:
    rows = []
    for line in Path(path).read_text().splitlines():
        s = line.strip()
        if not s or s.startswith('#'):
            continue
        parts = s.split()
        if len(parts) < 3:
            continue
        rho_b = float(parts[0])
        pressure = float(parts[1])
        epsilon = float(parts[2])
        rows.append(BPSEntry(rho_b=rho_b, pressure_mev_fm3=pressure, energy_density_mev_fm3=epsilon))
    rows.sort(key=lambda x: x.rho_b)
    return rows


def core_point_to_dict(point) -> dict:
    if isinstance(point, dict):
        return dict(point)
    if hasattr(point, 'to_dict'):
        return point.to_dict()
    return {
        'rho_b': point.rho_b,
        'pressure_total_mev_fm3': point.pressure_total_mev_fm3,
        'epsilon_total_mev_fm3': point.epsilon_total_mev_fm3,
    }


def build_hybrid_eos_from_tables(
    bps_table: list[BPSEntry],
    core_table: list[object],
    *,
    rho_min_core: float = 0.08,
):
    """Join a BPS crust and an RMF core with a FIXED-DENSITY split.

    Convention for this toolkit/benchmark: BPS crust is used below
    ``rho_min_core`` (default 0.08 fm^-3) and the RMF core at/above it.
    The two segments are concatenated and cleaned to be strictly
    monotonic in both pressure and energy density (required for a stable
    TOV interpolation). This replaces the older 'match at the top of the
    BPS table' behaviour, which over-extended the soft crust and produced
    radii that were systematically too small.
    """
    if not bps_table:
        raise ValueError('empty BPS table')
    if not core_table:
        raise ValueError('empty core EOS table')

    core_dicts = [core_point_to_dict(x) for x in core_table]
    core_dicts.sort(key=lambda x: float(x['rho_b']))

    rows = []  # (segment, rho_b, pressure, epsilon, extra_dict_or_None)
    for b in bps_table:
        if float(b.rho_b) < rho_min_core:
            rows.append(('bps', float(b.rho_b), float(b.pressure_mev_fm3), float(b.energy_density_mev_fm3), None))
    for r in core_dicts:
        if float(r['rho_b']) >= rho_min_core:
            rows.append(('core', float(r['rho_b']), float(r['pressure_total_mev_fm3']),
                         float(r['epsilon_total_mev_fm3']), r))

    if not any(t == 'core' for t, *_ in rows):
        raise RuntimeError('no core points at or above rho_min_core=%g fm^-3' % rho_min_core)

    # sort by pressure and enforce strict monotonicity in P and epsilon
    rows.sort(key=lambda x: x[2])
    merged: list[dict] = []
    last_p = last_e = -1.0
    match = None
    for seg, rho, p, e, extra in rows:
        if p > last_p + 1e-15 and e > last_e + 1e-15:
            row = {'segment': seg, 'rho_b': rho,
                   'pressure_total_mev_fm3': p, 'epsilon_total_mev_fm3': e}
            if extra:
                row.update({k: v for k, v in extra.items() if k not in row})
            merged.append(row)
            last_p, last_e = p, e
            if seg == 'core' and match is None:
                match = row

    return HybridEOSResult(
        bps_table=bps_table,
        core_table=core_table,
        merged_table=merged,
        match_density=float(match['rho_b']) if match else float(rho_min_core),
        match_pressure=float(match['pressure_total_mev_fm3']) if match else None,
        match_energy_density=float(match['epsilon_total_mev_fm3']) if match else None,
    )


def build_hybrid_eos_with_bundled_crust(
    core_table: list[object],
    *,
    rho_min_core: float = 0.08,
    bps_path: str | None = None,
):
    """Convenience: load the bundled BPS crust and join at rho_min_core."""
    bps = read_bps_table(bps_path or DEFAULT_BPS_PATH)
    return build_hybrid_eos_from_tables(bps, core_table, rho_min_core=rho_min_core)


def build_core_eos_table_from_functions(
    rho_grid: list[float],
    epsilon_func: Callable,
    pressure_func: Callable,
):
    out = []
    for rho_b in rho_grid:
        rho_n = rho_b / 2.0
        rho_p = rho_b / 2.0
        out.append({
            'rho_b': rho_b,
            'epsilon_total_mev_fm3': float(epsilon_func(rho_n, rho_p)),
            'pressure_total_mev_fm3': float(pressure_func(rho_n, rho_p)),
        })
    return out
