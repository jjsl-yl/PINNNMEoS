from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import bisect

MEVFM3_TO_KM2 = 1.3234e-6
KM2_TO_MEVFM3 = 1.0 / MEVFM3_TO_KM2
MODOT_TO_KM = 1.4766
KM_TO_MODOT = 1.0 / MODOT_TO_KM


@dataclass
class TOVPoint:
    central_pressure_mev_fm3: float
    mass_g: float
    mass_msun: float
    radius_cm: float
    radius_km: float

    def to_dict(self):
        return asdict(self)


@dataclass
class MRCurve:
    central_pressures_mev_fm3: list[float]
    masses_msun: list[float]
    radii_km: list[float]

    def to_dict(self):
        return asdict(self)


class EOSInterpolator:
    def __init__(self, pressures_mev_fm3: list[float], energy_densities_mev_fm3: list[float]):
        pairs = sorted(zip(pressures_mev_fm3, energy_densities_mev_fm3), key=lambda x: x[0])
        self.pressures_mev = [p for p, _ in pairs]
        self.energies_mev = [e for _, e in pairs]
        self.pressures_km2 = [p * MEVFM3_TO_KM2 for p in self.pressures_mev]
        self.energies_km2 = [e * MEVFM3_TO_KM2 for e in self.energies_mev]
        if len(self.pressures_mev) < 2:
            raise ValueError('EOS table needs at least two points')

    def energy_density_from_pressure_km2(self, p_km2: float) -> float:
        ps = self.pressures_km2
        es = self.energies_km2
        if p_km2 <= ps[0]:
            i = 0
        elif p_km2 >= ps[-1]:
            i = len(ps) - 2
        else:
            i = bisect.bisect_right(ps, p_km2) - 1
        p1, p2 = ps[i], ps[i + 1]
        e1, e2 = es[i], es[i + 1]
        if p2 == p1:
            return e1
        t = (p_km2 - p1) / (p2 - p1)
        return e1 + t * (e2 - e1)


def build_eos_interpolator(eos_table: list[dict] | list[object]):
    pressures = []
    energies = []
    for row in eos_table:
        if isinstance(row, dict):
            pressures.append(float(row['pressure_total_mev_fm3']))
            energies.append(float(row['epsilon_total_mev_fm3']))
        else:
            pressures.append(float(row.pressure_total_mev_fm3))
            energies.append(float(row.epsilon_total_mev_fm3))
    return EOSInterpolator(pressures, energies)


def _tov_rhs(r_km: float, m_km: float, p_km2: float, eos: EOSInterpolator):
    if p_km2 <= 0.0:
        return 0.0, 0.0
    eps_km2 = eos.energy_density_from_pressure_km2(p_km2)
    dm_dr = 4.0 * math.pi * r_km * r_km * eps_km2
    dp_dr = - (eps_km2 + p_km2) * (m_km + 4.0 * math.pi * r_km**3 * p_km2) / (r_km * (r_km - 2.0 * m_km))
    return dm_dr, dp_dr


def integrate_tov_star(
    eos: EOSInterpolator,
    central_pressure_mev_fm3: float,
    *,
    dr_km: float = 0.001,
    p_surface_mev_fm3: float = 1e-20,
    r0_km: float = 1e-6,
):
    p = central_pressure_mev_fm3 * MEVFM3_TO_KM2
    p_surface = p_surface_mev_fm3 * MEVFM3_TO_KM2
    eps0_km2 = eos.energy_density_from_pressure_km2(p)
    r = r0_km
    m = 4.0 * math.pi * r**3 * eps0_km2 / 3.0

    while p > p_surface and r < 100.0:
        k1_m, k1_p = _tov_rhs(r, m, p, eos)
        k2_m, k2_p = _tov_rhs(r + 0.5 * dr_km, m + 0.5 * dr_km * k1_m, p + 0.5 * dr_km * k1_p, eos)
        k3_m, k3_p = _tov_rhs(r + 0.5 * dr_km, m + 0.5 * dr_km * k2_m, p + 0.5 * dr_km * k2_p, eos)
        k4_m, k4_p = _tov_rhs(r + dr_km, m + dr_km * k3_m, p + dr_km * k3_p, eos)
        m += (dr_km / 6.0) * (k1_m + 2.0 * k2_m + 2.0 * k3_m + k4_m)
        p += (dr_km / 6.0) * (k1_p + 2.0 * k2_p + 2.0 * k3_p + k4_p)
        r += dr_km
        if not math.isfinite(m) or not math.isfinite(p):
            break
        if 2.0 * m / r >= 0.999:
            break

    return TOVPoint(
        central_pressure_mev_fm3=central_pressure_mev_fm3,
        mass_g=m,
        mass_msun=m * KM_TO_MODOT,
        radius_cm=r,
        radius_km=r,
    )


def build_mr_curve(
    eos: EOSInterpolator,
    central_pressures_mev_fm3: list[float],
    *,
    dr_km: float = 0.001,
    p_surface_mev_fm3: float = 1e-20,
    progress: bool = False,
):
    masses = []
    radii = []
    for i, pc in enumerate(central_pressures_mev_fm3, start=1):
        star = integrate_tov_star(eos, pc, dr_km=dr_km, p_surface_mev_fm3=p_surface_mev_fm3)
        masses.append(star.mass_msun)
        radii.append(star.radius_km)
        if progress:
            print(f'MR point {i}/{len(central_pressures_mev_fm3)}: Pc={pc:.6f} MeV/fm^3, M={star.mass_msun:.6f} Msun, R={star.radius_km:.6f} km')
    return MRCurve(
        central_pressures_mev_fm3=list(central_pressures_mev_fm3),
        masses_msun=masses,
        radii_km=radii,
    )
