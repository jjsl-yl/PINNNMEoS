from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable
import math

import numpy as np
from scipy.optimize import brentq, fsolve

HBARC = 197.326968
PI = math.pi
M_E = 0.511


@dataclass
class ElectronEOSPoint:
    rho_e: float
    kf_e: float
    mu_e: float
    epsilon_e_mev_fm3: float
    pressure_e_mev_fm3: float


@dataclass
class BetaEquilibriumEOSPoint:
    rho_b: float
    rho_n: float
    rho_p: float
    rho_e: float
    mu_n: float
    mu_p: float
    mu_e: float
    delta_beta: float
    charge_neutrality: float
    epsilon_baryon_mev_fm3: float
    pressure_baryon_mev_fm3: float
    epsilon_electron_mev_fm3: float
    pressure_electron_mev_fm3: float
    epsilon_total_mev_fm3: float
    pressure_total_mev_fm3: float

    def to_dict(self):
        return asdict(self)


def kf_from_rho_fm3_single_species(rho_fm3: float) -> float:
    rho_nat = rho_fm3 * HBARC**3
    return 0.0 if rho_nat <= 0.0 else (3.0 * PI**2 * rho_nat) ** (1.0 / 3.0)


def electron_chemical_potential(rho_e: float, m_e: float = M_E) -> float:
    kf = kf_from_rho_fm3_single_species(rho_e)
    return math.sqrt(kf * kf + m_e * m_e)


def _free_fermion_energy_density_nat(kf: float, mass: float) -> float:
    if kf <= 0.0:
        return 0.0
    t = kf / max(mass, 1e-12)
    root = math.sqrt(1.0 + t * t)
    a = t * root * (1.0 + 2.0 * t * t) - math.asinh(t)
    return (mass**4) / (8.0 * PI**2) * a


def _free_fermion_pressure_nat(kf: float, mass: float) -> float:
    if kf <= 0.0:
        return 0.0
    mu = math.sqrt(kf * kf + mass * mass)
    eps = _free_fermion_energy_density_nat(kf, mass)
    rho_nat = kf**3 / (3.0 * PI**2)
    return rho_nat * mu - eps


def electron_energy_density_mev_fm3(rho_e: float, m_e: float = M_E) -> float:
    kf = kf_from_rho_fm3_single_species(rho_e)
    return _free_fermion_energy_density_nat(kf, m_e) / (HBARC**3)


def electron_pressure_mev_fm3(rho_e: float, m_e: float = M_E) -> float:
    kf = kf_from_rho_fm3_single_species(rho_e)
    return _free_fermion_pressure_nat(kf, m_e) / (HBARC**3)


def _d_eps_drho_n(epsilon_func: Callable, rho_n: float, rho_p: float, h: float = 1e-5) -> float:
    return (epsilon_func(rho_n + h, rho_p) - epsilon_func(rho_n - h, rho_p)) / (2.0 * h)


def _d_eps_drho_p(epsilon_func: Callable, rho_n: float, rho_p: float, h: float = 1e-5) -> float:
    return (epsilon_func(rho_n, rho_p + h) - epsilon_func(rho_n, rho_p - h)) / (2.0 * h)


def solve_beta_equilibrium_point_numeric(
    compiled_model,
    params: dict,
    rho_b: float,
    *,
    proton_fraction_guess: float = 0.1,
    field_guess=None,
    max_iter: int = 120,
    tol: float = 1e-8,
    m_e: float = M_E,
):
    from NMtoolsV1.numeric_layer import kf_from_rho_fm3, rho_s_one_species

    field_names = list(getattr(compiled_model, 'field_names', []))
    n_fields = len(field_names)
    if field_guess is None:
        field_guess = [0.0] * n_fields
        alias_map = {
            'sigma': ['sigma', 's'],
            'omega': ['omega', 'o'],
            'rho': ['rho', 'r'],
            'delta': ['delta', 'a'],
        }
        idx = {name: i for i, name in enumerate(field_names)}
        def _first_index(keys):
            for k in keys:
                if k in idx:
                    return idx[k]
            return None
        i_sigma = _first_index(alias_map['sigma'])
        i_omega = _first_index(alias_map['omega'])
        i_rho = _first_index(alias_map['rho'])
        i_delta = _first_index(alias_map['delta'])
        if i_sigma is not None:
            field_guess[i_sigma] = 20.0
        if i_omega is not None:
            field_guess[i_omega] = 15.0
        if i_rho is not None:
            field_guess[i_rho] = 0.0
        if i_delta is not None:
            field_guess[i_delta] = 0.0
    field_guess = np.array(field_guess, dtype=float)
    g_omega = float(params.get('g_omega', 0.0))
    g_rho = float(params.get('g_rho', 0.0))

    def solve_fields(rho_n: float, rho_p: float, x0):
        fn = lambda xx: compiled_model.eom_numeric(xx, rho_n, rho_p)
        sol = fsolve(fn, x0, xtol=1e-10, maxfev=20000)
        res = compiled_model.eom_numeric(sol, rho_n, rho_p)
        return np.array(sol, dtype=float), np.array(res, dtype=float)

    alias_map = {
        'sigma': ['sigma', 's'],
        'omega': ['omega', 'o'],
        'rho': ['rho', 'r'],
        'delta': ['delta', 'a'],
    }
    idx = {name: i for i, name in enumerate(field_names)}

    def _field_value(sol, aliases, default=0.0):
        for name in aliases:
            if name in idx:
                return float(sol[idx[name]])
        return float(default)

    def state_from_solution(sol, rho_n: float, rho_p: float):
        mp = compiled_model.mass_p_numeric(sol)
        mn = compiled_model.mass_n_numeric(sol)
        kfp = kf_from_rho_fm3(rho_p)
        kfn = kf_from_rho_fm3(rho_n)
        o = _field_value(sol, alias_map['omega'])
        r = _field_value(sol, alias_map['rho'])
        mu_p = math.sqrt(kfp * kfp + mp * mp) + g_omega * o + g_rho * r
        mu_n = math.sqrt(kfn * kfn + mn * mn) + g_omega * o - g_rho * r
        rho_sp = rho_s_one_species(kfp, mp)
        rho_sn = rho_s_one_species(kfn, mn)
        return {
            'mp': float(mp),
            'mn': float(mn),
            'kfp': float(kfp),
            'kfn': float(kfn),
            'mu_p': float(mu_p),
            'mu_n': float(mu_n),
            'rho_sp': float(rho_sp),
            'rho_sn': float(rho_sn),
        }

    def meson_energy_from_fields(sol):
        s = _field_value(sol, alias_map['sigma'])
        o = _field_value(sol, alias_map['omega'])
        r = _field_value(sol, alias_map['rho'])
        a = _field_value(sol, alias_map['delta'])
        return (
            0.5 * params.get('m_sigma', 0.0)**2 * s**2
            + (1.0 / 3.0) * params.get('g2', 0.0) * s**3
            + 0.25 * params.get('g3', 0.0) * s**4
            + 0.5 * params.get('g4', 0.0) * s**2 * o**2
            + 0.5 * params.get('g5', 0.0) * s**2 * r**2
            + params.get('g6', 0.0) * s * r**2
            + params.get('g7', 0.0) * s * o**2
            - 0.5 * params.get('m_omega', 0.0)**2 * o**2
            - 0.25 * params.get('c3', 0.0) * o**4
            - 0.5 * params.get('c4', 0.0) * o**2 * r**2
            - 0.5 * params.get('m_rho', 0.0)**2 * r**2
            - 0.25 * params.get('d3', 0.0) * r**4
            + 0.5 * params.get('m_a0', 0.0)**2 * a**2
            - 0.25 * params.get('b3', 0.0) * a**4
            - 0.5 * params.get('b4', 0.0) * a**2 * r**2
            - 0.5 * params.get('b5', 0.0) * s * a**2
            - 0.5 * params.get('b6', 0.0) * a**2 * s**2
            - 0.5 * params.get('b7', 0.0) * a**2 * o**2
            - params.get('b8', 0.0) * a * r * o * s
            - params.get('b9', 0.0) * a * r * o
        ) / (HBARC**3)

    def fermion_energy_density_one_species_mev_fm3(kf: float, mass: float) -> float:
        return _free_fermion_energy_density_nat(kf, mass) / (HBARC**3)

    def beta_residual(yp: float, x0):
        yp = min(max(float(yp), 1e-8), 0.5 - 1e-8)
        rho_p = rho_b * yp
        rho_n = rho_b - rho_p
        sol, eom = solve_fields(rho_n, rho_p, x0)
        st = state_from_solution(sol, rho_n, rho_p)
        mu_e = electron_chemical_potential(rho_p, m_e=m_e)
        return float(st['mu_n'] - st['mu_p'] - mu_e), rho_n, rho_p, sol, eom, st, mu_e

    yp_lo = 1e-8
    yp_hi = 0.5 - 1e-8
    f_lo, rho_n_lo, rho_p_lo, sol_lo, eom_lo, st_lo, mu_e_lo = beta_residual(yp_lo, field_guess)
    f_hi, rho_n_hi, rho_p_hi, sol_hi, eom_hi, st_hi, mu_e_hi = beta_residual(yp_hi, sol_lo)

    if f_lo == 0.0:
        yp, rho_n, rho_p, sol, eom, st, mu_e = yp_lo, rho_n_lo, rho_p_lo, sol_lo, eom_lo, st_lo, mu_e_lo
    elif f_hi == 0.0:
        yp, rho_n, rho_p, sol, eom, st, mu_e = yp_hi, rho_n_hi, rho_p_hi, sol_hi, eom_hi, st_hi, mu_e_hi
    elif f_lo * f_hi > 0.0:
        samples = []
        x_guess = field_guess
        for yp_try in np.linspace(yp_lo, yp_hi, 81):
            ff, rn, rp, ss, ee, stt, me = beta_residual(float(yp_try), x_guess)
            x_guess = ss
            samples.append((abs(ff), float(yp_try), rn, rp, ss, ee, stt, me, ff))
        _, yp, rho_n, rho_p, sol, eom, st, mu_e, f_mid = min(samples, key=lambda x: x[0])
        if abs(f_mid) > 1e-4:
            raise RuntimeError(f'beta equilibrium root not bracketed at rho_b={rho_b}; best residual={f_mid}')
    else:
        def scalar_residual(y):
            return beta_residual(float(y), field_guess)[0]
        yp = brentq(scalar_residual, yp_lo, yp_hi, xtol=tol, rtol=tol, maxiter=max_iter)
        _, rho_n, rho_p, sol, eom, st, mu_e = beta_residual(yp, field_guess)

    eps_b = float(compiled_model.energy_numeric(sol, rho_n, rho_p, params))
    p_b = rho_n * st['mu_n'] + rho_p * st['mu_p'] - eps_b
    rho_e = rho_p
    eps_e = electron_energy_density_mev_fm3(rho_e, m_e=m_e)
    p_e = electron_pressure_mev_fm3(rho_e, m_e=m_e)
    eps_tot = eps_b + eps_e
    p_tot = rho_n * st['mu_n'] + rho_p * st['mu_p'] + rho_e * mu_e - eps_tot
    point = BetaEquilibriumEOSPoint(
        rho_b=float(rho_b),
        rho_n=float(rho_n),
        rho_p=float(rho_p),
        rho_e=float(rho_e),
        mu_n=float(st['mu_n']),
        mu_p=float(st['mu_p']),
        mu_e=float(mu_e),
        delta_beta=float(st['mu_n'] - st['mu_p'] - mu_e),
        charge_neutrality=float(rho_p - rho_e),
        epsilon_baryon_mev_fm3=float(eps_b),
        pressure_baryon_mev_fm3=float(p_b),
        epsilon_electron_mev_fm3=float(eps_e),
        pressure_electron_mev_fm3=float(p_e),
        epsilon_total_mev_fm3=float(eps_tot),
        pressure_total_mev_fm3=float(p_tot),
    )
    return point, sol


def solve_beta_equilibrium_point(
    epsilon_func: Callable,
    pressure_func: Callable,
    rho_b: float,
    *,
    proton_fraction_guess: float = 0.1,
    max_iter: int = 120,
    tol: float = 1e-8,
    m_e: float = M_E,
    compiled_model=None,
    params: dict | None = None,
    field_guess=None,
):
    if compiled_model is not None and params is not None:
        point, _ = solve_beta_equilibrium_point_numeric(
            compiled_model,
            params,
            rho_b,
            proton_fraction_guess=proton_fraction_guess,
            field_guess=field_guess,
            max_iter=max_iter,
            tol=tol,
            m_e=m_e,
        )
        return point

    yp_lo = 1e-8
    yp_hi = 0.5 - 1e-8

    def residual(yp: float):
        rho_p = rho_b * yp
        rho_n = rho_b - rho_p
        mu_n = _d_eps_drho_n(epsilon_func, rho_n, rho_p)
        mu_p = _d_eps_drho_p(epsilon_func, rho_n, rho_p)
        rho_e = rho_p
        mu_e = electron_chemical_potential(rho_e, m_e=m_e)
        return (mu_n - mu_p - mu_e), (rho_n, rho_p, rho_e, mu_n, mu_p, mu_e)

    f_lo, data_lo = residual(yp_lo)
    f_hi, data_hi = residual(yp_hi)

    if f_lo == 0.0:
        yp = yp_lo
        data = data_lo
    elif f_hi == 0.0:
        yp = yp_hi
        data = data_hi
    elif f_lo * f_hi > 0.0:
        samples = []
        for i in range(200):
            yp_try = yp_lo + (yp_hi - yp_lo) * i / 199.0
            f_try, data_try = residual(yp_try)
            samples.append((abs(f_try), yp_try, data_try, f_try))
        _, yp, data, f_mid = min(samples, key=lambda x: x[0])
        if abs(f_mid) > 1e-4:
            raise RuntimeError(f'beta equilibrium root not bracketed at rho_b={rho_b}; best residual={f_mid}')
    else:
        lo = yp_lo
        hi = yp_hi
        data = None
        yp = proton_fraction_guess
        for _ in range(max_iter):
            yp = 0.5 * (lo + hi)
            f_mid, data_mid = residual(yp)
            data = data_mid
            if abs(f_mid) < tol or abs(hi - lo) < tol:
                break
            if f_lo * f_mid <= 0.0:
                hi = yp
                f_hi = f_mid
            else:
                lo = yp
                f_lo = f_mid

    rho_n, rho_p, rho_e, mu_n, mu_p, mu_e = data
    eps_b = float(epsilon_func(rho_n, rho_p))
    p_b = float(pressure_func(rho_n, rho_p))
    eps_e = electron_energy_density_mev_fm3(rho_e, m_e=m_e)
    p_e = electron_pressure_mev_fm3(rho_e, m_e=m_e)
    eps_tot = eps_b + eps_e
    p_tot = rho_n * mu_n + rho_p * mu_p + rho_e * mu_e - eps_tot
    return BetaEquilibriumEOSPoint(
        rho_b=rho_b,
        rho_n=rho_n,
        rho_p=rho_p,
        rho_e=rho_e,
        mu_n=mu_n,
        mu_p=mu_p,
        mu_e=mu_e,
        delta_beta=mu_n - mu_p - mu_e,
        charge_neutrality=rho_p - rho_e,
        epsilon_baryon_mev_fm3=eps_b,
        pressure_baryon_mev_fm3=p_b,
        epsilon_electron_mev_fm3=eps_e,
        pressure_electron_mev_fm3=p_e,
        epsilon_total_mev_fm3=eps_tot,
        pressure_total_mev_fm3=p_tot,
    )


def build_beta_equilibrium_eos_table(
    epsilon_func: Callable,
    pressure_func: Callable,
    rho_grid: list[float],
    *,
    proton_fraction_guess: float = 0.1,
    m_e: float = M_E,
    compiled_model=None,
    params: dict | None = None,
    field_guess=None,
):
    out = []
    yp_guess = proton_fraction_guess
    x_guess = field_guess
    for rho_b in rho_grid:
        if compiled_model is not None and params is not None:
            point, x_guess = solve_beta_equilibrium_point_numeric(
                compiled_model,
                params,
                rho_b,
                proton_fraction_guess=yp_guess,
                field_guess=x_guess,
                m_e=m_e,
            )
        else:
            point = solve_beta_equilibrium_point(
                epsilon_func,
                pressure_func,
                rho_b,
                proton_fraction_guess=yp_guess,
                m_e=m_e,
            )
        yp_guess = point.rho_p / point.rho_b if point.rho_b > 0.0 else yp_guess
        out.append(point)
    return out
