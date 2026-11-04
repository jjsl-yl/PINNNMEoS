from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from numeric_layer import evaluate_numeric_model


@dataclass
class DensityPointResult:
    rho_p: float
    rho_n: float
    energy_density: float
    binding_energy: float


def epsilon_function_factory(analytic_model, params: dict, initial_guess=None):
    cache = {
        'sym': list(initial_guess or [35.0, 22.0, 0.0]),
        'asym': list(initial_guess or [35.0, 22.0, 0.0]),
    }

    def epsilon_func(rho_n: float, rho_p: float):
        key = 'sym' if abs(rho_n - rho_p) < 1e-12 else 'asym'
        result = evaluate_numeric_model(analytic_model, params, rho_p=rho_p, rho_n=rho_n, initial_guess=cache[key])
        cache[key] = [result.field_solution[name] for name in analytic_model.fields.keys()]
        return result.energy_density_mev_fm3

    return epsilon_func


def binding_energy_from_epsilon(epsilon_func: Callable, rho_n: float, rho_p: float, m_N: float):
    rho_B = rho_n + rho_p
    return epsilon_func(rho_n, rho_p) / rho_B - m_N


def pressure_from_epsilon(epsilon_func: Callable, rho_n: float, rho_p: float, h=1e-4):
    mu_n = (epsilon_func(rho_n + h, rho_p) - epsilon_func(rho_n - h, rho_p)) / (2.0 * h)
    mu_p = (epsilon_func(rho_n, rho_p + h) - epsilon_func(rho_n, rho_p - h)) / (2.0 * h)
    eps = epsilon_func(rho_n, rho_p)
    return rho_n * mu_n + rho_p * mu_p - eps


def compressibility_from_epsilon(epsilon_func: Callable, rho_B: float, m_N: float, h=1e-4):
    r0 = rho_B
    rm = max(r0 - h, 1e-8)
    rp = r0 + h

    em = epsilon_func(rm / 2.0, rm / 2.0) / rm - m_N
    e0 = epsilon_func(r0 / 2.0, r0 / 2.0) / r0 - m_N
    ep = epsilon_func(rp / 2.0, rp / 2.0) / rp - m_N

    first = (ep - em) / (rp - rm)
    second = (ep - 2.0 * e0 + em) / (h * h)
    return 9.0 * (2.0 * rho_B * first + rho_B * rho_B * second)


def symmetry_energy_from_epsilon(epsilon_func: Callable, rho_B: float, m_N: float, h_delta=1e-3):
    d = h_delta
    rp1 = rho_B * (1.0 - d) / 2.0
    rn1 = rho_B * (1.0 + d) / 2.0
    rp2 = rho_B * (1.0 + d) / 2.0
    rn2 = rho_B * (1.0 - d) / 2.0
    e0 = epsilon_func(rho_B / 2.0, rho_B / 2.0) / rho_B - m_N
    e1 = epsilon_func(rn1, rp1) / rho_B - m_N
    e2 = epsilon_func(rn2, rp2) / rho_B - m_N
    return (e1 - 2.0 * e0 + e2) / (2.0 * d * d)


def symmetry_slope_from_epsilon(epsilon_func: Callable, rho_B: float, m_N: float, h_rho=1e-4, h_delta=1e-3):
    rm = max(rho_B - h_rho, 1e-8)
    rp = rho_B + h_rho
    esm = symmetry_energy_from_epsilon(epsilon_func, rm, m_N, h_delta=h_delta)
    esp = symmetry_energy_from_epsilon(epsilon_func, rp, m_N, h_delta=h_delta)
    dEs = (esp - esm) / (rp - rm)
    return 3.0 * rho_B * dEs
