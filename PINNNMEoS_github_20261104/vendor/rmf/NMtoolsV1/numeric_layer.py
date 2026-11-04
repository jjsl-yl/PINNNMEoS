from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import sympy as sp
from scipy.optimize import fsolve

HBARC = 197.326968
PI = np.pi


@dataclass
class NumericResult:
    field_solution: dict
    effective_masses: tuple
    scalar_densities: tuple
    energy_term_values: list
    energy_density_natural: float
    energy_density_mev_fm3: float


def kf_from_rho_fm3(rho_fm3: float) -> float:
    rho_nat = rho_fm3 * HBARC**3
    return 0.0 if rho_nat <= 0.0 else (3.0 * PI**2 * rho_nat) ** (1.0 / 3.0)


def rho_s_one_species(kf: float, m_eff: float) -> float:
    if kf <= 0.0:
        return 0.0
    t = kf / max(m_eff, 1e-12)
    return m_eff**3 / PI**2 * 0.5 * (t * np.sqrt(1.0 + t * t) - np.arcsinh(t))


def eps_one_species_from_meff(kf: float, m_eff: float) -> float:
    if kf <= 0.0:
        return 0.0
    t = kf / max(m_eff, 1e-12)
    a = t * np.sqrt(1.0 + t * t) * (1.0 + 2.0 * t * t) - np.arcsinh(t)
    return (m_eff**4) / (8.0 * PI**2) * a


def solve_fields(analytic_model, params: dict, rho_p: float, rho_n: float, initial_guess: list):
    field_names = list(analytic_model.fields.keys())
    field_symbols = [analytic_model.fields[name] for name in field_names]
    param_symbols = list(analytic_model.params.values())
    rho_s_sym = sp.Symbol('rho_s', real=True)
    rho_B_sym = sp.Symbol('rho_B', real=True)
    rho_3_sym = sp.Symbol('rho_3', real=True)
    rho_s3_sym = sp.Symbol('rho_s3', real=True)
    eom_funcs = []
    for name in field_names:
        expr = analytic_model.eom[name]
        expr = expr.subs({
            analytic_model.extra_symbols['psibar_psi']: rho_s_sym,
            analytic_model.extra_symbols['psibar_gamma0_psi']: rho_B_sym,
            analytic_model.extra_symbols['psibar_gamma0_tau3_psi']: rho_3_sym,
            analytic_model.extra_symbols.get('psibar_tau3_psi', sp.Symbol('psibar_tau3_psi', real=True)): rho_s3_sym,
        })
        fn = sp.lambdify(field_symbols + param_symbols + [rho_s_sym, rho_B_sym, rho_3_sym, rho_s3_sym], expr, 'numpy')
        eom_funcs.append(fn)

    def residual(x):
        sol = {name: float(val) for name, val in zip(field_names, x)}
        subs = {analytic_model.fields[k]: v for k, v in sol.items()}
        subs.update({sym: params[sym.name] for sym in analytic_model.params.values()})
        m0 = float(sp.N(analytic_model.effective_mass_expr.subs(subs)))
        m3 = float(sp.N(analytic_model.isovector_scalar_mass_expr.subs(subs)))
        mp = m0 + m3
        mn = m0 - m3
        kfp = kf_from_rho_fm3(rho_p)
        kfn = kf_from_rho_fm3(rho_n)
        rho_sp = rho_s_one_species(kfp, mp)
        rho_sn = rho_s_one_species(kfn, mn)
        rho_s = rho_sp + rho_sn
        rho_s3 = rho_sp - rho_sn
        rho_B = (rho_p + rho_n) * HBARC**3
        rho_3 = (rho_p - rho_n) * HBARC**3
        args = list(x) + [params[s.name] for s in param_symbols] + [rho_s, rho_B, rho_3, rho_s3]
        return np.array([fn(*args) for fn in eom_funcs], float)

    sol = fsolve(residual, np.array(initial_guess, float), xtol=1e-10, maxfev=20000)
    return {name: float(val) for name, val in zip(field_names, sol)}


def evaluate_numeric_model(analytic_model, params: dict, rho_p: float, rho_n: float, initial_guess: list):
    sol = solve_fields(analytic_model, params, rho_p, rho_n, initial_guess)
    subs = {analytic_model.fields[k]: v for k, v in sol.items()}
    subs.update({sym: params[sym.name] for sym in analytic_model.params.values()})

    m0 = float(sp.N(analytic_model.effective_mass_expr.subs(subs)))
    m3 = float(sp.N(analytic_model.isovector_scalar_mass_expr.subs(subs)))
    mp, mn = m0 + m3, m0 - m3
    kfp = kf_from_rho_fm3(rho_p)
    kfn = kf_from_rho_fm3(rho_n)
    rho_sp = rho_s_one_species(kfp, mp)
    rho_sn = rho_s_one_species(kfn, mn)
    rho_s = rho_sp + rho_sn
    rho_B = (rho_p + rho_n) * HBARC**3
    rho_3 = (rho_p - rho_n) * HBARC**3
    eps_p = eps_one_species_from_meff(kfp, mp)
    eps_n = eps_one_species_from_meff(kfn, mn)
    eps_f = eps_p + eps_n

    value_subs = dict(subs)
    value_subs[sp.Symbol('eps_fermion', real=True)] = eps_f
    value_subs[sp.Symbol('rho_B', real=True)] = rho_B
    value_subs[sp.Symbol('rho_3', real=True)] = rho_3

    term_values = []
    total_natural = 0.0
    for expr, label in analytic_model.energy_density_terms:
        val = float(sp.N(expr.subs(value_subs)))
        term_values.append((label, val))
        total_natural += val

    return NumericResult(
        field_solution=sol,
        effective_masses=(mp, mn),
        scalar_densities=(rho_sp, rho_sn, rho_s),
        energy_term_values=term_values,
        energy_density_natural=total_natural,
        energy_density_mev_fm3=total_natural / (HBARC**3),
    )
