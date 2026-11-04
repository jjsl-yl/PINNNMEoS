from __future__ import annotations

from dataclasses import dataclass
import sympy as sp
import numpy as np

from NMtoolsV1.numeric_layer import HBARC, kf_from_rho_fm3, rho_s_one_species


@dataclass
class CompiledBetaModel:
    field_names: list[str]
    eom_numeric: callable
    mass_p_numeric: callable
    mass_n_numeric: callable
    energy_numeric: callable


def compile_beta_effective_model(analytic, params: dict) -> CompiledBetaModel:
    """
    Build pure numeric residual/energy functions directly from analytic structure.

    Output interface is model-independent:
      eom_numeric(x, rho_n, rho_p, params_override=None) -> np.ndarray
      mass_p_numeric(x, params_override=None) -> float
      mass_n_numeric(x, params_override=None) -> float
      energy_numeric(x, rho_n, rho_p, params_override=None) -> float
    """
    field_names = list(analytic.fields.keys())
    field_symbols = [analytic.fields[name] for name in field_names]
    param_symbols = list(analytic.params.values())

    rho_s_sym = sp.Symbol('rho_s', real=True)
    rho_B_sym = sp.Symbol('rho_B', real=True)
    rho_3_sym = sp.Symbol('rho_3', real=True)
    rho_s3_sym = sp.Symbol('rho_s3', real=True)

    eom_funcs = []
    for name in field_names:
        expr = analytic.eom[name].subs({
            analytic.extra_symbols['psibar_psi']: rho_s_sym,
            analytic.extra_symbols['psibar_gamma0_psi']: rho_B_sym,
            analytic.extra_symbols['psibar_gamma0_tau3_psi']: rho_3_sym,
            analytic.extra_symbols['psibar_tau3_psi']: rho_s3_sym,
        })
        fn = sp.lambdify(field_symbols + param_symbols + [rho_s_sym, rho_B_sym, rho_3_sym, rho_s3_sym], expr, 'numpy')
        eom_funcs.append(fn)

    mp_fn = sp.lambdify(field_symbols + param_symbols, analytic.effective_mass_p_expr, 'numpy')
    mn_fn = sp.lambdify(field_symbols + param_symbols, analytic.effective_mass_n_expr, 'numpy')

    eps_fermion_sym = sp.Symbol('eps_fermion', real=True)
    energy_term_fns = []
    for expr, _label in analytic.energy_density_terms:
        fn = sp.lambdify(field_symbols + param_symbols + [eps_fermion_sym, rho_B_sym, rho_3_sym], expr, 'numpy')
        energy_term_fns.append(fn)

    default_param_values = [params[sym.name] for sym in param_symbols]

    def _param_values(params_override=None):
        p = params if params_override is None else params_override
        return [p[sym.name] for sym in param_symbols]

    def mass_p_numeric(x, params_override=None):
        vals = list(map(float, x)) + _param_values(params_override)
        return float(mp_fn(*vals))

    def mass_n_numeric(x, params_override=None):
        vals = list(map(float, x)) + _param_values(params_override)
        return float(mn_fn(*vals))

    def _densities_from_x(x, rho_n, rho_p, params_override=None):
        mp = mass_p_numeric(x, params_override)
        mn = mass_n_numeric(x, params_override)
        kfp = kf_from_rho_fm3(rho_p)
        kfn = kf_from_rho_fm3(rho_n)
        rho_sp = rho_s_one_species(kfp, mp)
        rho_sn = rho_s_one_species(kfn, mn)
        rho_s = rho_sp + rho_sn
        rho_s3 = rho_sp - rho_sn
        rho_B = (rho_p + rho_n) * HBARC**3
        rho_3 = (rho_p - rho_n) * HBARC**3
        return rho_s, rho_B, rho_3, rho_s3, mp, mn, kfp, kfn

    def eom_numeric(x, rho_n, rho_p, params_override=None):
        x = list(map(float, x))
        param_vals = _param_values(params_override)
        rho_s, rho_B, rho_3, rho_s3, *_ = _densities_from_x(x, rho_n, rho_p, params_override)
        args = x + param_vals + [rho_s, rho_B, rho_3, rho_s3]
        return np.array([float(fn(*args)) for fn in eom_funcs], dtype=float)

    def energy_numeric(x, rho_n, rho_p, params_override=None):
        x = list(map(float, x))
        rho_s, rho_B, rho_3, rho_s3, mp, mn, kfp, kfn = _densities_from_x(x, rho_n, rho_p, params_override)
        param_vals = _param_values(params_override)
        try:
            from NMtoolsV1.numeric_layer import fermion_energy_density_one_species
            eps_fp_nat = fermion_energy_density_one_species(kfp, mp)
            eps_fn_nat = fermion_energy_density_one_species(kfn, mn)
        except ImportError:
            from NMtoolsV1.numeric_layer import eps_one_species_from_meff
            eps_fp_nat = eps_one_species_from_meff(kfp, mp)
            eps_fn_nat = eps_one_species_from_meff(kfn, mn)
        eps_f_nat = float(eps_fp_nat + eps_fn_nat)
        total_nat = 0.0
        args = x + param_vals + [eps_f_nat, rho_B, rho_3]
        for fn in energy_term_fns:
            total_nat += float(fn(*args))
        return float(total_nat / (HBARC**3))

    return CompiledBetaModel(
        field_names=field_names,
        eom_numeric=eom_numeric,
        mass_p_numeric=mass_p_numeric,
        mass_n_numeric=mass_n_numeric,
        energy_numeric=energy_numeric,
    )
