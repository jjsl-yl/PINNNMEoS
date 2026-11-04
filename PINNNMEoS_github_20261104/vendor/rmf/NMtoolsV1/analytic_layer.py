from __future__ import annotations

from dataclasses import dataclass
import ast
import sympy as sp


@dataclass
class AnalyticModel:
    params: dict
    fields: dict
    extra_symbols: dict
    term_exprs: list
    total_lagrangian: sp.Expr
    eom: dict
    effective_mass_expr: sp.Expr
    isovector_scalar_mass_expr: sp.Expr
    effective_mass_p_expr: sp.Expr
    effective_mass_n_expr: sp.Expr
    energy_density_terms: list
    energy_density_expr: sp.Expr


def _symbol_name_from_param_key(key: str) -> str:
    if key.startswith("P['") and key.endswith("']"):
        return key[3:-2]
    if key.startswith('P["') and key.endswith('"]'):
        return key[3:-2]
    return key


def build_symbol_table(param_keys, field_names, extra_symbol_names=None):
    params = {}
    p_dict = {}
    for key in param_keys:
        sym_name = _symbol_name_from_param_key(key)
        sym = sp.Symbol(sym_name, real=True)
        params[key] = sym
        p_dict[sym_name] = sym
    fields = {name: sp.Symbol(name, real=True) for name in field_names}
    extra_symbols = {name: sp.Symbol(name, real=True) for name in (extra_symbol_names or [])}
    namespace = {'P': p_dict, 'sp': sp}
    namespace.update(fields)
    namespace.update(extra_symbols)
    return params, fields, extra_symbols, namespace


def compile_terms(term_texts, namespace):
    out = []
    for text in term_texts:
        expr = ast.parse(text, mode='eval')
        code = compile(expr, '<NMtoolsV1-term>', 'eval')
        out.append(sp.expand(eval(code, {'__builtins__': {}, 'sp': sp}, namespace)))
    return out


def total_lagrangian_from_terms(term_exprs):
    return sp.expand(sum(term_exprs))


def mean_field_eom_from_lagrangian(total_lagrangian: sp.Expr, fields: dict, field_order: list):
    return {name: sp.expand(sp.diff(total_lagrangian, fields[name])) for name in field_order}


def effective_mass_from_lagrangian(total_lagrangian: sp.Expr, scalar_bilinear_symbol: sp.Symbol) -> sp.Expr:
    coeff = sp.expand(total_lagrangian.coeff(scalar_bilinear_symbol, 1))
    return sp.expand(-coeff)


def isovector_scalar_mass_from_lagrangian(total_lagrangian: sp.Expr, tau3_scalar_symbol: sp.Symbol) -> sp.Expr:
    coeff = sp.expand(total_lagrangian.coeff(tau3_scalar_symbol, 1))
    return sp.expand(-coeff)


def energy_density_terms_from_lagrangian(total_lagrangian: sp.Expr, extra_symbols: dict, fields: dict, field_order: list):
    psibar_i_dslash = extra_symbols['psibar_i_dslash_psi']
    psibar_psi = extra_symbols['psibar_psi']
    psibar_gamma0 = extra_symbols['psibar_gamma0_psi']
    psibar_gamma0_tau3 = extra_symbols['psibar_gamma0_tau3_psi']
    psibar_tau3 = extra_symbols.get('psibar_tau3_psi', sp.Symbol('psibar_tau3_psi', real=True))

    expr = sp.expand(total_lagrangian)
    all_terms = sp.Add.make_args(expr)
    field_syms = [fields[name] for name in field_order]

    meson_terms = []
    scalar_coeff = sp.expand(expr.coeff(psibar_psi, 1))
    kinetic_coeff = sp.expand(expr.coeff(psibar_i_dslash, 1))
    vector_coeff = sp.expand(expr.coeff(psibar_gamma0, 1))
    isovector_coeff = sp.expand(expr.coeff(psibar_gamma0_tau3, 1))

    for term in all_terms:
        if any(sym in term.free_symbols for sym in [psibar_i_dslash, psibar_psi, psibar_gamma0, psibar_gamma0_tau3, psibar_tau3]):
            continue
        meson_terms.append(term)

    energy_terms = []
    for term in meson_terms:
        energy_terms.append((-sp.expand(term), f'-({sp.sstr(term)})'))
    energy_terms.append((sp.Symbol('eps_fermion', real=True), 'eps_fermion'))
    energy_terms.append((-vector_coeff * sp.Symbol('rho_B', real=True), f'-({sp.sstr(vector_coeff)})*rho_B'))
    energy_terms.append((-isovector_coeff * sp.Symbol('rho_3', real=True), f'-({sp.sstr(isovector_coeff)})*rho_3'))

    return energy_terms


def build_analytic_model(param_keys, field_names, extra_symbol_names, lagrangian_term_texts):
    params, fields, extra_symbols, namespace = build_symbol_table(param_keys, field_names, extra_symbol_names)
    term_exprs = compile_terms(lagrangian_term_texts, namespace)
    total_lagrangian = total_lagrangian_from_terms(term_exprs)
    eom = mean_field_eom_from_lagrangian(total_lagrangian, fields, field_names)
    effective_mass_expr = effective_mass_from_lagrangian(total_lagrangian, extra_symbols['psibar_psi'])
    isovector_scalar_mass_expr = isovector_scalar_mass_from_lagrangian(total_lagrangian, extra_symbols.get('psibar_tau3_psi', sp.Symbol('psibar_tau3_psi', real=True)))
    effective_mass_p_expr = sp.expand(effective_mass_expr + isovector_scalar_mass_expr)
    effective_mass_n_expr = sp.expand(effective_mass_expr - isovector_scalar_mass_expr)
    energy_density_terms = energy_density_terms_from_lagrangian(total_lagrangian, extra_symbols, fields, field_names)
    energy_density_expr = sp.expand(sum(expr for expr, _ in energy_density_terms))
    return AnalyticModel(
        params=params,
        fields=fields,
        extra_symbols=extra_symbols,
        term_exprs=term_exprs,
        total_lagrangian=total_lagrangian,
        eom=eom,
        effective_mass_expr=effective_mass_expr,
        isovector_scalar_mass_expr=isovector_scalar_mass_expr,
        effective_mass_p_expr=effective_mass_p_expr,
        effective_mass_n_expr=effective_mass_n_expr,
        energy_density_terms=energy_density_terms,
        energy_density_expr=energy_density_expr,
    )
