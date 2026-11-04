#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar


TOOLKIT = Path(os.environ.get("RMF_TOOLKIT", str(Path(__file__).resolve().parents[1]/"vendor/rmf")))
sys.path.insert(0, str(TOOLKIT))
sys.path.insert(0, str(TOOLKIT / "NMtoolsV1"))
sys.path.insert(0, str(TOOLKIT / "TOVtools"))

from analytic_layer import build_analytic_model
from nuclear_properties import (
    binding_energy_from_epsilon,
    compressibility_from_epsilon,
    epsilon_function_factory,
    symmetry_energy_from_epsilon,
    symmetry_slope_from_epsilon,
)
from tool_beta_eff import compile_beta_effective_model
from beta_eos import build_beta_equilibrium_eos_table
from hybrid_eos import build_hybrid_eos_with_bundled_crust
from tov_solver import build_eos_interpolator, build_mr_curve


PARAM_KEYS = [
    "m_sigma", "m_omega", "m_rho", "m_a0", "M_N",
    "g_sigma", "g_omega", "g_rho", "g_a0",
    "g2", "g3", "g4", "g5", "g6", "g7",
    "c3", "c4", "d3", "b3", "b4", "b5", "b6", "b7", "b8", "b9",
]
FIELDS = ["rho", "omega", "sigma", "a"]
EXTRA = [
    "psibar_i_dslash_psi",
    "psibar_psi",
    "psibar_gamma0_psi",
    "psibar_gamma0_tau3_psi",
    "psibar_tau3_psi",
]
TERMS = [
    "psibar_i_dslash_psi",
    "-P['M_N']*psibar_psi",
    "-P['g_sigma']*sigma*psibar_psi",
    "-P['g_a0']*a*psibar_tau3_psi",
    "-P['g_omega']*omega*psibar_gamma0_psi",
    "-P['g_rho']*rho*psibar_gamma0_tau3_psi",
    "-0.5*P['m_sigma']**2*sigma**2",
    "-(1/3)*P['g2']*sigma**3",
    "-0.25*P['g3']*sigma**4",
    "-0.5*P['g4']*sigma**2*omega**2",
    "-0.5*P['g5']*sigma**2*rho**2",
    "0.5*P['m_omega']**2*omega**2",
    "0.25*P['c3']*omega**4",
    "0.5*P['c4']*omega**2*rho**2",
    "0.5*P['m_rho']**2*rho**2",
    "0.25*P['d3']*rho**4",
    "-0.5*P['m_a0']**2*a**2",
    "0.25*P['b3']*a**4",
    "0.5*P['b4']*rho**2*a**2",
    "0.5*P['b6']*sigma**2*a**2",
    "0.5*P['b7']*omega**2*a**2",
    "P['b8']*rho*omega*sigma*a",
    "-P['g6']*sigma*rho**2",
    "-0.5*P['g7']*sigma*omega**2",
    "0.5*P['b5']*sigma*a**2",
    "P['b9']*a*rho*omega",
]

BASE_ZERO = dict(b3=0.0, b4=0.0, b5=0.0, b7=0.0, b8=0.0, b9=0.0, g4=0.0, g5=0.0, g6=0.0, g7=0.0, d3=0.0)
OPTIMIZED = dict(
    g_sigma=-9.770109960655295,
    g_omega=12.884663945566437,
    g_rho=7.008954689269367,
    g_a0=-6.7423962928532575,
    b6=187.942028466664,
    g2=1644.1640459214912,
    g3=5.983086268779212,
    c3=161.82097238231964,
    c4=203.68789171819614,
    m_sigma=483.64105174950686,
    m_omega=782.5,
    m_rho=763.0,
    m_a0=980.0,
    M_N=939.0,
    **BASE_ZERO,
)

SAMPLED_KEYS = ["g_sigma", "g_omega", "g_rho", "g_a0", "g2", "g3", "c3", "c4", "b6", "m_sigma"]
REL_SIGMA = {
    "g_sigma": 0.035,
    "g_omega": 0.035,
    "g_rho": 0.060,
    "g_a0": 0.060,
    "g2": 0.070,
    "g3": 0.100,
    "c3": 0.100,
    "c4": 0.100,
    "b6": 0.100,
    "m_sigma": 0.025,
}
HARD_BOUNDS = {
    "g_sigma": (-11.2, -8.4),
    "g_omega": (11.4, 14.2),
    "g_rho": (5.7, 8.5),
    "g_a0": (-8.1, -5.4),
    "g2": (1250.0, 2050.0),
    "g3": (3.5, 8.5),
    "c3": (115.0, 210.0),
    "c4": (150.0, 260.0),
    "b6": (135.0, 245.0),
    "m_sigma": (455.0, 510.0),
}

ANALYTIC = None


def init_worker():
    global ANALYTIC
    ANALYTIC = build_analytic_model(PARAM_KEYS, FIELDS, EXTRA, TERMS)


def sample_params(sample_id: int, seed: int, scale: float) -> dict:
    rng = np.random.default_rng(seed + sample_id * 104729)
    p = dict(OPTIMIZED)
    for key in SAMPLED_KEYS:
        base = OPTIMIZED[key]
        sigma = abs(base) * REL_SIGMA[key] * scale
        value = base + rng.normal(0.0, sigma)
        lo, hi = HARD_BOUNDS[key]
        value = float(np.clip(value, lo, hi))
        if key in ("g_sigma", "g_a0"):
            value = -abs(value)
        p[key] = value
    return p


def calc_nuclear_properties(params: dict) -> dict:
    eps = epsilon_function_factory(ANALYTIC, params, initial_guess=[0.0, 22.0, 36.0, 0.0])

    def ebind(rho_b: float) -> float:
        return binding_energy_from_epsilon(eps, rho_b / 2.0, rho_b / 2.0, params["M_N"])

    opt = minimize_scalar(ebind, bounds=(0.125, 0.175), method="bounded", options={"xatol": 8e-4})
    rho0 = float(opt.x)
    be = float(ebind(rho0))
    K = float(compressibility_from_epsilon(eps, rho0, params["M_N"], h=8e-4))
    J = float(symmetry_energy_from_epsilon(eps, rho0, params["M_N"], h_delta=2e-3))
    L = float(symmetry_slope_from_epsilon(eps, rho0, params["M_N"], h_rho=1.2e-3, h_delta=2e-3))
    return {"rho0": rho0, "Ebind": be, "K": K, "J": J, "L": L}


def passes_r1(nmp: dict) -> bool:
    return (
        0.135 <= nmp["rho0"] <= 0.160
        and -17.2 <= nmp["Ebind"] <= -15.2
        and 180.0 <= nmp["K"] <= 280.0
        and 26.0 <= nmp["J"] <= 33.5
        and 20.0 <= nmp["L"] <= 75.0
    )


def interp_radius_at_mass(masses, radii, target):
    m = np.asarray(masses, dtype=float)
    r = np.asarray(radii, dtype=float)
    mask = np.isfinite(m) & np.isfinite(r) & (m > 0) & (r > 0) & (r < 60)
    m, r = m[mask], r[mask]
    if m.size < 5:
        return None
    imax = int(np.nanargmax(m))
    m, r = m[: imax + 1], r[: imax + 1]
    order = np.argsort(m)
    m, r = m[order], r[order]
    if target < m[0] or target > m[-1]:
        return None
    return float(np.interp(target, m, r))


def build_star_outputs(params: dict, args) -> tuple[list[dict], list[dict], dict]:
    compiled = compile_beta_effective_model(ANALYTIC, params)
    rho_grid = np.linspace(args.rho_min_core, args.rho_max_core, args.eos_points).tolist()
    eos = build_beta_equilibrium_eos_table(
        lambda rn, rp: 0.0,
        lambda rn, rp: 0.0,
        rho_grid,
        compiled_model=compiled,
        params=params,
        field_guess=[0.0, 22.0, 36.0, 0.0],
    )
    eos_rows = []
    for p in eos:
        d = p.to_dict()
        d["Yp"] = d["rho_p"] / d["rho_b"] if d["rho_b"] else math.nan
        eos_rows.append(d)
    hybrid = build_hybrid_eos_with_bundled_crust(eos_rows, rho_min_core=args.rho_min_core)
    hrows = hybrid.merged_table
    eos_interp = build_eos_interpolator(hrows)
    pvals = np.array([r["pressure_total_mev_fm3"] for r in hrows if r["pressure_total_mev_fm3"] > 0.0], dtype=float)
    pc_min = max(float(np.nanmin(pvals)) * 5.0, 0.05)
    pc_max = float(np.nanmax(pvals)) * 0.98
    pcs = np.geomspace(pc_min, pc_max, args.mr_points).tolist()
    mr = build_mr_curve(eos_interp, pcs, dr_km=args.dr_km, progress=False)
    mr_rows = [{"pc": pc, "M": m, "R": r} for pc, m, r in zip(mr.central_pressures_mev_fm3, mr.masses_msun, mr.radii_km)]
    masses = np.asarray(mr.masses_msun, dtype=float)
    radii = np.asarray(mr.radii_km, dtype=float)
    imax = int(np.nanargmax(masses))
    summary = {
        "Mmax": float(masses[imax]),
        "R_at_Mmax": float(radii[imax]),
        "R14": interp_radius_at_mass(masses, radii, 1.4),
        "R20": interp_radius_at_mass(masses, radii, 2.0),
        "max_abs_delta_beta": float(max(abs(r["delta_beta"]) for r in eos_rows)),
        "pressure_core_min": float(min(r["pressure_total_mev_fm3"] for r in eos_rows)),
        "pressure_core_max": float(max(r["pressure_total_mev_fm3"] for r in eos_rows)),
    }
    return eos_rows, mr_rows, summary


def passes_r2(star: dict) -> bool:
    r14 = star.get("R14")
    r20 = star.get("R20")
    return (
        star["Mmax"] >= 2.0
        and r14 is not None
        and 10.5 <= r14 <= 13.2
        and (r20 is None or 10.5 <= r20 <= 13.3)
    )


def write_csv(path: Path, rows: list[dict], keys: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_one(task):
    sample_id, seed, args_dict = task
    class Args:
        pass
    args = Args()
    args.__dict__.update(args_dict)
    params = sample_params(sample_id, seed, args.sample_scale)
    out = {
        "sample_id": sample_id,
        "ok": False,
        "r1_pass": False,
        "r2_pass": False,
        "params": params,
        "nmp": None,
        "star": None,
        "error": None,
    }
    try:
        nmp = calc_nuclear_properties(params)
        out["nmp"] = nmp
        out["r1_pass"] = passes_r1(nmp)
        if out["r1_pass"] or args.keep_all_star:
            eos_rows, mr_rows, star = build_star_outputs(params, args)
            out["star"] = star
            out["r2_pass"] = out["r1_pass"] and passes_r2(star)
            sample_dir = Path(args.output_dir) / f"sample_{sample_id:06d}"
            if out["r1_pass"] or out["r2_pass"] or args.save_failed_curves:
                write_csv(
                    sample_dir / "beta_core.csv",
                    eos_rows,
                    [
                        "rho_b", "Yp", "rho_n", "rho_p", "mu_n", "mu_p", "mu_e", "delta_beta",
                        "epsilon_total_mev_fm3", "pressure_total_mev_fm3",
                        "epsilon_baryon_mev_fm3", "pressure_baryon_mev_fm3",
                        "epsilon_electron_mev_fm3", "pressure_electron_mev_fm3",
                    ],
                )
                write_csv(sample_dir / "mr.csv", mr_rows, ["pc", "M", "R"])
                (sample_dir / "params.json").write_text(json.dumps(params, indent=2))
                (sample_dir / "summary.json").write_text(json.dumps({"nmp": nmp, "star": star, "r1_pass": out["r1_pass"], "r2_pass": out["r2_pass"]}, indent=2))
        out["ok"] = True
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=6)}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--sample-start", type=int, default=0)
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260707)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--sample-scale", type=float, default=1.0)
    ap.add_argument("--rho-min-core", type=float, default=0.08)
    ap.add_argument("--rho-max-core", type=float, default=1.4)
    ap.add_argument("--eos-points", type=int, default=180)
    ap.add_argument("--mr-points", type=int, default=100)
    ap.add_argument("--dr-km", type=float, default=0.003)
    ap.add_argument("--keep-all-star", action="store_true")
    ap.add_argument("--save-failed-curves", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    args_dict = vars(args).copy()
    tasks = [(args.sample_start + i, args.seed, args_dict) for i in range(args.samples)]

    results = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker) as pool:
        futs = [pool.submit(run_one, task) for task in tasks]
        for done, fut in enumerate(as_completed(futs), 1):
            result = fut.result()
            results.append(result)
            if done % 5 == 0 or done == len(futs):
                ok = sum(r["ok"] for r in results)
                r1 = sum(r["r1_pass"] for r in results)
                r2 = sum(r["r2_pass"] for r in results)
                print(f"completed {done}/{len(futs)} ok={ok} R1={r1} R2={r2}", flush=True)

    summary = {
        "sample_start": args.sample_start,
        "samples": args.samples,
        "workers": args.workers,
        "sample_scale": args.sample_scale,
        "eos_points": args.eos_points,
        "mr_points": args.mr_points,
        "dr_km": args.dr_km,
        "ok": sum(r["ok"] for r in results),
        "r1_pass": sum(r["r1_pass"] for r in results),
        "r2_pass": sum(r["r2_pass"] for r in results),
    }
    (outdir / "results.jsonl").write_text("\n".join(json.dumps(r) for r in sorted(results, key=lambda x: x["sample_id"])) + "\n")
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    init_worker()
    main()
