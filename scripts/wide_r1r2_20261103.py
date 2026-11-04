"""Broader proposals; unchanged nuclear intervals and downstream stellar tests."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time
import platform

import numpy as np
from scipy.optimize import minimize_scalar
import sample_fig5_optimized_neighborhood as model
from recompute_joint_constraint_bank import calculate
from manuscript_settings import ORIGINAL, PROPOSAL_RELATIVE_SIGMA, NUCLEAR_INTERVALS

LIMITS = NUCLEAR_INTERVALS
SCALES = (1.,)
BOUNDS = {k: (model.OPTIMIZED[k] + 2 * (lo - model.OPTIMIZED[k]),
              model.OPTIMIZED[k] + 2 * (hi - model.OPTIMIZED[k]))
          for k, (lo, hi) in model.HARD_BOUNDS.items()}


def proposal(sid, seed):
    rng = np.random.default_rng(seed + sid * 104729)
    scale = SCALES[sid % len(SCALES)]
    p = dict(model.OPTIMIZED)
    for k in model.SAMPLED_KEYS:
        lo, hi = BOUNDS[k]
        sigma = abs(ORIGINAL[k]) * PROPOSAL_RELATIVE_SIGMA
        # Truncate by rejection, not clipping: no artificial mass on the bounds.
        while True:
            v = rng.normal(p[k], sigma)
            if lo <= v <= hi:
                p[k] = float(v)
                break
    return p, scale


def nuclear(p):
    eps = model.epsilon_function_factory(model.ANALYTIC, p, initial_guess=[0.,22.,36.,0.])
    def energy(n):
        return model.binding_energy_from_epsilon(eps,n/2,n/2,p['M_N'])
    grid = np.linspace(.08,.24,17)
    vals = np.array([energy(n) for n in grid])
    if not np.all(np.isfinite(vals)):
        raise ValueError('nonfinite saturation scan')
    j = int(np.argmin(vals))
    if j in (0, len(grid)-1):
        raise ValueError('saturation minimum outside search bracket [.08,.24]')
    opt = minimize_scalar(energy,bounds=(grid[j-1],grid[j+1]),method='bounded',
                          options={'xatol':8e-4})
    n = float(opt.x)
    return dict(rho0=n,Ebind=float(energy(n)),
        K=float(model.compressibility_from_epsilon(eps,n,p['M_N'],h=8e-4)),
        J=float(model.symmetry_energy_from_epsilon(eps,n,p['M_N'],h_delta=2e-3)),
        L=float(model.symmetry_slope_from_epsilon(eps,n,p['M_N'],h_rho=1.2e-3,h_delta=2e-3)))


def work(job):
    sid, seed, root, bps, eos_points, only_nuclear = job
    start = time.monotonic()
    p, scale = proposal(sid,seed)
    result = dict(sample_id=sid,params=p,scale=scale,status='started')
    try:
        nmp = nuclear(p)
        if not all(np.isfinite(v) for v in nmp.values()):
            raise ValueError('nonfinite nuclear property')
        result['nmp'] = nmp
        result['nuclear_pass'] = all(np.isfinite(nmp[k]) and lo<=nmp[k]<=hi
                                     for k,(lo,hi) in LIMITS.items())
        result['status'] = 'nuclear_reject'
        if not result['nuclear_pass'] or only_nuclear:
            return result
        result['status'] = 'eos_failure'
        compiled = model.compile_beta_effective_model(model.ANALYTIC,p)
        rows = model.build_beta_equilibrium_eos_table(lambda rn,rp:0.,lambda rn,rp:0.,
            np.linspace(.08,1.4,eos_points).tolist(),compiled_model=compiled,
            params=p,field_guess=[0.,22.,36.,0.])
        result['beta_max_abs_MeV'] = max(abs(r.delta_beta) for r in rows)
        t = np.array([[r.rho_b,r.pressure_total_mev_fm3,r.epsilon_total_mev_fm3] for r in rows])
        result['status'] = 'core_reject'
        if not np.all(np.isfinite(t)) or np.any(t[:,1]<=0) or np.any(np.diff(t[:,1])<=0) or np.any(np.diff(t[:,2])<=0):
            return result
        result['status'] = 'beta_reject'
        if result['beta_max_abs_MeV']>1e-4:
            return result
        np.savez_compressed(Path(root)/'cores'/f'core_{sid:07d}.npz',table=t)
        result['status'] = 'stellar_failure'
        _, diagnostics = calculate((sid,t,bps,str(Path(root)/'curves')))
        curve_path = Path(root)/'curves'/f'curve_{sid:04d}.npz'
        if diagnostics == 'cached':
            with np.load(curve_path) as saved:
                diagnostics = json.loads(str(saved['diagnostics']))
        # Do not silently discard incomplete TOV sequences.
        if diagnostics.get('failures') or not diagnostics.get('maximum_bracketed'):
            result['stellar'] = diagnostics
            result['status'] = 'stellar_incomplete'
            return result
        result['stellar'] = diagnostics
        result['status'] = 'r1'
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        if result['status']=='started': result['status']='nuclear_failure'
    finally:
        result['seconds'] = time.monotonic()-start
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',required=True)
    ap.add_argument('--bps',required=True)
    ap.add_argument('--count',type=int,default=432000)
    ap.add_argument('--offset',type=int,default=0)
    ap.add_argument('--shards',type=int,default=1)
    ap.add_argument('--shard',type=int,default=0)
    ap.add_argument('--workers',type=int,default=1)
    ap.add_argument('--seed',type=int,default=20261103)
    ap.add_argument('--eos-points',type=int,default=400)
    ap.add_argument('--only-nuclear',action='store_true')
    args=ap.parse_args()
    out=Path(args.output)
    for folder in (out,out/'cores',out/'curves'): folder.mkdir(parents=True,exist_ok=True)
    manifest=dict(args=vars(args),width_scales=SCALES,hard_bounds=BOUNDS,
        sigma_rule='0.40 * abs(original Table I reference)',nuclear_limits=LIMITS,
        saturation_search=[.08,.24],proposal='independent truncated Gaussian; no clipping',
        interpretation='equal-proposal compatibility sample, not Bayesian posterior',
        stellar_selection='R1 generation only; apply J0740 + J0437 + GW170817 in select_r2.py',
        host=platform.node())
    manifest_path=out/f'manifest_{args.shard}.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('Existing output uses a different configuration; choose a fresh directory')
    manifest_path.write_text(json.dumps(manifest,indent=2))
    done=set()
    path=out/f'records_{args.shard}.jsonl'
    if path.exists():
        for line in path.read_text().splitlines(): done.add(json.loads(line)['sample_id'])
    ids=[i+args.offset for i in range(args.count) if i%args.shards==args.shard and i+args.offset not in done]
    model.init_worker()
    with path.open('a',buffering=1) as handle, ProcessPoolExecutor(args.workers,initializer=model.init_worker) as pool:
        futures=[pool.submit(work,(i,args.seed,str(out),args.bps,args.eos_points,args.only_nuclear)) for i in ids]
        counts={}
        for k,f in enumerate(as_completed(futures),1):
            r=f.result()
            handle.write(json.dumps(r,allow_nan=False)+'\n')
            counts[r['status']]=counts.get(r['status'],0)+1
            if k%25==0 or k==len(futures): print(json.dumps(dict(done=k,total=len(futures),counts=counts)),flush=True)

if __name__=='__main__': main()
