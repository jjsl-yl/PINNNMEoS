"""Recompute existing EOSs; never overwrite the original selections."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize_scalar, brentq
from tov_love import BarotropicEOS, merge_bps_core, integrate_star


def calculate(job):
    index, table, bps, output = job
    target = Path(output) / f"curve_{index:04d}.npz"
    table = np.asarray(table, dtype=float)
    if (np.any(table[:,1] <= 0) or np.any(np.diff(table[:,1]) <= 0)
            or np.any(np.diff(table[:,2]) <= 0)):
        raise ValueError(f"EOS {index}: invalid core monotonicity/positive-pressure check")
    if target.exists():
        return index, "cached"
    p, e = merge_bps_core(bps, *table.T)
    eos = BarotropicEOS(p, e)
    def star(pc):
        return integrate_star(eos, float(pc), rtol=2e-7, atol=1e-12, max_step_km=.03)
    pcs = np.geomspace(.8, .995 * p[-1], 100)
    rows = []
    failures = []
    for pc in pcs:
        try:
            s = star(pc)
            rows.append([pc, s.mass_Msun, s.radius_km, s.Lambda])
        except (ValueError, RuntimeError) as exc:
            failures.append([float(pc), str(exc)])
    arr = np.asarray(rows)
    if len(arr) < 10:
        raise RuntimeError(f"EOS {index}: too few valid stars")
    peak = int(np.argmax(arr[:, 1]))
    if 0 < peak < len(arr)-1:
        opt = minimize_scalar(lambda x: -star(x).mass_Msun,
                              bounds=(arr[peak-1, 0], arr[peak+1, 0]),
                              method="bounded", options={"xatol":1e-5})
        s = star(opt.x)
        arr = np.vstack([arr[arr[:,0] < opt.x],
                         [opt.x, s.mass_Msun, s.radius_km, s.Lambda]])
    else:
        arr = arr[:peak+1]
    arr = arr[np.r_[True, np.diff(arr[:, 1]) > 0]]
    diagnostics = {"index":index,"Mmax":float(arr[-1,1]),
                   "maximum_bracketed":bool(0 < peak < len(rows)-1),
                   "failures":failures}
    for mass in [1.4, 2.0]:
        if arr[0,1] < mass < arr[-1,1]:
            j = int(np.searchsorted(arr[:,1], mass))
            pc = brentq(lambda x:star(x).mass_Msun-mass, arr[j-1,0], arr[j,0], xtol=1e-7)
            s = star(pc)
            diagnostics[f"R{mass}"] = s.radius_km
            diagnostics[f"Lambda{mass}"] = s.Lambda
    np.savez_compressed(target, curve=arr, diagnostics=json.dumps(diagnostics))
    return index, diagnostics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--bps", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--workers", type=int, default=64)
    args = ap.parse_args()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    tables = np.load(args.bank, allow_pickle=True)["tables"]
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(calculate, (i,t,args.bps,args.output)) for i,t in enumerate(tables)]
        for k,future in enumerate(as_completed(futures), 1):
            try:
                result = future.result()
                print(json.dumps({"done":k,"result":result}), flush=True)
            except Exception as exc:
                print(json.dumps({"done":k,"error":str(exc)}), flush=True)
