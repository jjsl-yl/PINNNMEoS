"""Separate nominal posterior-region compatibility: J0740, J0437, GW170817."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator, RegularGridInterpolator
from gw_tidal_kde import GWTidalKDE

LEVELS=(.68,.90,.95)

def mr_estimate(path):
    with np.load(path,allow_pickle=False) as g:
        ma,ra,d=(g[k] for k in ('mass_axis_Msun','radius_axis_km','density'))
    weight=np.outer(np.gradient(ma),np.gradient(ra))
    order=np.argsort(d.ravel())[::-1]
    cumulative=np.cumsum((d*weight).ravel()[order]); cumulative/=cumulative[-1]
    cuts={str(c):float(d.ravel()[order[np.searchsorted(cumulative,c)]]) for c in LEVELS}
    return ma,RegularGridInterpolator((ma,ra),d,bounds_error=False,fill_value=0),cuts

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bank',type=Path,required=True)
    ap.add_argument('--observations',type=Path,required=True)
    args=ap.parse_args()
    sources={name:mr_estimate(args.observations/f'{name}_mr_posterior_grid.npz') for name in ('j0740','j0437')}
    raw=np.genfromtxt(args.observations/'gw170817/low_spin_PhenomPNRT_posterior_samples.dat.gz',names=True)
    m1=raw['m1_detector_frame_Msun']/1.0099
    m2=raw['m2_detector_frame_Msun']/1.0099
    mc=(m1*m2)**.6/(m1+m2)**.2; q=m2/m1
    points=np.column_stack([mc,q,raw['lambda1'],raw['lambda2']])
    kde=GWTidalKDE.fit(points,bandwidth=.25); kde.model.rtol=1e-4
    density=kde.log_density(points)
    gw_cuts={str(c):float(np.quantile(density,1-c)) for c in LEVELS}
    cm,qq=np.meshgrid(np.linspace(*np.quantile(mc,[.001,.999]),7),np.linspace(max(.5,q.min()),1,80))
    cm,qq=cm.ravel(),qq.ravel()
    mass1=cm*(1+qq)**.2/qq**.6; mass2=qq*mass1
    records={}
    for path in sorted(args.bank.glob('records_*.jsonl')):
        for line in path.read_text().splitlines():
            r=json.loads(line)
            if r['sample_id'] in records: raise ValueError('Duplicate sample IDs across shards')
            records[r['sample_id']]=r
    output=[]
    for sid,r in sorted(records.items()):
        if r['status']!='r1': continue
        with np.load(args.bank/'curves'/f'curve_{sid:04d}.npz',allow_pickle=False) as saved:
            curve=saved['curve']
        curve=curve[curve[:,1]>=.8]
        mass,radius,lam=curve[:,1:].T
        if len(mass)<4 or np.any(np.diff(mass)<=0): raise ValueError(f'Invalid stable branch: {sid}')
        radius_at=PchipInterpolator(mass,radius,extrapolate=False)
        tidal_at=PchipInterpolator(mass,lam,extrapolate=False)
        peaks={}
        for name,(ma,pdf,cuts) in sources.items():
            lo,hi=max(ma[0],mass[0]),min(ma[-1],mass[-1])
            test=np.linspace(lo,hi,1000) if lo<=hi else np.array([])
            peaks[name]=float(np.max(pdf(np.column_stack([test,radius_at(test)])))) if len(test) else 0.
        l1,l2=tidal_at(mass1),tidal_at(mass2)
        ok=np.isfinite(l1)&np.isfinite(l2)&(l1>=0)&(l2>=0)
        gp=kde.log_density(np.column_stack([cm[ok],qq[ok],l1[ok],l2[ok]]))
        gw_peak=float(np.max(gp)) if len(gp) else -1e300
        accepted={str(c):bool(all(peaks[n]>=s[2][str(c)] for n,s in sources.items()) and gw_peak>=gw_cuts[str(c)]) for c in LEVELS}
        output.append(dict(sample_id=sid,mr_peaks=peaks,gw_peak_log_density=gw_peak,compatibility=accepted))
    report=dict(selection_sources=['J0740','J0437','GW170817'],R1=len(output),
                R2_by_level={str(c):sum(r['compatibility'][str(c)] for r in output) for c in LEVELS},
                MR_cutoffs={n:s[2] for n,s in sources.items()},GW_cutoffs=gw_cuts,
                interpretation='Separate approximate contour intersections, not a combined credible region or an EOS posterior',
                records=output)
    (args.bank/'r2_results.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))

if __name__=='__main__': main()
