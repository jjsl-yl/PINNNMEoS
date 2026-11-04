"""Pointwise field and baryonic thermodynamic validation at identical inputs."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.optimize import fsolve
from benchmark_field_solver_numpy import NumpyMLP, network_input, residual, PARAMS, HBARC

def thermodynamics(fields,rp_fm,rn_fm):
    p=PARAMS
    sigma,omega,r,a=fields
    rp,rn=np.array([rp_fm,rn_fm])*HBARC**3
    mp=p['M_N']+p['g_sigma']*sigma+p['g_a0']*a
    mn=p['M_N']+p['g_sigma']*sigma-p['g_a0']*a
    if min(mp,mn)<=0: raise ValueError('Nonpositive effective mass')
    kp,kn=(3*np.pi**2*np.array([rp,rn]))**(1/3)
    kinetic=sum(quad(lambda k:k*k*np.sqrt(k*k+m*m),0,k,epsabs=1e-6,epsrel=1e-11)[0]/np.pi**2 for k,m in [(kp,mp),(kn,mn)])
    energy=(kinetic+p['g_omega']*omega*(rp+rn)+p['g_rho']*r*(rp-rn)
            +.5*p['m_sigma']**2*sigma**2+p['g2']*sigma**3/3+p['g3']*sigma**4/4
            -.5*p['m_omega']**2*omega**2-p['c3']*omega**4/4
            -.5*p['m_rho']**2*r*r-.5*p['c4']*omega**2*r*r
            +.5*p['m_a0']**2*a*a-.5*p['b6']*sigma*sigma*a*a)
    mu_p=np.sqrt(kp*kp+mp*mp)+p['g_omega']*omega+p['g_rho']*r
    mu_n=np.sqrt(kn*kn+mn*mn)+p['g_omega']*omega-p['g_rho']*r
    return np.array([energy,mu_p*rp+mu_n*rn-energy])/HBARC**3

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--output',type=Path,default=Path('outputs/validation.json'))
    ap.add_argument('--points',type=int,default=256)
    ap.add_argument('--seed',type=int,default=20261104)
    ap.add_argument('--field-denominator-floor',type=float,default=1.)
    args=ap.parse_args()
    if args.points<1 or args.field_denominator_floor<=0: ap.error('positive points/floor required')
    rng=np.random.default_rng(args.seed)
    nb=rng.uniform(.001,1.,args.points); yp=rng.uniform(1e-5,.5,args.points)
    pairs=np.column_stack([nb*yp,nb*(1-yp)])
    model=NumpyMLP(args.checkpoint)
    pred=model(network_input(pairs,model)).astype(float)
    rows=[]; exact=[]; pred_eos=[]; exact_eos=[]
    for fields,(rp,rn) in zip(pred,pairs):
        sol,info,status,message=fsolve(residual,fields,args=(rp,rn),xtol=1e-10,maxfev=500,full_output=True)
        norm=float(np.max(np.abs(residual(sol,rp,rn))))
        if status!=1 or not np.all(np.isfinite(sol)) or norm>1e-2:
            raise RuntimeError(f'Exact correction failed: {message}; residual={norm}')
        exact.append(sol)
        pred_eos.append(thermodynamics(fields,rp,rn))
        exact_eos.append(thermodynamics(sol,rp,rn))
        rows.append(dict(nB_fm3=rp+rn,Yp=rp/(rp+rn),exact_residual_max_MeV3=norm))
    prediction=np.column_stack([pred,np.array(pred_eos)])
    reference=np.column_stack([np.array(exact),np.array(exact_eos)])
    absolute=abs(prediction-reference)
    floor=np.array([args.field_denominator_floor]*4+[1e-12]*2)
    relative=absolute/np.maximum(abs(reference),floor)
    names=['sigma','omega','r','a','energy_density','pressure']
    summary={name:dict(mean_relative=float(relative[:,i].mean()),p95_relative=float(np.quantile(relative[:,i],.95)),max_absolute=float(absolute[:,i].max())) for i,name in enumerate(names)}
    result=dict(settings=vars(args)|{'output':str(args.output)},summary=summary,
                prediction=prediction.tolist(),exact=reference.tolist(),points=rows,
                note='New random validation, not a replay of the historical table. Baryonic matter without leptons; fields MeV, energy/pressure MeV fm^-3. Exact solves initialized by PINN; roots are not independently proven unique.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
