#!/usr/bin/env python3
"""PINN for FSU-delta RMF mean-field equations.

Input: 25 RMF parameters + rho_n + rho_p (fm^-3)
Output: sigma, omega, rho, a0 mean fields (MeV units in the code convention)

This script trains on the Optimized parameter set by default, but keeps the
parameter vector in the input so the same network interface can be extended to
parameter ensembles later.
"""
import argparse, json, math, os, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import psutil
from manuscript_settings import ORIGINAL, LOSS_REDUCTION

HBARC = 197.326968
PI = math.pi
PARAM_NAMES = ['g_sigma','g_omega','g_rho','g_a0','b3','b4','b5','b6','b7','b8','b9','g2','g3','g4','g5','g6','g7','c3','c4','d3','m_sigma','m_omega','m_rho','m_a0','M_N']
OPT = dict(g_sigma=-9.770109960655295,g_omega=12.884663945566437,g_rho=7.008954689269367,g_a0=-6.7423962928532575,b3=0,b4=0,b5=0,b6=187.942028466664,b7=0,b8=0,b9=0,g2=1644.1640459214912,g3=5.983086268779212,g4=0,g5=0,g6=0,g7=0,c3=161.82097238231964,c4=203.68789171819614,d3=0,m_sigma=483.64105174950686,m_omega=782.5,m_rho=763.0,m_a0=980.0,M_N=939.0)

TRAIN_INPUT_PARAM_NAMES = ['g_sigma','g_omega','g_rho','g_a0','b6','g2','g3','c3','c4','m_sigma']

class MLP(nn.Module):
    def __init__(self, in_dim=None, width=256, depth=6):
        if in_dim is None:
            in_dim = len(TRAIN_INPUT_PARAM_NAMES) + 2
        super().__init__()
        layers=[]
        d=in_dim
        for _ in range(depth):
            layers += [nn.Linear(d,width), nn.SiLU()]
            d=width
        layers += [nn.Linear(d,4)]
        self.net=nn.Sequential(*layers)
    def forward(self,x):
        y=self.net(x)
        # Bound fields to a physically useful range for stable residual training.
        sigma = 120.0*torch.sigmoid(y[:,0:1])        # usually positive, ~tens MeV
        omega = 120.0*torch.sigmoid(y[:,1:2])        # usually positive, ~tens MeV
        rho   =  40.0*torch.tanh(y[:,2:3])           # isovector-vector field
        a0    =  40.0*torch.tanh(y[:,3:4])           # delta/isovector-scalar field
        return torch.cat([sigma,omega,rho,a0],dim=1)

def torch_kf(rho_fm3):
    x = torch.clamp(rho_fm3, min=0.0) * HBARC**3
    return torch.pow(torch.clamp(3*PI**2*x, min=0.0), 1.0/3.0)

def torch_rho_s(k,m):
    m_safe=torch.clamp(m, min=1e-6)
    t=k/m_safe
    return m_safe**3/PI**2 * 0.5*(t*torch.sqrt(1+t*t)-torch.asinh(t))

def make_batch(batch, device, param_vec, param_scale, param_input_idx, rho_min=1e-4, rho_max=1.2):
    # sample total baryon density log-uniform at low densities + uniform over full range
    u=torch.rand(batch,1,device=device)
    mix=torch.rand(batch,1,device=device)
    nb_log=torch.exp(math.log(rho_min)+(math.log(rho_max)-math.log(rho_min))*u)
    nb_lin=rho_min+(rho_max-rho_min)*u
    nb=torch.where(mix<0.35, nb_log, nb_lin)
    yp=1e-5+(0.5-1e-5)*torch.rand(batch,1,device=device)
    rp=nb*yp
    rn=nb*(1-yp)
    p_in=param_vec[:,param_input_idx].expand(batch,-1)
    s_in=param_scale[:,param_input_idx].expand(batch,-1)
    x=torch.cat([p_in/s_in, rn/0.16, rp/0.16], dim=1)
    return x, rn, rp

def residuals(fields, rn_fm, rp_fm, p):
    # fields order: sigma, omega, rho, a0; EOM residual order: rho, omega, sigma, a0
    sigma=fields[:,0:1]; omega=fields[:,1:2]; rho=fields[:,2:3]; a0=fields[:,3:4]
    gp=lambda name: p[:,PARAM_NAMES.index(name):PARAM_NAMES.index(name)+1]
    rn=rn_fm*HBARC**3; rp=rp_fm*HBARC**3
    mn=gp('M_N') + gp('g_sigma')*sigma - gp('g_a0')*a0
    mp=gp('M_N') + gp('g_sigma')*sigma + gp('g_a0')*a0
    kfn=torch_kf(rn_fm); kfp=torch_kf(rp_fm)
    rns=torch_rho_s(kfn,mn); rps=torch_rho_s(kfp,mp)
    Rrho=(gp('m_rho')**2*rho + gp('d3')*rho**3 + gp('c4')*omega**2*rho - gp('g5')*sigma**2*rho + gp('b4')*a0**2*rho + gp('b8')*omega*sigma*a0 - 2*gp('g6')*sigma*rho + gp('b9')*omega*a0 - gp('g_rho')*(rp-rn))
    Romega=(gp('m_omega')**2*omega + gp('c3')*omega**3 + gp('c4')*omega*rho**2 - gp('g4')*sigma**2*omega + gp('b7')*a0**2*omega + gp('b8')*rho*sigma*a0 - 2*gp('g7')*sigma*omega + gp('b9')*rho*a0 - gp('g_omega')*(rp+rn))
    Rsigma=(gp('m_sigma')**2*sigma + gp('g2')*sigma**2 + gp('g3')*sigma**3 + gp('g4')*sigma*omega**2 + gp('g5')*rho**2*sigma - gp('b6')*a0**2*sigma - gp('b8')*omega*rho*a0 + gp('g6')*rho**2 + 0.5*gp('g7')*omega**2 - 0.5*gp('b5')*a0**2 + gp('g_sigma')*(rps+rns))
    Ra0=(gp('m_a0')**2*a0 - gp('b3')*a0**3 - gp('b4')*rho**2*a0 - gp('b6')*sigma**2*a0 - gp('b7')*omega**2*a0 - gp('b8')*rho*omega*sigma - gp('b5')*a0*sigma - gp('b9')*omega*rho + gp('g_a0')*(rps-rns))
    # scale each residual by its dominant source magnitude to keep losses balanced
    eps=1e6
    srho=torch.abs(gp('g_rho')*(rp-rn))+eps
    somega=torch.abs(gp('g_omega')*(rp+rn))+eps
    ssigma=torch.abs(gp('g_sigma')*(rps+rns))+eps
    sa0=torch.abs(gp('g_a0')*(rps-rns))+eps
    return torch.cat([Rrho/srho, Romega/somega, Rsigma/ssigma, Ra0/sa0], dim=1)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--outdir', default='outputs/training')
    ap.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    ap.add_argument('--steps', type=int, default=200000)
    ap.add_argument('--batch', type=int, default=8192)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--width', type=int, default=256)
    ap.add_argument('--depth', type=int, default=6)
    ap.add_argument('--save-every', type=int, default=5000)
    ap.add_argument('--seed', type=int, default=20260619)
    args=ap.parse_args()
    out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if min(args.steps,args.batch,args.save_every,args.width,args.depth) < 1 or args.lr <= 0:
        ap.error('steps, batch, save-every, width, depth and lr must be positive')
    if (out/'train_log.jsonl').exists():
        ap.error('output contains an existing training log; use a new directory')
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device)
    param_np=np.array([OPT[n] for n in PARAM_NAMES], dtype=np.float32)
    scale_np=np.maximum(np.abs(param_np), 1.0).astype(np.float32)
    for name, value in ORIGINAL.items():
        scale_np[PARAM_NAMES.index(name)] = abs(value)
    param_vec=torch.tensor(param_np,device=device).view(1,-1)
    param_scale=torch.tensor(scale_np,device=device).view(1,-1)
    model=MLP(width=args.width, depth=args.depth).to(device)
    opt=torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr*0.05)
    param_input_idx=torch.tensor([PARAM_NAMES.index(n) for n in TRAIN_INPUT_PARAM_NAMES],device=device,dtype=torch.long)
    meta=dict(args=vars(args), param_names=PARAM_NAMES, input_param_names=TRAIN_INPUT_PARAM_NAMES, input_dim=len(TRAIN_INPUT_PARAM_NAMES)+2, optimized=OPT, original=ORIGINAL, loss_reduction=LOSS_REDUCTION, normalization='absolute original Table I reference; rounded table values', device=str(device), torch=torch.__version__)
    (out/'metadata.json').write_text(json.dumps(meta, indent=2))
    log_path=out/'train_log.jsonl'
    process=psutil.Process()
    peak_rss=0
    if device.type=='cuda':
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    t0=time.perf_counter()
    for step in range(1,args.steps+1):
        x,rn,rp=make_batch(args.batch,device,param_vec,param_scale,param_input_idx)
        p=param_vec.expand(args.batch,-1)
        f=model(x)
        res=residuals(f,rn,rp,p)
        components=(res**2).mean(dim=0)
        loss=components.sum()
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step(); sched.step()
        if True:
            with torch.no_grad():
                rss=process.memory_info().rss
                peak_rss=max(peak_rss,rss)
                item=dict(step=step, loss=float(loss.detach().cpu()), component_loss=components.detach().cpu().tolist(), component_order=['r','omega','sigma','a'], res_abs_mean=[float(v) for v in res.abs().mean(0).detach().cpu()], lr=float(sched.get_last_lr()[0]), sec=time.perf_counter()-t0, sampled_host_rss_MiB=rss/2**20, sampled_peak_host_rss_MiB=peak_rss/2**20)
                if device.type=='cuda':
                    item.update(cuda_peak_allocated_MiB=torch.cuda.max_memory_allocated(device)/2**20, cuda_peak_reserved_MiB=torch.cuda.max_memory_reserved(device)/2**20)
            with log_path.open('a') as fp: fp.write(json.dumps(item)+'\n')
            if step==1 or step%100==0: print(json.dumps(item), flush=True)
        if step%args.save_every==0 or step==args.steps:
            ckpt={'model':model.state_dict(),'step':step,'param_names':PARAM_NAMES,'input_param_names':TRAIN_INPUT_PARAM_NAMES,'optimized':OPT,'param_scale':scale_np,'meta':meta,'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state()}
            torch.save(ckpt, out/f'checkpoint_step_{step:07d}.pt')
            torch.save(ckpt, out/'latest.pt')
    if device.type=='cuda': torch.cuda.synchronize(device)
    (out/'summary.json').write_text(json.dumps(dict(training_loop_seconds=time.perf_counter()-t0, final_loss=item['loss'], sampled_peak_host_rss_MiB=peak_rss/2**20, steps=args.steps, note='Fresh manuscript configuration, not reproduction of historical measured numbers. RSS is sampled, not OS peak working set. Timing includes log/checkpoint I/O.'),indent=2))
    print('DONE', out/'latest.pt')
if __name__=='__main__': main()
