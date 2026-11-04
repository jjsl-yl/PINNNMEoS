import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import train_fsu_delta_pinn_fields_variable10 as train
import benchmark_field_solver_numpy as bench
from manuscript_settings import ORIGINAL
from export_checkpoint import export

class SettingsTests(unittest.TestCase):
    def test_sampler_and_field_bounds(self):
        p=torch.tensor([[train.OPT[n] for n in train.PARAM_NAMES]])
        scales=torch.maximum(p.abs(),torch.ones_like(p))
        for name,value in ORIGINAL.items(): scales[0,train.PARAM_NAMES.index(name)]=abs(value)
        indices=torch.tensor([train.PARAM_NAMES.index(n) for n in train.TRAIN_INPUT_PARAM_NAMES])
        x,rn,rp=train.make_batch(4096,'cpu',p,scales,indices)
        n=rn+rp; y=rp/n
        self.assertTrue(torch.all((n>=1e-4)&(n<=1.2)))
        self.assertTrue(torch.all((y>=1e-5)&(y<=.5)))
        np.testing.assert_allclose(x[0,:10].numpy(),[train.OPT[n]/abs(ORIGINAL[n]) for n in train.TRAIN_INPUT_PARAM_NAMES],rtol=1e-6)
        fields=train.MLP()(x[:8])
        self.assertTrue(torch.all((fields[:,:2]>0)&(fields[:,:2]<120)))
        self.assertTrue(torch.all(fields[:,2:].abs()<40))

    def test_residuals_match_numpy_and_sum_loss(self):
        p=torch.tensor([[train.OPT[n] for n in train.PARAM_NAMES]],dtype=torch.float64)
        fields=torch.tensor([[35.,20.,-2.,1.]],dtype=torch.float64)
        rn,rp=.12,.04
        normalized=train.residuals(fields,torch.tensor([[rn]],dtype=torch.float64),torch.tensor([[rp]],dtype=torch.float64),p).numpy()[0]
        raw=bench.residual(fields.numpy()[0],rp,rn)
        pars=bench.PARAMS
        mp=pars['M_N']+pars['g_sigma']*35+pars['g_a0']
        mn=pars['M_N']+pars['g_sigma']*35-pars['g_a0']
        rs_p=bench.scalar_density((3*np.pi**2*rp*train.HBARC**3)**(1/3),mp)
        rs_n=bench.scalar_density((3*np.pi**2*rn*train.HBARC**3)**(1/3),mn)
        sources=np.array([pars['g_sigma']*(rs_p+rs_n),pars['g_omega']*(rp+rn)*train.HBARC**3,pars['g_rho']*(rp-rn)*train.HBARC**3,pars['g_a0']*(rs_p-rs_n)])
        expected=raw/(abs(sources)+1e6)
        np.testing.assert_allclose(normalized,expected[[2,1,0,3]],rtol=1e-10,atol=1e-10)
        res=torch.tensor([[1.,2.,3.,4.],[1.,2.,3.,4.]])
        self.assertEqual(float((res**2).mean(dim=0).sum()),30.)

    def test_portable_export(self):
        torch.manual_seed(17)
        model=train.MLP()
        scales=np.array([max(abs(train.OPT[n]),1) for n in train.PARAM_NAMES],dtype=np.float32)
        for name,value in ORIGINAL.items(): scales[train.PARAM_NAMES.index(name)]=abs(value)
        ckpt=dict(model=model.state_dict(),param_names=train.PARAM_NAMES,input_param_names=train.TRAIN_INPUT_PARAM_NAMES,optimized=train.OPT,param_scale=scales,step=3)
        with tempfile.TemporaryDirectory() as tmp:
            pt=Path(tmp)/'model.pt'; npz=Path(tmp)/'model.npz'
            torch.save(ckpt,pt); export(pt,npz)
            numpy_model=bench.NumpyMLP(npz)
            x=bench.network_input(np.array([[.04,.12],[.01,.2]]),numpy_model)
            with torch.no_grad(): expected=model(torch.from_numpy(x)).numpy()
            np.testing.assert_allclose(numpy_model(x),expected,rtol=3e-5,atol=3e-5)

    def test_proposal_and_local_dependencies(self):
        import wide_r1r2_20261103 as scan
        for sid in range(10):
            p,_=scan.proposal(sid,20261103)
            for name in scan.model.SAMPLED_KEYS:
                lo,hi=scan.BOUNDS[name]
                self.assertTrue(lo<=p[name]<=hi)
        self.assertEqual(scan.PROPOSAL_RELATIVE_SIGMA,.4)
        self.assertTrue((scan.model.TOOLKIT/'TOVtools/bps.dat').is_file())

if __name__=='__main__': unittest.main()
