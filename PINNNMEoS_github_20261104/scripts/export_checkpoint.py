"""Export a trusted local PyTorch checkpoint to portable, pickle-free NumPy."""
import argparse
from pathlib import Path
import numpy as np
import torch
from train_fsu_delta_pinn_fields_variable10 import PARAM_NAMES, TRAIN_INPUT_PARAM_NAMES

def export(source, output):
    # Only load checkpoints produced by you or otherwise trusted; PyTorch pickle
    # checkpoints may execute code. Downstream inference uses pickle-free NPZ.
    checkpoint=torch.load(source,map_location='cpu',weights_only=False)
    state=checkpoint['model']
    keys=[f'net.{layer}.{kind}' for layer in range(0,14,2) for kind in ('weight','bias')]
    arrays={f'array_{i}':state[key].detach().numpy() for i,key in enumerate(keys)}
    names=checkpoint['param_names']
    input_names=checkpoint.get('input_param_names',TRAIN_INPUT_PARAM_NAMES)
    if input_names != TRAIN_INPUT_PARAM_NAMES:
        raise ValueError('Unexpected input parameter order')
    arrays.update(input_scale=np.asarray(checkpoint['param_scale'])[[names.index(n) for n in input_names]],
                  input_parameters=np.array([checkpoint['optimized'][n] for n in input_names],dtype=np.float32),
                  step=np.array(checkpoint['step']))
    np.savez_compressed(output,**arrays)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True,help='Trusted local .pt file')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    export(args.checkpoint,args.output)
