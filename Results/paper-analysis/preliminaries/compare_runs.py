"""
Table of the local metrics and constraint losses of chosen generation folders (each with a `generated.npy`), e.g. the
runs of `Diffusion_Model/generate.py` in `test_new_sampler_clip/` :
inversion fraction (any decrease and > 0.01 kg/m3), level-mean / anomaly variance, roughness, diversity, distance to a
reference run of the same seed, and the median of the constraint losses.

    python compare_runs.py --folder <dir> --runs none 'combo_*' --ref none
"""
from __init__ import PRP; import sys
sys.path.append(f'{PRP}/Diffusion_Model/')
sys.path.append(PRP)

import argparse, glob, os
import numpy as np
import pandas as pd
import torch
from configs.base_config import TrainingConfig
from pipelines.State import Observer
from pipelines.losses import get_loss
from pd_explore import ocean_mask, variance_split, inversion_fraction, roughness, diversity, distance

LOSS_COLUMNS = ['zero_mean', 'mean_density', 'negative_grad_density', 'dist_isoton']


if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--folder', type=str, default=f'{os.environ.get("OCEANDATA", "")}/models/dino-fusion/tav0h83b/inference/test_new_sampler_clip/')
    parser.add_argument('--runs', type=str, nargs='*', default=['none', 'combo_*'], help='run folder names or glob patterns')
    parser.add_argument('--ref', type=str, default='none', help='run used for the distance')
    parser.add_argument('--csv', type=str, default=None, help='optional output csv')
    args = parser.parse_args()

    observer = Observer('data', TrainingConfig())
    mask = ocean_mask(observer)
    losses = {n : get_loss(n, observer) for n in LOSS_COLUMNS}
    ref = torch.tensor(np.load(f'{args.folder}/{args.ref}/generated.npy'))
    names = sorted({os.path.basename(os.path.dirname(p)) for r in args.runs for p in glob.glob(f'{args.folder}/{r}/generated.npy')})

    rows = []
    for name in names :
        x = torch.tensor(np.load(f'{args.folder}/{name}/generated.npy'))
        if not torch.isfinite(x).all() :
            rows.append({'run' : name})
            continue
        with torch.no_grad() :
            v = variance_split(x, mask)
            rows.append({'run' : name, 'inv' : inversion_fraction(x, observer).mean().item(),
                         'inv>0.01' : inversion_fraction(x, observer, tol=0.01).mean().item(),
                         'lvl' : v['level_mean'], 'anom' : v['anomaly'], 'rough' : roughness(x, mask)['all'],
                         'div' : diversity(x, mask)['all'], 'dist' : distance(x, ref, mask).mean().item(),
                         **{n : L(x).median().item() for n, L in losses.items()}})
    df = pd.DataFrame(rows).set_index('run')
    if args.csv is not None :
        df.to_csv(args.csv)
    print(df.to_string(float_format=lambda v : f'{v:.3g}'))
