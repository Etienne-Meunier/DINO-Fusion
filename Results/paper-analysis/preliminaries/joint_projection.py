"""
Post-hoc joint projection of saved generations : `OptimProjection` (LBFGS) on mean_density + c * dist_isoton with eps = 0,
then BorderZero, as proposed to remove the profile bias left by the guided sampling without recreating inversions.
Each input folder `<run>` gets a sibling `<run>_joint<c>/generated.npy`.

    python joint_projection.py --folder <dir> --runs 'dist_isoton+zero_mean*x1.0-64.0*' --c 0.01
"""
from __init__ import PRP; import sys
sys.path.append(f'{PRP}/Diffusion_Model/')
sys.path.append(PRP)

import argparse, glob, os
import numpy as np
import torch
from configs.base_config import TrainingConfig
from pipelines.State import Observer
from pipelines.losses import get_loss, OptimProjection, BorderZero


class WeightedSum :
    """Per-sample sum of losses with weights, (b, c, i, j) -> (b,)."""
    def __init__(self, losses, weights) :
        self.losses, self.weights = losses, weights

    def __call__(self, field) :
        return sum(w * l(field) for l, w in zip(self.losses, self.weights))


if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--folder', type=str, default=f'{os.environ.get("OCEANDATA", "")}/models/dino-fusion/tav0h83b/inference/prelim_sweep_clip/')
    parser.add_argument('--runs', type=str, nargs='*', required=True, help='run folder names or glob patterns')
    parser.add_argument('--c', type=float, default=0.01, help='weight of dist_isoton')
    parser.add_argument('--steps', type=int, default=10, help='LBFGS steps of OptimProjection')
    args = parser.parse_args()

    observer = Observer('data', TrainingConfig())
    loss = WeightedSum([get_loss('mean_density', observer), get_loss('dist_isoton', observer)], [1.0, args.c])
    projection = OptimProjection(loss, eps=0.0, max_opt_steps=args.steps)
    border = BorderZero(observer)
    names = sorted({os.path.basename(os.path.dirname(p)) for r in args.runs for p in glob.glob(f'{args.folder}/{r}/generated.npy')})
    for name in names :
        if '_joint' in name :
            continue
        x = torch.tensor(np.load(f'{args.folder}/{name}/generated.npy'))
        out = f'{args.folder}/{name}_joint{args.c}'
        os.makedirs(out, exist_ok=True)
        np.save(f'{out}/generated.npy', border(projection(x)).numpy())
        print(f'Saved {out}')
