"""
Load generated states and a batch of training states for the a priori evaluation.

Generated states are folders with `{toce,soce,ssh}.npy` (toce, soce : (b, 36, 199, 62), ssh : (b, 1, 199, 62)),
written by `Diffusion_Model/post_processing.py` or `Diffusion_Model/generate_constrained.py` (`clean/`).

Sets of generations (`--set`) :
    paper : generations of the paper (constraints TS, C1, C2 and no constraint)
    new   : generations of `generate_constrained.py`, with the paper generations as reference
"""
from __init__ import PRP; import sys, os
sys.path.append(f'{PRP}/Diffusion_Model/')
sys.path.append(f'{PRP}/Results/analysis_scripts/')

import argparse
import numpy as np
import xarray as xr
from DataLoader import get_dataloader
from configs.base_config import TrainingConfig

models_path = f'{os.environ["OCEANDATA"]}/models/dino-fusion/'
generated_path = f'{models_path}/Publication-Blandine/generated_npy_files/'
new_path = f'{models_path}/tav0h83b/inference/new_constraints/'

paper_path = {
    'constraint_C1' : f'{generated_path}/chamon_C1_clean',
    'constraint_C2' : f'{generated_path}/chamon_C2_clean',
    'no_constraint' : f'{models_path}/tav0h83b/inference/infesteps_1000/constraints_no_constraints/20250131-110120_clean',
    'constraint_TS' : f'{generated_path}/constraint_TS_clean'
    }

new_runs = ['none', 'gradient_mean_density', 'gradient_mean_grad_density', 'gradient_negative_grad_density',
            'projection_mean_profile', 'projection_isotonic']

SETS = {
    'paper' : {
        'path' : paper_path,
        # Paper Table 1 uses a larger generation (200 states) for the unconstrained model
        'path_box_metrics' : paper_path | {'no_constraint' : f'{models_path}/tav0h83b/inference/infesteps_1000/constraints_no_constraints/20250203-141645_clean'},
        'spatial' : ['constraint_C1', 'constraint_C2'],
        'variability' : ['no_constraint', 'constraint_C2', 'constraint_TS'],
        'density' : ['constraint_C1', 'constraint_C2', 'no_constraint'],
    },
    'new' : {
        'path' : paper_path | {k : f'{new_path}/{k}/clean' for k in new_runs},
        'spatial' : ['projection_isotonic', 'gradient_mean_grad_density'],
        'variability' : ['none', 'constraint_C2', 'gradient_mean_grad_density', 'projection_isotonic'],
        'density' : ['none', 'constraint_C2'] + new_runs[1:],
    },
}

LABELS = {
    'no_constraint' : 'No constraint', 'constraint_TS' : 'Constraint TS', 'constraint_C1' : 'Constraint C1', 'constraint_C2' : 'Constraint C2',
    'none' : 'No constraint (new)', 'gradient_mean_density' : 'Gradient mean density', 'gradient_mean_grad_density' : 'Gradient mean density gradient',
    'gradient_negative_grad_density' : 'Gradient negative density gradient', 'projection_mean_profile' : 'Projection mean profile',
    'projection_isotonic' : 'Projection isotonic',
}

UNCONSTRAINED = ['no_constraint', 'none']


def get_args(description='') :
    """
        Parse `--set` (paper | new) and `--gens` (generations to plot, default from the set).
        Returns the set dictionary with its 'name', 'figures_path' and 'gens' (None if not given).
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--set', choices=SETS.keys(), default='paper')
    parser.add_argument('--gens', nargs='*', default=None, help='generations to plot (keys of the set)')
    args = parser.parse_args()
    return SETS[args.set] | {'name' : args.set, 'figures_path' : f'{PRP}/Results/a_priori/figures/{args.set}/', 'gens' : args.gens}


def load_generations(path) :
    """
        Return {generation_name : {'ssh', 'soce', 'toce' : np.array}} for each folder in `path`.
    """
    bg = {}
    for key, p in path.items() :
          bg[key] = {'ssh' : np.load(p + '/ssh.npy'),
                     'soce' : np.load(p + '/soce.npy'),
                     'toce' : np.load(p + '/toce.npy')}
    return bg


def load_data_batch(batch_size=8, data_file=None) :
    """
        Return a random batch of un-normalised training states {'toce', 'soce', 'ssh' : tensor (b, z, y, x)}.
    """
    data_file = TrainingConfig().data_file if data_file is None else data_file
    train_dataloader = get_dataloader(data_file, transform=None, batch_size=batch_size, shuffle=True)
    return next(iter(train_dataloader))


def load_mask() :
    """
        Return the 1° DINO mesh mask with dimensions renamed to (depth, nav_lat, nav_lon).
    """
    file_mask_LR = xr.open_dataset(f"{PRP}/Results/analysis_scripts/data/DINO_1deg_mesh_mask_david_renamed.nc").sel(time_counter=0)
    return file_mask_LR.rename({"nav_lev":"depth","y":"nav_lat","x":"nav_lon"})
