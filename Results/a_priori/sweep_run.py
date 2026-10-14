"""
Hyper-parameter sweep of the constraints of `Diffusion_Model/generate_constrained.py`.

    gradient   : initial beta (--beta) x step size of beta (--eta), for each loss
    gradient with --tweedie : step size of beta (--eta), initial beta 0, for each loss
    projection : step size of lambda (--eta) x step size of rho (--eta_rho), for each projection

Each run is written in `<sweep_path>/<run_name>/` and skipped if it already exists, so the script can be relaunched.
Analyse with `sweep_analysis.py`.
"""
from __init__ import PRP; import sys, os
import subprocess
from itertools import product
from pathlib import Path

model_path = f'{os.environ["OCEANDATA"]}/models/dino-fusion/tav0h83b'
sweep_path = f'{model_path}/inference/new_constraints/sweep'
common = ['--batch', '8', '--inf_steps', '1000', '--seed', '0']

# Loss values differ by ~20x between zero_mean (normalised fields) and the density losses (kg/m3)
gradient_grid = {
    'zero_mean' :             {'beta' : [0, 1e-1], 'eta' : [1e-2, 1e-1, 1]},
    'mean_density' :          {'beta' : [0, 1e-3], 'eta' : [1e-6, 1e-5, 1e-4]},
    'mean_grad_density' :     {'beta' : [0, 1e-3], 'eta' : [1e-6, 1e-5, 1e-4]},
    'negative_grad_density' : {'beta' : [0, 1e-3], 'eta' : [1e-6, 1e-5, 1e-4]},
}
# Loss on the Tweedie estimate (--tweedie). eta = s / (rms|grad_x0 L| * sum_t L) measured on an unconstrained trajectory,
# for a final step beta * rms|grad| of s = 1e-4, 1e-3, 1e-2 (normalised units per step)
tweedie_grid = {
    'zero_mean' :             [5e-4, 5e-3, 5e-2],
    'mean_density' :          [6e-6, 6e-5, 6e-4],
    'mean_grad_density' :     [1e-5, 1e-4, 1e-3],
    'negative_grad_density' : [7e-5, 7e-4, 7e-3],
    'dist_isoton' :           [1e-13, 1e-12, 1e-11],
}
projection_grid = {
    'mean_profile' : {'eta' : [0, 7e-3, 7e-2], 'eta_rho' : [3e-4, 3e-3, 3e-2]},
    'isotonic' :     {'eta' : [0, 7e-3, 7e-2], 'eta_rho' : [3e-4, 3e-3, 3e-2]},
}


def runs() :
    """
        Returns {run_name : list of arguments of generate_constrained.py}.
    """
    r = {'none' : ['--method', 'none'],
         # Control : BorderZero after each step without constraint gradient
         'border_zero_only' : ['--method', 'gradient', '--loss', 'mean_density', '--beta', '0', '--beta_type', 'constant']}
    for loss, grid in gradient_grid.items() :
        for beta, eta in product(grid['beta'], grid['eta']) :
            r[f'gradient_{loss}_beta={beta:g}_eta={eta:g}'] = ['--method', 'gradient', '--loss', loss, '--beta', str(beta), '--eta', str(eta)]
    for loss, etas in tweedie_grid.items() :
        for eta in etas :
            r[f'gradient_tweedie_{loss}_eta={eta:g}'] = ['--method', 'gradient', '--tweedie', '--loss', loss, '--beta', '0', '--eta', str(eta)]
    for proj, grid in projection_grid.items() :
        for eta, eta_rho in product(grid['eta'], grid['eta_rho']) :
            r[f'projection_{proj}_eta={eta:g}_etarho={eta_rho:g}'] = ['--method', 'projection', '--projection', proj, '--eta', str(eta), '--eta_rho', str(eta_rho)]
    return r


if __name__ == '__main__' :
    all_runs = runs()
    for i, (name, args) in enumerate(all_runs.items()) :
        output = f'{sweep_path}/{name}'
        if Path(f'{output}/clean/toce.npy').exists() :
            print(f'[{i+1}/{len(all_runs)}] {name} : done, skipped', flush=True)
            continue
        print(f'[{i+1}/{len(all_runs)}] {name} : running', flush=True)
        cmd = [sys.executable, f'{PRP}/Diffusion_Model/generate_constrained.py', '--model_path', model_path, '--output', output] + common + args
        res = subprocess.run(cmd, cwd=f'{PRP}/Diffusion_Model', capture_output=True, text=True)
        if res.returncode != 0 :
            print(f'[{i+1}/{len(all_runs)}] {name} : FAILED\n{res.stderr[-2000:]}', flush=True)
