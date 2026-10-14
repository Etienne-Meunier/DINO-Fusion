"""
Campaign on the weight beta of the gradient constraints, with the loss evaluated on x_t or on the Tweedie estimate x0_hat (--tweedie).

    constant              : beta in a log scan, the scale of beta is set by the final step s = beta * rms|grad_x0 L|
    linear_up/linear_down : linear schedules with the same mean as the constant beta
    eq                    : beta <- beta + eta * C (etas of `sweep_run.tweedie_grid`)
    ineq                  : beta <- max(beta + eta * (C - eps), 0), Chamon dual ascent of C <= eps

Runs are written in `sweep_run.sweep_path` next to the sweep runs and skipped if done. The remaining runs of `sweep_run.py` are
included. Runs are launched by priority on `--workers` parallel processes.
Analyse with `sweep_analysis.py` (database) and `campaign_figures.py`.
"""
from __init__ import PRP; import sys, os
import argparse, subprocess, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from sweep_run import model_path, sweep_path, common, runs as sweep_runs, tweedie_grid

# rms|grad_x0 L| at the end of an unconstrained trajectory : beta = s / rms gives a final step s (normalised units)
rms_grad = {'zero_mean' : 1.26e-5, 'mean_grad_density' : 1.56e-4, 'negative_grad_density' : 7.25e-5, 'dist_isoton' : 1.36}
s_constant = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
s_linear = [3e-4, 3e-3, 3e-2]
# Loss of training batches of 8 states : zero_mean 0.069, mean_grad_density 0.013, negative_grad_density 0, dist_isoton 3e-4.
# negative_grad_density and dist_isoton use levels reached by the best runs of the sweep (data levels are never reached)
eps_grid = {'zero_mean' : [0.07, 0.7], 'mean_grad_density' : [0.013, 0.13], 'negative_grad_density' : [0.01, 0.1], 'dist_isoton' : [300, 3000]}


def runs() :
    """
        Returns {run_name : (priority, list of arguments of generate_constrained.py)}, lower priority first.
    """
    r = {name : (5, args) for name, args in sweep_runs().items()}
    for loss, rms in rms_grad.items() :
        for tweedie in [True, False] :
            prefix = f'gradient{"_tweedie" if tweedie else ""}_{loss}'
            base = ['--method', 'gradient', '--loss', loss] + (['--tweedie'] if tweedie else [])
            # Loss on x_t diverges for dist_isoton except with a weak constant / increasing beta
            unstable = loss == 'dist_isoton' and not tweedie
            for s in s_constant :
                r[f'{prefix}_constant_beta={s / rms:.3g}'] = (0, base + ['--beta_type', 'constant', '--beta', f'{s / rms:.3g}'])
            for eta in tweedie_grid[loss] :
                if not unstable :
                    r[f'{prefix}_eq_eta={eta:g}' if not tweedie else f'{prefix}_eta={eta:g}'] = (1, base + ['--beta_type', 'eq', '--beta', '0', '--eta', str(eta)])
            for eta in tweedie_grid[loss][1:] :
                for eps in eps_grid[loss] :
                    if not unstable :
                        r[f'{prefix}_ineq_eta={eta:g}_eps={eps:g}'] = (2, base + ['--beta_type', 'ineq', '--beta', '0', '--eta', str(eta), '--eps', str(eps)])
            for s in s_linear :
                for schedule in ['linear_up', 'linear_down'] :
                    if not (unstable and schedule == 'linear_down') :
                        r[f'{prefix}_{schedule}_beta={s / rms:.3g}'] = (3, base + ['--beta_type', schedule, '--beta', f'{s / rms:.3g}'])
    return r


def launch(name, args, log) :
    """
        Runs generate_constrained.py for one run and logs its duration.
    """
    t0 = time.time()
    cmd = [sys.executable, f'{PRP}/Diffusion_Model/generate_constrained.py', '--model_path', model_path, '--output', f'{sweep_path}/{name}'] + common + args
    res = subprocess.run(cmd, cwd=f'{PRP}/Diffusion_Model', capture_output=True, text=True)
    status = 'done' if res.returncode == 0 else f'FAILED\n{res.stderr[-2000:]}'
    print(f'{time.strftime("%H:%M:%S")} {name} : {status} ({(time.time() - t0) / 60:.1f} min)', file=log, flush=True)


if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    todo = sorted([(p, name, a) for name, (p, a) in runs().items() if not Path(f'{sweep_path}/{name}/clean/toce.npy').exists()])
    print(f'{len(todo)} runs to do', flush=True)
    with ThreadPoolExecutor(args.workers) as ex :
        for p, name, a in todo :
            ex.submit(launch, name, a, sys.stdout)
