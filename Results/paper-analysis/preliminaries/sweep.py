"""
Sweep of the constrained sampler `pd_explore.sample_pd` (always with the BorderZero projection).

Each run is saved in `<out>/<run name>/` :
    generated.npy  normalised states (b, 37, 208, 64), `Diffusion_Model/post_processing.py --generated` un-normalises them
    lbd.npy        final dual variables (b,)
    trace.npz      per-step trace of `sample_pd`
    metrics.json   arguments, final losses of every constraint, distance to the unconstrained state of the same seed,
                   diversity over the batch, run time
Runs whose `metrics.json` exists are skipped, so a sweep can be restarted. The unconstrained run of each (seed, steps,
batch) is generated first, as `none_s<seed>_n<steps>_b<batch>`.

eps is given as a quantile of the training losses (`eps_data.py`), eps_c(q) read in `--eps_csv`, times `eps_scale`
(default 1, > 1 to relax the constraint beyond the data, one value per loss for stacked losses), floored at EPS_FLOOR for the losses that are 0 on the data.
"""
from __init__ import PRP; import sys
sys.path.append(f'{PRP}/Diffusion_Model/')
sys.path.append(PRP)

import argparse, itertools, json, os, time
from pathlib import Path
import numpy as np
import torch
from accelerate import Accelerator
from configs.base_config import TrainingConfig
from pipelines.pipeline_tensor import DDPMPipeline_Tensor
from pipelines.State import Observer
from pipelines.losses import BorderZero, LOSSES, get_loss
from pd_explore import sample_pd, ocean_mask, diversity, distance, Stacked

EPS_FLOOR = 1e-6
DEFAULT_OUT = f'{os.environ.get("OCEANDATA", "")}/models/dino-fusion/tav0h83b/inference/prelim_sweep_clip/'


def grid(**axes) :
    """List of run dictionaries, cartesian product of the `axes` lists."""
    keys = list(axes)
    return [dict(zip(keys, v)) for v in itertools.product(*axes.values())]


GRIDS = {
    'smoke' : grid(loss=['dist_isoton'], mode=['log'], eta=[1.0], q=[0.5], t_start=[None]),
    'calib' : sum([grid(loss=[l], mode=['raw'], eta=list(etas), q=[0.5]) for l, etas in {
                    'mean_density' : 10.0**np.arange(-5, 2), 'mean_grad_density' : 10.0**np.arange(-5, 2),
                    'negative_grad_density' : 10.0**np.arange(-5, 2), 'zero_mean' : 10.0**np.arange(-4, 3),
                    'dist_isoton' : 10.0**np.arange(-10, -3)}.items()], [])
            + grid(loss=LOSSES, mode=['sqrt'], eta=[1e-3, 1e-2, 1e-1, 1.0, 10.0], q=[0.5]),
    'relax' : grid(loss=['zero_mean'], mode=['raw'], eta=[10.0], q=[0.5], eps_scale=[0.25, 1.0, 4.0, 16.0, 64.0])
            + grid(loss=['zero_mean'], mode=['raw'], eta=[1.0, 3.0, 30.0], q=[0.5]),
    'variants_sqrt' : grid(loss=['mean_density', 'mean_grad_density'], mode=['sqrt'], eta=[0.3, 1.0, 3.0], q=[0.5], t_start=[700])
            + grid(loss=['mean_density', 'mean_grad_density'], mode=['sqrt'], eta=[1.0], q=[0.5], final_opt=[10, 50])
            + grid(loss=['mean_density', 'mean_grad_density'], mode=['sqrt'], eta=[1.0, 3.0], q=[0.5], lbd_max=[20.0])
            + grid(loss=['mean_density'], mode=['sqrt'], eta=[1.0], q=[0.5], through_unet=[False])
            + grid(loss=['mean_density', 'mean_grad_density'], mode=['sqrt'], eta=[0.3, 3.0], q=[0.5])
            + grid(loss=['dist_isoton'], mode=['sqrt'], eta=[1e-5, 1e-4], q=[0.5])
            + grid(loss=['dist_isoton'], mode=['sqrt'], eta=[1e-4, 1e-3, 1e-2], q=[0.5], t_start=[300])
            + grid(loss=['dist_isoton'], mode=['sqrt'], eta=[1e-4, 1e-3], q=[0.5], through_unet=[False])
            + grid(loss=['mean_density+zero_mean', 'dist_isoton+zero_mean'], mode=['sqrt'], eta=[[1.0, 1.0], [1.0, 10.0]], q=[0.5]),
    'isoton' : grid(loss=['dist_isoton'], mode=['sqrt'], eta=[3e-5, 1e-4, 3e-4], q=[0.5], through_unet=[False])
            + grid(loss=['dist_isoton'], mode=['sqrt'], eta=[3e-5, 3e-4], q=[0.5])
            + grid(loss=['dist_isoton'], mode=['sqrt'], eta=[1e-4], q=[0.5], final_opt=[10])
            + grid(loss=['dist_isoton+zero_mean'], mode=['sqrt'], eta=[[1e-4, 10.0], [3e-5, 10.0], [1e-4, 1.0]], q=[0.5])
            + grid(loss=['dist_isoton+mean_density'], mode=['sqrt'], eta=[[1e-4, 1.0], [3e-5, 1.0]], q=[0.5]),
    'isoton_seeds' : grid(loss=['dist_isoton'], mode=['sqrt'], eta=[1e-4], q=[0.5])
            + grid(loss=['dist_isoton+zero_mean'], mode=['sqrt'], eta=[[1e-4, 10.0]], q=[0.5])
            + grid(loss=['zero_mean'], mode=['raw'], eta=[10.0], q=[0.5])
            + grid(loss=['mean_density'], mode=['sqrt'], eta=[1.0], q=[0.5]),
    'combo' : grid(loss=['dist_isoton+zero_mean'], mode=['sqrt'], eta=[[1e-4, 1.0], [1e-4, 3.0], [3e-5, 1.0]], q=[0.5]),
    'combo_fo' : grid(loss=['dist_isoton+zero_mean'], mode=['sqrt'], eta=[[1e-4, 1.0]], q=[0.5], final_opt=[10]),
    'triple' : grid(loss=['dist_isoton+zero_mean+mean_density'], mode=['sqrt'], eta=[[1e-4, 1.0, 1.0], [1e-4, 1.0, 0.3], [1e-4, 0.5, 1.0]], q=[0.5]),
    'combo_relax' : grid(loss=['dist_isoton+zero_mean'], mode=['sqrt'], eta=[[1e-4, 1.0]], q=[0.5], eps_scale=[[1.0, 64.0], [1.0, 128.0], [1.0, 256.0]]),
    'triple2' : grid(loss=['dist_isoton+zero_mean+mean_density'], mode=['sqrt'], eta=[[2e-4, 1.0, 0.3], [3e-4, 1.0, 0.3], [1e-4, 1.0, 0.1]], q=[0.5]),
    'posthoc' : grid(loss=['mean_density', 'mean_grad_density', 'negative_grad_density', 'zero_mean'], mode=['raw'], eta=[0.0], q=[0.5], final_opt=[10])
            + grid(loss=['zero_mean'], mode=['raw'], eta=[1.0], q=[0.5], final_opt=[10])
            + grid(loss=['negative_grad_density'], mode=['sqrt'], eta=[1.0], q=[0.5], final_opt=[10]),
    'variants' : grid(loss=['mean_density', 'negative_grad_density'], mode=['raw'], eta=[1e-2, 3e-2, 1e-1], q=[0.5], t_start=[700, 400])
            + grid(loss=['mean_density', 'negative_grad_density'], mode=['raw'], eta=[1e-1, 1.0], q=[0.5], lbd_max=[2.0, 5.0])
            + grid(loss=['mean_density', 'negative_grad_density'], mode=['raw'], eta=[1e-2, 1e-1], q=[0.5], through_unet=[False])
            + grid(loss=['mean_density', 'negative_grad_density'], mode=['raw'], eta=[1e-2], q=[0.5], final_opt=[10, 50])
            + grid(loss=['mean_density+zero_mean'], mode=['raw'], eta=[[1e-2, 1e-2], [1e-2, 1e-1], [3e-2, 1e-1]], q=[0.5]),
}


def read_eps(path) :
    """{loss : {q : eps}} from the csv of `eps_data.py`."""
    lines = open(path).read().split()
    qs = [float(c[1:]) for c in lines[0].split(',')[1:]]
    return {l.split(',')[0] : dict(zip(qs, map(float, l.split(',')[1:]))) for l in lines[1:]}


def run_name(r) :
    """Folder name of a run dictionary."""
    if r['loss'] == 'none' :
        return f"none_s{r['seed']}_n{r['steps']}_b{r['batch']}"
    keys = ['loss', 'mode', 'eta', 'q', 'eps_scale', 't_start', 'lbd_max', 'final_opt', 'through_unet', 'seed', 'steps', 'batch']
    short = {'loss' : '', 'mode' : '', 'eta' : 'eta', 'q' : 'q', 'eps_scale' : 'x', 't_start' : 'ts', 'lbd_max' : 'lm', 'final_opt' : 'fo',
             'through_unet' : 'tu', 'seed' : 's', 'steps' : 'n', 'batch' : 'b'}
    fmt = lambda v : '-'.join(map(str, v)) if isinstance(v, (list, tuple)) else v
    return '_'.join(f'{short[k]}{fmt(r[k])}' for k in keys if r.get(k) is not None)


class Runner :
    """Holds the model, the losses and the unconstrained references, and runs / saves one configuration at a time."""
    def __init__(self, model_path, out, eps_csv) :
        self.device = Accelerator().device
        self.pipeline = DDPMPipeline_Tensor.from_pretrained(model_path).to(self.device)
        self.pipeline.set_progress_bar_config(disable=True)
        self.observer = Observer('data', TrainingConfig(), device=self.device)
        self.losses = {n : get_loss(n, self.observer) for n in LOSSES}
        self.projection = BorderZero(self.observer)
        self.mask = ocean_mask(self.observer).to(self.device)
        self.eps = read_eps(eps_csv)
        self.out = out

    def run(self, r) :
        """Run the configuration `r` (defaults filled by the caller) unless already saved, returns its metrics."""
        path = Path(self.out) / run_name(r)
        if (path / 'metrics.json').exists() :
            return json.load(open(path / 'metrics.json'))
        ref = None
        if r['loss'] != 'none' :
            ref = self.run({**r, 'loss' : 'none'})
            ref = torch.tensor(np.load(Path(self.out) / ref['name'] / 'generated.npy')).to(self.device)

        self.pipeline.scheduler.set_timesteps(r['steps'])
        loss_c, eps, names = None, None, r['loss'].split('+')
        if r['loss'] != 'none' :
            scales = np.broadcast_to(r.get('eps_scale', 1.0), len(names))
            eps = [max(self.eps[n][r['q']] * float(x), EPS_FLOOR) for n, x in zip(names, scales)]
            loss_c = Stacked([self.losses[n] for n in names]) if len(names) > 1 else self.losses[names[0]]
            eps = eps if len(names) > 1 else eps[0]
        t0 = time.time()
        x, lbd, trace = sample_pd(self.pipeline, r['batch'], loss_c=loss_c, eta=r.get('eta', 0.0), eps=eps,
                                  projection=self.projection, seed=r['seed'], mode=r.get('mode', 'raw'),
                                  t_start=r.get('t_start'), lbd_max=r.get('lbd_max'), final_opt=r.get('final_opt') or 0,
                                  through_unet=r.get('through_unet', True))
        elapsed = time.time() - t0

        with torch.no_grad() :
            finite = bool(torch.isfinite(x).all())
            final = {n : L(x).cpu().numpy().tolist() for n, L in self.losses.items()}
            metrics = {'name' : path.name, 'args' : r, 'eps' : eps, 'finite' : finite, 'time' : elapsed,
                       'losses' : final, 'diversity' : diversity(x, self.mask) if finite else None,
                       'dist_unc' : distance(x, ref, self.mask).cpu().numpy().tolist() if ref is not None and finite else None,
                       'lbd' : None if lbd is None else lbd.cpu().numpy().tolist()}
            if loss_c is not None :
                ok = np.all([np.array(final[n]) <= e for n, e in zip(names, np.atleast_1d(eps))], axis=0)
                metrics['satisfied'] = float(np.mean(ok)) if finite else 0.0

        path.mkdir(parents=True, exist_ok=True)
        np.save(path / 'generated.npy', x.cpu().numpy())
        if lbd is not None :
            np.save(path / 'lbd.npy', lbd.cpu().numpy())
        np.savez(path / 'trace.npz', **trace)
        json.dump(metrics, open(path / 'metrics.json', 'w'), indent=1)
        return metrics


def summary(m) :
    """One line summary of the metrics of a run."""
    a = m['args']
    if not m['finite'] :
        return f"{m['name']:60s} DIVERGED"
    names = a['loss'].split('+') if a['loss'] != 'none' else ['mean_density']
    eps = np.atleast_1d(m['eps'] if m['eps'] is not None else [0.0] * len(names))
    losses = ' '.join(f"{n}={np.median(m['losses'][n]):.2e}/{e:.1e}" for n, e in zip(names, eps))
    sat = m.get('satisfied', float('nan'))
    d = np.mean(m['dist_unc']) if m['dist_unc'] is not None else 0.0
    return f"{m['name']:60s} {losses} sat={sat:.2f} dist={d:.3f} div={m['diversity']['all']:.3f} t={m['time']:.0f}s"


if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--grid', choices=GRIDS.keys(), default='smoke')
    parser.add_argument('--model_path', type=str, default=f'{os.environ.get("OCEANDATA", "")}/models/dino-fusion/tav0h83b')
    parser.add_argument('--out', type=str, default=DEFAULT_OUT)
    parser.add_argument('--eps_csv', type=str, default=f'{PRP}/Results/paper-analysis/preliminaries/outputs/eps_data_quantiles.csv')
    parser.add_argument('--seeds', type=int, nargs='*', default=[0])
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--batch', type=int, default=8)
    args = parser.parse_args()

    runner = Runner(args.model_path, args.out, args.eps_csv)
    for seed in args.seeds :
        for r in GRIDS[args.grid] :
            m = runner.run({'seed' : seed, 'steps' : args.steps, 'batch' : args.batch, **r})
            print(summary(m), flush=True)
