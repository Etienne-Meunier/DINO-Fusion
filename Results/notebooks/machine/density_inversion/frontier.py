"""
Post-hoc frontier of density-only losses (prior : density increases with depth, nothing else). From the unconstrained
states x0, solve
    min_x  D(x, x0) + lam * L(x) / L(x0)
with D the mean squared difference over ocean cells (normalised space) and L a candidate loss, by LBFGS, for a log
sweep of lam, then BorderZero. Each point records the eval density error E (float64, volume-weighted) split by |g|,
the distance to x0, the roughness and the level-mean / anomaly variances.

    python frontier.py --losses C0 C1 --lams 1e-4 1e-2 1 --iters 100

Output : `frontier.csv` (one row per loss, lam, sample) in this folder.
"""
from pathlib import Path
import sys, argparse, time
PRP = f'{Path(__file__).resolve().parents[4]}/'
sys.path.append(f'{PRP}/Diffusion_Model/')
sys.path.append(f'{PRP}/Results/paper-analysis/preliminaries/')
sys.path.append(PRP)

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from configs.base_config import TrainingConfig
from pipelines.State import Observer
from pipelines.losses import get_loss, BorderZero, IsotonicDensity
from pd_explore import ocean_mask, variance_split, roughness
from step1_bins import R, HERE, BIN_NAMES, physical, split_E


class DensityLoss :
    """
        Per-sample density-only losses on the normalised state x (b, c, i, j) -> (b,), built on
        g = rho[k+1] - rho[k] (file levels, inversion where g < 0) and w = volume of the lower cell, normalised to sum 1
        over the wet pairs (the weighting of the eval density error).
    """
    def __init__(self, observer, kind, tau=1e-4) :
        self.observer, self.kind, self.tau = observer, kind, tau
        tmask = observer.tmask.bool()
        self.valid = tmask[1:] & tmask[:-1]
        w = observer.volume[1:] * self.valid
        self.w = w / w.sum()
        self.w_level = self.w.sum(dim=(1, 2))
        self.isoton = IsotonicDensity()
        self.col_cells = observer.tmask.sum(dim=0).clamp(min=1)
        self.col_w = observer.volume.sum(dim=0) / observer.volume.sum()

    def g(self, x) :
        rho = self.observer.density(x)
        g = rho[:, 1:] - rho[:, :-1]
        return torch.where(self.valid, g, torch.zeros_like(g))

    def __call__(self, x) :
        g = self.g(x)
        if self.kind == 'C1' :                                    # volume-weighted squared hinge
            return (self.w * F.relu(-g) ** 2).sum(dim=(1, 2, 3))
        if self.kind == 'C2' :                                    # volume-weighted linear hinge, softplus-smoothed
            return (self.w * self.tau * F.softplus(-g / self.tau)).sum(dim=(1, 2, 3))
        if self.kind == 'C7' :                                    # volume-weighted one-sided Huber hinge : zero for g >= 0, no margin
            h = torch.where(g < -self.tau, -g - self.tau / 2, g ** 2 / (2 * self.tau))
            return (self.w * torch.where(g < 0, h, torch.zeros_like(g))).sum(dim=(1, 2, 3))
        if self.kind == 'C3' :                                    # volume-weighted smooth indicator, tends to E / 100
            return (self.w * torch.sigmoid(-g / self.tau)).sum(dim=(1, 2, 3))
        if self.kind == 'C6' :                                    # volume-weighted (non-squared) distance of each column to the isotonic cone
            rho = self.observer.density(x)
            wet = self.observer.tmask.bool()
            # land lies below the sea floor : a filler denser than any water keeps the columns monotone there
            rho = torch.where(wet, rho, torch.full_like(rho, 1100.0))
            r = torch.where(wet, rho - self.isoton(rho.detach()), torch.zeros_like(rho))
            # unweighted along the column : P is the unweighted isotonic projection, so detaching it gives the exact gradient
            d = (r ** 2).sum(dim=1) / self.col_cells                                # (b, i, j) mean squared residual
            return (self.col_w * (d + self.tau ** 2).sqrt()).sum(dim=(1, 2))
        if self.kind == 'C5' :                                    # monotonicity of the sample's own horizontal-mean profile
            n = self.valid.sum(dim=(1, 2)).clamp(min=1)
            g_mean = g.sum(dim=(2, 3)) / n
            return (self.w_level * F.relu(-g_mean) ** 2).sum(dim=1)
        raise ValueError(self.kind)


class Sum :
    def __init__(self, *losses) :
        self.losses = losses

    def __call__(self, x) :
        return sum(l(x) for l in self.losses)


def get_candidate(name, observer) :
    """Candidate loss by name, see the module docstring of PLAN2.md."""
    if name == 'C0' :
        return get_loss('negative_grad_density', observer)
    if name == 'C4' :
        return get_loss('dist_isoton', observer)
    if name[:3] in ('C2_', 'C3_', 'C6_', 'C7_') :
        return DensityLoss(observer, name[:2], tau=float(name[3:]))
    if name == 'C3anneal' :
        return DensityLoss(observer, 'C3', tau=1e-2)
    if name == 'C1+C5' :
        c1, c5 = DensityLoss(observer, 'C1'), DensityLoss(observer, 'C5')
        return Sum(lambda x : c1(x), lambda x : c5(x))
    return DensityLoss(observer, name)


def project(x0, loss, lam, mask, iters, x_init=None, taus=None) :
    """
        LBFGS on D(x, x0) + lam * loss(x) / loss(x0), per-sample terms summed over the batch, started from x_init
        (warm start along the lam sweep, default x0). taus : tau continuation of a sigmoid loss, iters split evenly.
    """
    x = (x0 if x_init is None else x_init).clone().requires_grad_(True)
    for tau in (taus or [None]) :
        if tau is not None :
            loss.tau = tau
        with torch.no_grad() :
            scale = loss(x0).clamp(min=1e-30)
        opt = torch.optim.LBFGS([x], history_size=20, max_iter=iters // len(taus or [None]), line_search_fn='strong_wolfe')
        def closure() :
            opt.zero_grad()
            d = ((x - x0)[:, mask] ** 2).mean(dim=1)
            obj = (d + lam * loss(x) / scale).sum()
            obj.backward()
            return obj
        opt.step(closure)
    return x.detach()


def evaluate(x, x0, observer, mask) :
    """Per-sample rows : E and its |g| bins (float64), distance to x0, plus batch-level roughness and variances."""
    T, S = physical(x, observer)
    E, E_bins, _, _, _ = split_E(T, S, observer.volume, torch.float64)
    dist = ((x - x0)[:, mask] ** 2).mean(dim=1).sqrt()
    batch = {'rough' : roughness(x, mask)['all'], **variance_split(x, mask)}
    return [{'sample' : b, 'E' : E[b].item(), **{f'E_{n}' : E_bins[b, i].item() for i, n in enumerate(BIN_NAMES)},
             'dist' : dist[b].item(), **batch} for b in range(len(x))]


if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--losses', nargs='*', default=['C0', 'C1', 'C2_1e-4', 'C2_1e-3', 'C3_1e-3', 'C3anneal', 'C4', 'C5', 'C1+C5'])
    parser.add_argument('--lams', type=float, nargs='*', default=[1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4, 1e5])
    parser.add_argument('--iters', type=int, default=300)
    parser.add_argument('--out', type=str, default='frontier.csv')
    args = parser.parse_args()

    observer = Observer('data', TrainingConfig())
    mask = ocean_mask(observer)
    border = BorderZero(observer)
    x0 = torch.tensor(np.load(f'{R}/none/generated.npy'))
    rows = [{'loss' : 'none', 'lam' : 0.0, 'iters' : 0, 'time' : 0.0, **r} for r in evaluate(x0, x0, observer, mask)]
    for name in args.losses :
        loss = get_candidate(name, observer)
        taus = [1e-2, 1e-3, 1e-4] if name == 'C3anneal' else None
        x = None
        for lam in sorted(args.lams) :
            t = time.time()
            x = border(project(x0, loss, lam, mask, args.iters, x_init=x, taus=taus))
            dt = time.time() - t
            ev = evaluate(x, x0, observer, mask)
            rows += [{'loss' : name, 'lam' : lam, 'iters' : args.iters, 'time' : dt, **r} for r in ev]
            print(f'{name:8s} lam {lam:.0e}  E {np.mean([r["E"] for r in ev]):.3f}  dist {np.mean([r["dist"] for r in ev]):.4f}  ({dt:.0f}s)', flush=True)
            pd.DataFrame(rows).to_csv(HERE / args.out, index=False)
