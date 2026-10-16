"""
Distribution of each constraint loss (`pipelines.losses.LOSSES`) on the training states, to choose the tolerance eps of
`sampling.sample` : eps_c(q) = quantile_q(loss_c(data)), the loss of a generated state is then as low as the one of a
fraction q of the training states.

Outputs in `--out` :
    eps_data_losses.npz : per-sample losses {loss name : (n,)}
    eps_data_quantiles.csv : eps_c(q) for each loss c and quantile q
    eps_data_hist.png : histogram of each loss with the quantiles
"""
from __init__ import PRP; import sys
sys.path.append(f'{PRP}/Diffusion_Model/')
sys.path.append(PRP)

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from configs.base_config import TrainingConfig
from DataLoader import get_dataloader, get_infos
from TransformFields import TransformationPipeline
from pipelines.State import Observer
from pipelines.losses import LOSSES, get_loss

QUANTILES = [0.5, 0.9, 0.95, 0.99, 1.0]


def compute_losses(losses, dataloader, max_samples=None) :
    """
        Evaluate each loss on the normalised training states of `dataloader`.
            losses : {name : Loss}
            max_samples (int) : stop after this number of states, None for the whole dataset
        Returns :
            {name : np.array (n,)}
    """
    values = {name : [] for name in losses}
    n = 0
    with torch.no_grad() :
        for batch in dataloader :
            for name, loss in losses.items() :
                values[name].append(loss(batch).cpu().numpy())
            n += batch.shape[0]
            print(f'{n} states', end='\r')
            if max_samples is not None and n >= max_samples :
                break
    return {name : np.concatenate(v)[:max_samples] for name, v in values.items()}


def quantile_table(values, quantiles=QUANTILES) :
    """
        Returns {name : np.array (len(quantiles),)} the eps_c(q) of each loss.
    """
    return {name : np.quantile(v, quantiles) for name, v in values.items()}


def plot_histograms(values, eps, quantiles=QUANTILES, path=None) :
    """
        One panel per loss : histogram of log10(loss) over the training states with a vertical line per quantile.
        States with a loss exactly 0 cannot be placed on the log axis, their fraction is given in the title.
    """
    fig, axs = plt.subplots(1, len(values), figsize=(4*len(values), 3.5), squeeze=False)
    for ax, (name, v) in zip(axs[0], values.items()) :
        pos = v[v > 0]
        if len(pos) > 0 :
            ax.hist(np.log10(pos), bins=50, color='0.6')
        for q, e in zip(quantiles, eps[name]) :
            if e > 0 :
                ax.axvline(np.log10(e), ls='--', lw=1, color=plt.cm.viridis(q), label=f'q={q} : {e:.2e}')
        ax.set_title(f'{name}\n(loss = 0 : {np.mean(v == 0):.0%})')
        ax.set_xlabel('log10(loss)')
        if ax.get_legend_handles_labels()[0] :
            ax.legend(fontsize=7)
    fig.tight_layout()
    if path is not None :
        fig.savefig(path, dpi=150)
    return fig


if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--losses', nargs='*', choices=LOSSES, default=LOSSES)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--max_samples', type=int, default=None, help='default : whole training set')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--out', type=str, default=f'{PRP}/Results/paper-analysis/preliminaries/outputs/')
    args = parser.parse_args()

    import os; os.makedirs(args.out, exist_ok=True)
    config = TrainingConfig()
    transform = TransformationPipeline(get_infos(config.data_file), config.fields, config.normalisation, device=args.device)
    dataloader = get_dataloader(config.data_file, transform, batch_size=args.batch, shuffle=False)
    observer = Observer('data', config, device=args.device)
    losses = {name : get_loss(name, observer) for name in args.losses}




    values = compute_losses(losses, dataloader, args.max_samples)
    eps = quantile_table(values)

    np.savez(f'{args.out}/eps_data_losses.npz', **values)
    with open(f'{args.out}/eps_data_quantiles.csv', 'w') as f :
        f.write('loss,' + ','.join(f'q{q}' for q in QUANTILES) + '\n')
        for name, e in eps.items() :
            f.write(name + ',' + ','.join(f'{x:.6e}' for x in e) + '\n')
    plot_histograms(values, eps, path=f'{args.out}/eps_data_hist.png')

    n = len(next(iter(values.values())))
    print(f'\n{n} training states')
    print(open(f'{args.out}/eps_data_quantiles.csv').read())
