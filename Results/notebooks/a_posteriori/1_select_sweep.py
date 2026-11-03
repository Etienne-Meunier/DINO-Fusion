#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1 for a sweep with several etas in one constraint folder, which the a posteriori pipeline would reduce to its latest
timestamp (settings.aposteriori_dirs): export each generation in a separate tree where each eta is its own constraint,
so that steps 2 to 5 run unchanged with `--prefix <sweep>_aposteriori`. The source generation folders are not written.

    python 1_select_sweep.py      # --force to redo a selection already converted (refused once launched)

The criterion is that of 1_select_median.py: upper median (paper common.median_member) of the RMSE (common.rmse) of the
volume-weighted zonal sigma_0 section of each member (common.zonal_sections) to the batch-mean section of the data batch
of paper/common.py. The runs start on forcing day 0, as those of the optimal sweep: selection.json has no start_day.

A folder whose selection has a conversion record (any protocol) is skipped unless --force; one with a NEMO run
(launch*.json or nemo_run*) is always skipped. A generation with a NaN or inf value, or a non-finite RMSE, is reported
and not exported, and the script then exits with status 1.

Inputs:
    <INFERENCE>/<sweep>/<constraint>/<timestamp>/generated.npy, args.json     normalised (b, 37, 208, 64), one eta each
    paper common data batch (loaded by the import)
Outputs in <INFERENCE>/<sweep>_aposteriori/<constraint>_<eta>/<timestamp>/aposteriori/, <timestamp> that of the generation:
    {toce,soce,ssh}.npy  un-normalised selected member, shapes (1, 36, 199, 62) and (1, 1, 199, 62)
    selection.json       as 1_select_median.py (member, RMSEs, criterion, model, generation and its args.json) plus eta
Printed: per eta, the selected member, its RMSE and the mean RMSE.
"""
from __init__ import PRP; import sys
sys.path.append(f'{PRP}/Diffusion_Model/')
sys.path.append(f'{PRP}/Results/notebooks/paper/')

import argparse
import json
import os
import time
from glob import glob

import numpy as np
import torch

from common import INFERENCE, RUN, batch, load_mask, median_member, observer, rmse, zonal_sections
from post_processing import save_clean
from settings import APOSTERIORI, LAUNCH, MODEL_REL, NEMO_RUN, PROTOCOLS, SELECTION, write_json

CRITERION = ('upper median of the unweighted RMSE of the volume-weighted zonal sigma_0 section of each member to the '
             'batch-mean section of a data batch (paper/fig1_density_profile.py)')


def skip_reason(out, force):
    """Why the selection in `out` must not be (re)written, None if it can: a NEMO run started from it, or a conversion
    record without `force`."""
    launched = glob(f'{out}/{os.path.splitext(LAUNCH)[0]}*.json') + glob(f'{out}/{NEMO_RUN}*')
    if launched:
        return f'{", ".join(map(os.path.basename, launched))} in {out}, the runs come from this selection'
    done = [p['conversion'] for p in PROTOCOLS.values() if os.path.exists(f'{out}/{p["conversion"]}')]
    if done and not force:
        return f'{", ".join(done)} in {out} (--force to overwrite the selection)'
    return None


def sweep_generations(sweep, constraint):
    """[(eta, generation folder, its args.json)] of the sweep, sorted by eta. Raises ValueError if there is none or if
    two folders share an eta: they would be one constraint, of which the later steps keep the latest timestamp."""
    found = []
    for path in glob(f'{INFERENCE}/{sweep}/{constraint}/*/generated.npy'):
        folder = os.path.dirname(path)
        with open(f'{folder}/args.json') as f:
            gen_args = json.load(f)
        found.append((float(gen_args['eta']), folder, gen_args))
    if not found:
        raise ValueError(f'no generation in {INFERENCE}/{sweep}/{constraint}')
    etas = [eta for eta, _, _ in found]
    if len(set(etas)) != len(etas):
        raise ValueError(f'several generations with the same eta in {INFERENCE}/{sweep}/{constraint} : {sorted(etas)}')
    return sorted(found, key=lambda g: g[0])


def eta_label(eta):
    """`eta` as written in a run name: an integer when whole."""
    return str(int(eta)) if eta.is_integer() else str(eta)


def non_finite_report(x, errors):
    """Description of the non-finite values of the batch `x` (b, c, i, j) and of its member `errors`, None if none."""
    bad_members = np.flatnonzero(~torch.isfinite(x).flatten(1).all(1).numpy()).tolist()
    bad_errors = np.flatnonzero(~np.isfinite(errors)).tolist()
    if not bad_members and not bad_errors:
        return None
    return f'NaN or inf values in members {bad_members}, non-finite RMSE of members {bad_errors}'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export the median member of every eta of a sweep for the a posteriori pipeline')
    parser.add_argument('--sweep', default='sweep_density_huber', help='sweep folder under the inference folder')
    parser.add_argument('--constraint', default='negative_grad_density_huber_border0', help='constraint folder of the sweep, one timestamp per eta')
    parser.add_argument('--force', action='store_true', help='overwrite selections already converted (refused once launched)')
    args = parser.parse_args()

    prefix = f'{args.sweep}_aposteriori'
    mask = load_mask()
    data_mean = np.nanmean(zonal_sections(batch, mask), axis=0)
    failed = []
    for eta, folder, gen_args in sweep_generations(args.sweep, args.constraint):
        name = f'{args.constraint}_{eta_label(eta)}'
        out = f'{INFERENCE}/{prefix}/{name}/{os.path.basename(folder)}/{APOSTERIORI}'
        reason = skip_reason(out, args.force)
        if reason:
            print(f'{name} : {reason}, skip')
            continue
        x = torch.from_numpy(np.load(f'{folder}/generated.npy'))
        errors = rmse(zonal_sections(x, mask), data_mean)
        report = non_finite_report(x, errors)
        if report:
            print(f'!!!!! {name} ({folder}) : {report}, NOTHING SELECTED !!!!!')
            failed.append(name)
            continue
        m = median_member(errors)
        print(f'{name} : eta {eta_label(eta)}, member {m} / {len(errors)}, RMSE {errors[m]:.4f} (mean {errors.mean():.4f})')

        save_clean(x[m:m + 1], out, observer.normalizer)
        write_json(f'{out}/{SELECTION}', dict(
            constraint=name, eta=eta, member=m, rmse=float(errors[m]), rmse_mean=float(errors.mean()),
            rmse_members=errors.tolist(), criterion=CRITERION, data_batch_size=len(batch), model=RUN,
            model_rel=MODEL_REL, prefix=prefix, generation_rel=os.path.relpath(folder, os.environ['OCEANDATA']),
            generation_args=gen_args, created=time.strftime('%Y-%m-%d %H:%M:%S')))
    if failed:
        sys.exit(f'!!!!! NaN or inf, nothing selected for : {", ".join(failed)} !!!!!')
