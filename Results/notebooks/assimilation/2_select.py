#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2 of the assimilation study (see plan.md), step 1 of the NEMO integration: export one initial state per season
and config in the layout of the a posteriori pipeline (`a_posteriori/1_select_median.py`), so that its steps 2 to 5
run unchanged with `--prefix assimilation/phase2` (and `--experiment daily_2w_minimal` from step 3 on).

    python 2_select.py            # --force to redo a selection already converted (refused once launched)

Per season (phase1.SOURCES) and config of CONFIGS:
    - data: the source state itself in the model representation (phase1.normalised_state: the 18 model T/S levels,
      odd levels interpolated back, row y=1 copied from y=2 on export), the same round trip as the generated states
      and as the data run of the a posteriori pipeline, member 0;
    - none, B0, C7, TS: the SST-conditioned generation of phase 1 (phase1_standard/<season>_<config>), member at the
      upper median (paper common.median_member) of the RMSE (common.rmse) of the volume-weighted zonal sigma_0
      section of each member (common.zonal_sections) to the section of the source state of the season.
The NEMO start follows the phase of the source: phase p is forcing day of year start_day = YEAR_DAYS / seasonal.YEAR
* p (360-day calendar), stored in selection.json for step 2, which sets the restart kt, time_counter, adatrj and ndastp
from it (convert_state.start_calendar): the DINO seasonal forcing takes its phase from kt, not from the date. Summer
(phase 24) starts at day 240 (kt 10828800, ndastp 2500830), winter (phase 7) at day 70 (kt 10808400, ndastp 2500310).
The ndastp of selection.json is the expected restart date, for the record.

A folder whose selection has a conversion record (any protocol) is skipped unless --force; one with a NEMO run
(launch*.json or nemo_run*) is always skipped, as in 1_select_median.py.

Inputs:
    phase1.RUN_SETS['standard']/<season>_<config>/generated.npy, args.json   normalised (16, 37, 208, 64)
    phase1.CACHE1                                                             last-year data states (sources)
    paper common data batch (only loaded by the import)
Outputs in OUT2/<season>_<config>/<ddmmyy-HHMMSS>/aposteriori/ (the latest existing timestamp folder is reused):
    {toce,soce,ssh}.npy  un-normalised selected member, shapes (1, 36, 199, 62) and (1, 1, 199, 62)
    selection.json       as 1_select_median.py (member, RMSEs, criterion, model, generation and its args.json) plus
                         season, source_index, source_phase, config, start_day, ndastp, data_file
Printed: per run, the selected member, its RMSE and the mean RMSE.
"""
from __init__ import PRP; import sys
sys.path.append(f'{PRP}/Diffusion_Model/')
sys.path.append(f'{PRP}/Results/notebooks/paper/')
sys.path.append(f'{PRP}/Results/notebooks/a_posteriori/')

import argparse
import json
import os
import time
from glob import glob

import numpy as np
import torch

from common import RUN, load_mask, median_member, observer, rmse, zonal_sections
from post_processing import save_clean
from settings import APOSTERIORI, LAUNCH, MODEL_REL, NEMO_RUN, PROTOCOLS, SELECTION, timestamp, write_json

from phase1 import CACHE1, RUN_SETS, SOURCES, load_states, normalised_state
from seasonal import DATA_FILE, OUT, YEAR

PREFIX2 = 'assimilation/phase2'
OUT2 = OUT.replace('phase0', 'phase2')
GENERATIONS = RUN_SETS['standard']
SEASONS = ['summer', 'winter']
CONFIGS = ['data', 'none', 'B0', 'C7', 'TS']
YEAR_DAYS, MONTH_DAYS, START_YEAR = 360, 30, 250
CRITERION = ('upper median of the unweighted RMSE of the volume-weighted zonal sigma_0 section of each member to the '
             'section of the source state of the season (paper common.zonal_sections, rmse, median_member)')
CRITERION_DATA = 'the source state of the season itself (phase1.normalised_state), single member'


def start_day(phase):
    """Forcing day of year (360-day calendar) of the training phase `phase`."""
    return YEAR_DAYS // YEAR * phase


def ndastp(day, year=START_YEAR):
    """Restart date YYYYMMDD (3-digit year) of day of year `day` of `year`, MONTH_DAYS-day months: NEMO starts the
    next day."""
    return year * 10000 + ((day - 1) // MONTH_DAYS + 1) * 100 + (day - 1) % MONTH_DAYS + 1


def output_folder(name):
    """aposteriori/ folder of the run `name` (<season>_<config>): that of the latest OUT2/<name>/<timestamp>/, a new
    timestamp folder if there is none."""
    existing = glob(f'{OUT2}/{name}/*/')
    stamp = (os.path.basename(os.path.normpath(max(existing, key=timestamp))) if existing
             else time.strftime('%d%m%y-%H%M%S'))
    return f'{OUT2}/{name}/{stamp}/{APOSTERIORI}'


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


def candidates(season, config, states):
    """(normalised batch (b, 37, 208, 64) tensor, generation folder or None, its args.json or None) of `config` in
    `season`: the source state alone for data, the phase 1 generation otherwise."""
    if config == 'data':
        return normalised_state(states, SOURCES[season], observer)[None].cpu(), None, None
    folder = f'{GENERATIONS}/{season}_{config}'
    with open(f'{folder}/args.json') as f:
        args = json.load(f)
    if args['source_index'] != SOURCES[season]:
        raise ValueError(f'{folder} : source {args["source_index"]} != {SOURCES[season]}')
    return torch.from_numpy(np.load(f'{folder}/generated.npy')), folder, args


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export the phase 2 initial states for the a posteriori pipeline')
    parser.add_argument('--force', action='store_true', help='overwrite selections already converted (refused once launched)')
    args = parser.parse_args()

    mask = load_mask()
    states = load_states()
    for season in SEASONS:
        source = zonal_sections(normalised_state(states, SOURCES[season], observer)[None].cpu(), mask)[0]
        for config in CONFIGS:
            name = f'{season}_{config}'
            out = output_folder(name)
            reason = skip_reason(out, args.force)
            if reason:
                print(f'{name} : {reason}, skip')
                continue
            x, folder, gen_args = candidates(season, config, states)
            errors = rmse(zonal_sections(x, mask), source)
            m = median_member(errors)
            print(f'{name:14s} member {m:2d} / {len(errors)}, RMSE {errors[m]:.4f} (mean {errors.mean():.4f})')

            save_clean(x[m:m + 1], out, observer.normalizer)
            phase = int(SOURCES[season] % YEAR)
            write_json(f'{out}/{SELECTION}', dict(
                constraint=name, season=season, config=config, source_index=int(SOURCES[season]), source_phase=phase,
                start_day=start_day(phase), ndastp=ndastp(start_day(phase)), member=m, rmse=float(errors[m]),
                rmse_mean=float(errors.mean()),
                rmse_members=errors.tolist(), criterion=CRITERION_DATA if config == 'data' else CRITERION,
                model=RUN, model_rel=MODEL_REL, prefix=PREFIX2,
                generation_rel=folder and os.path.relpath(folder, os.environ['OCEANDATA']), generation_args=gen_args,
                data_file=DATA_FILE, source_cache=CACHE1, created=time.strftime('%Y-%m-%d %H:%M:%S')))
