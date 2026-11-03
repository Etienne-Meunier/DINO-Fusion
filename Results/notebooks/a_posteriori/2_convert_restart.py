#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 2 (Jean-Zay) : convert the median member exported by step 1 into a 1/4° NEMO restart, with
`Results/a_posteriori/convert_state.py` (needs the reference runs of its BASE_REFERENCES and xesmf for the regridding).

For every <generation folder>/aposteriori/ holding a selection.json, writes in that folder the files of the minimal
restart protocol (PROTOCOLS of settings.py, Results/a_posteriori/restart_initialisation.md) :
    restart_1deg_minimal.nc     generated state in the 1° restart template, input of the regridding
    restart_1_4deg_minimal.nc   regridded to 1/4°, initial state of the NEMO run (tn sn sshn un vn and the calendar only)
    conversion_minimal.json     protocol, selected member (selection.json), restart paths, time_counter of the 1/4°
                                restart (nn_it000 - 1 of the run), its calendar kt, adatrj, ndastp, the start_day of
                                selection.json if any, reference runs
The calendar is that of the 1/4° reference restart, unless selection.json has a start_day (forcing day of year of the
start): then kt, time_counter, adatrj and ndastp follow convert_state.start_calendar.
The files of the template protocol (restart_1deg.nc, restart_1_4deg.nc, conversion.json) are never written.
"""
from __init__ import PRP; import sys
sys.path.append(f'{PRP}/Results/a_posteriori/')

import argparse
import os
import time
import xarray as xr
from convert_state import BASE_REFERENCES, REFERENCES, convert
from settings import EXPERIMENTS, INFERENCE, PREFIX, PROTOCOLS, SELECTION, aposteriori_dirs, experiment_paths, read_json, write_json

parser = argparse.ArgumentParser(description='Convert the selected median members to 1/4° NEMO restarts')
parser.add_argument('--inference', default=INFERENCE, help='inference folder of the model, default from $OCEANDATA')
parser.add_argument('--prefix', default=PREFIX, help='sweep folder in the inference folder')
parser.add_argument('--constraints', nargs='*', default=None, help='constraints to convert, default all')
parser.add_argument('--force', action='store_true', help='convert again when the conversion record exists (refused once launched)')
args = parser.parse_args()

dirs = aposteriori_dirs(args.inference, args.prefix, constraints=args.constraints)
if not dirs:
    sys.exit(f'No {args.inference}/{args.prefix}/*/*/aposteriori/selection.json, run step 1 (and sync) first')

PROTOCOL = 'minimal'
files = PROTOCOLS[PROTOCOL]
runs = [experiment_paths(e)[k] for e, setup in EXPERIMENTS.items() if setup['protocol'] == PROTOCOL
        for k in ('launch', 'nemo_run')]
for constraint, folder in dirs.items():
    launched = [f for f in runs if os.path.lexists(f'{folder}/{f}')]
    if launched:
        print(f'{constraint} : {", ".join(launched)} in {folder}, the run reads this restart, skip')
        continue
    if os.path.exists(f'{folder}/{files["conversion"]}') and not args.force:
        print(f'{constraint} : already converted ({files["conversion"]}), skip (--force to redo)')
        continue
    print(f'{constraint} : converting {folder} ({PROTOCOL} restart)')
    path_1deg = os.path.abspath(f'{folder}/{files["restart_1deg"]}')
    path_1_4deg = os.path.abspath(f'{folder}/{files["restart_1_4deg"]}')
    selection = read_json(f'{folder}/{SELECTION}')
    convert(folder, path_1deg, path_1_4deg, idx_batch=0, start_day=selection.get('start_day'))
    with xr.open_dataset(path_1_4deg, decode_times=False) as restart:
        time_counter, kt = float(restart.time_counter.values[0]), float(restart.kt.values.squeeze())
        adatrj, ndastp = float(restart.adatrj.values.squeeze()), int(restart.ndastp.values.squeeze())
    if kt != time_counter:
        raise ValueError(f'{path_1_4deg} : kt {kt} != time_counter {time_counter}, nn_it000 is ambiguous')
    write_json(f'{folder}/{files["conversion"]}', dict(
        protocol=PROTOCOL, member=selection['member'], selection=selection, restart_1deg=path_1deg,
        restart_1_4deg=path_1_4deg, time_counter=time_counter, kt=kt, adatrj=adatrj, ndastp=ndastp,
        start_day=selection.get('start_day'),
        base_references=BASE_REFERENCES, references=REFERENCES, created=time.strftime('%Y-%m-%d %H:%M:%S')))
