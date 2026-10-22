#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 2 (Jean-Zay) : convert the median member exported by step 1 into a 1/4° NEMO restart, with
`Results/a_posteriori/convert_state.py` (needs the reference runs of its BASE_REFERENCES and xesmf for the regridding).

For every <generation folder>/aposteriori/ holding a selection.json, writes in that folder :
    restart_1deg.nc     generated state in the 1° restart template
    restart_1_4deg.nc   regridded to 1/4°, initial state of the NEMO run
    conversion.json     restart paths, time_counter of the 1/4° restart (nn_it000 - 1 of the run), reference runs
"""
from __init__ import PRP; import sys
sys.path.append(f'{PRP}/Results/a_posteriori/')

import argparse
import os
import time
import xarray as xr
from convert_state import BASE_REFERENCES, REFERENCES, convert
from settings import CONVERSION, INFERENCE, LAUNCH, NEMO_RUN, PREFIX, RESTART_1DEG, RESTART_1_4DEG, aposteriori_dirs, write_json

parser = argparse.ArgumentParser(description='Convert the selected median members to 1/4° NEMO restarts')
parser.add_argument('--inference', default=INFERENCE, help='inference folder of the model, default from $OCEANDATA')
parser.add_argument('--prefix', default=PREFIX, help='sweep folder in the inference folder')
parser.add_argument('--constraints', nargs='*', default=None, help='constraints to convert, default all')
parser.add_argument('--force', action='store_true', help='convert again when conversion.json exists (refused once launched)')
args = parser.parse_args()

dirs = aposteriori_dirs(args.inference, args.prefix, constraints=args.constraints)
if not dirs:
    sys.exit(f'No {args.inference}/{args.prefix}/*/*/aposteriori/selection.json, run step 1 (and sync) first')

for constraint, folder in dirs.items():
    launched = [f for f in (LAUNCH, NEMO_RUN) if os.path.lexists(f'{folder}/{f}')]
    if launched:
        print(f'{constraint} : {", ".join(launched)} in {folder}, the run reads this restart, skip')
        continue
    if os.path.exists(f'{folder}/{CONVERSION}') and not args.force:
        print(f'{constraint} : already converted, skip (--force to redo)')
        continue
    print(f'{constraint} : converting {folder}')
    path_1deg, path_1_4deg = os.path.abspath(f'{folder}/{RESTART_1DEG}'), os.path.abspath(f'{folder}/{RESTART_1_4DEG}')
    convert(folder, path_1deg, path_1_4deg, idx_batch=0)
    with xr.open_dataset(path_1_4deg, decode_times=False) as restart:
        time_counter, kt = float(restart.time_counter.values[0]), float(restart.kt.values.squeeze())
    if kt != time_counter:
        raise ValueError(f'{path_1_4deg} : kt {kt} != time_counter {time_counter}, nn_it000 is ambiguous')
    write_json(f'{folder}/{CONVERSION}', dict(
        restart_1deg=path_1deg, restart_1_4deg=path_1_4deg, time_counter=time_counter,
        base_references=BASE_REFERENCES, references=REFERENCES, created=time.strftime('%Y-%m-%d %H:%M:%S')))
