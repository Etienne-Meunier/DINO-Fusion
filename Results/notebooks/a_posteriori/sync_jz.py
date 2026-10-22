#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rsync the aposteriori/ folders of the sweep between the local $OCEANDATA and its Jean-Zay mirror JZ_OCEANDATA.

    push : local -> JZ, the outputs of step 1 ({toce,soce,ssh}.npy, selection.json), --all to add those of a local step 2
           (.nc restarts, conversion.json), --references to also send the reference runs REFERENCES_REL used by step 2
    pull : JZ -> local, conversion.json and the launch records of steps 2 and 3 (--all to also pull the .nc restarts),
           --results to also pull the results folder of step 4 of the experiment --experiment (merged restart, XIOS
           outputs, logs, collect.json)
Only these files of the `<constraint>/<timestamp>/aposteriori/` sub-trees are transferred (never nemo_run/ nor macOS `._*`),
and --update keeps the files that are newer on the receiving side.

Prints the rsync commands only by default, --dry-run to run them with rsync --dry-run (connects to JZ), --run to transfer.
"""
import argparse
import os
import shlex
import subprocess
from settings import (APOSTERIORI, CONVERSION, DEFAULT_EXPERIMENT, EXPERIMENTS, JZ_HOST, JZ_OCEANDATA, LAUNCH, MODEL_REL,
                      PREFIX, REFERENCES_REL, SELECTION, experiment_paths)

parser = argparse.ArgumentParser(description='Rsync the aposteriori folders with Jean-Zay')
parser.add_argument('direction', choices=['push', 'pull'])
parser.add_argument('--prefix', default=PREFIX, help='sweep folder in the inference folder')
parser.add_argument('--constraints', nargs='*', default=['*'], help='constraints to transfer, default all')
parser.add_argument('--all', action='store_true', help='also the .nc restarts (push: and conversion.json, for a local step 2)')
parser.add_argument('--references', action='store_true', help='push: also the reference runs of step 2')
parser.add_argument('--results', action='store_true', help='pull: also the results folder of step 4 (~25 GB per 10y run)')
parser.add_argument('--experiment', default=DEFAULT_EXPERIMENT, choices=list(EXPERIMENTS), help='pull --results: experiment')
mode = parser.add_mutually_exclusive_group()
mode.add_argument('--dry-run', action='store_true', help='run rsync --dry-run, connects to JZ')
mode.add_argument('--run', action='store_true', help='transfer')
args = parser.parse_args()

local_root, remote_root = os.environ['OCEANDATA'].rstrip('/'), f'{JZ_HOST}:{JZ_OCEANDATA.rstrip("/")}'
rsync = ['rsync', '-av', '--update', '--prune-empty-dirs', *(['--dry-run'] if args.dry_run else [])]


def parents(rel):
    """rsync filters including the parent directories of `rel` (relative to the transfer root) and `rel` itself."""
    parts = rel.strip('/').split('/')
    return [f'--include=/{"/".join(parts[:i])}/' for i in range(1, len(parts) + 1)]


if args.direction == 'push':
    files = ['*.npy', SELECTION] + (['*.nc', CONVERSION] if args.all else [])
else:
    files = [CONVERSION, f'{os.path.splitext(LAUNCH)[0]}*.json'] + (['*.nc'] if args.all else [])
# Transfers are root to root with anchored filters, so the receiving side needs no pre-existing parent directory.
sweep = f'{MODEL_REL}/inference/{args.prefix}'
filters = ['--exclude=._*', *parents(sweep)]
for c in args.constraints:
    filters += [f'--include=/{sweep}/{c}/', f'--include=/{sweep}/{c}/*/', f'--include=/{sweep}/{c}/*/{APOSTERIORI}/',
                *[f'--include=/{sweep}/{c}/*/{APOSTERIORI}/{f}' for f in files]]
    if args.direction == 'pull' and args.results:
        results = experiment_paths(args.experiment)['results']
        filters += [f'--include=/{sweep}/{c}/*/{APOSTERIORI}/{results}/',
                    *[f'--include=/{sweep}/{c}/*/{APOSTERIORI}/{results}/{f}' for f in ['*.nc', '*.json', '*.out', 'ocean.output']]]
filters.append('--exclude=*')
if args.direction == 'push' and args.references:
    filters[-1:] = [*parents(REFERENCES_REL), f'--include=/{REFERENCES_REL}/***', '--exclude=*']
local, remote = f'{local_root}/', f'{remote_root}/'
commands = [rsync + filters + ([local, remote] if args.direction == 'push' else [remote, local])]

for command in commands:
    print(shlex.join(command))
    if args.run or args.dry_run:
        subprocess.run(command, check=True)
