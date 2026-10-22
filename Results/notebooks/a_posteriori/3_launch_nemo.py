#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 3 (Jean-Zay) : create and submit one NEMO DINO 1/4° run of an experiment of EXPERIMENTS (--experiment, default
DEFAULT_EXPERIMENT) per converted generation.

For every <generation folder>/aposteriori/ holding a conversion.json (step 2) :
    1. `nnemo.sh aposteriori/<nemo_run> diffusion` copies the Dinonline run structure (XIOS outputs, climatology link and
       slurm options come from the branch), the run directory is <nemo_run>/restart0 ; the XIOS template of the
       experiment, if any, is copied onto FILE_DEF. If NNEMO_MIRROR exists, git clones it instead of GitHub
    2. RESTART_LINK -> the 1/4° restart in restart0, STALE_LINKS removed
    3. restart.config gets `cn_ocerst_in=RESTART_LINK`, namelist_cfg nn_it000 = time_counter + 1, nn_itend = nn_it000 + nsteps - 1 (nsteps of
       the experiment)
       (one restart, at nn_itend)
    4. checks that the restart, the REQUIRED_LINKS targets and the executables of jobs/mpmd.conf exist, then sbatch
       jobs/launch.sh from restart0 with a per-constraint job name and log and the sbatch options of the experiment;
       job id, namelist and XIOS output groups in aposteriori/<launch>
<nemo_run> and <launch> are named by `experiment_paths` (nemo_run and launch.json for 10y).

Dry run by default (prints the actions), --submit to execute them.
"""
import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from xml.etree import ElementTree
from settings import (CONVERSION, DEFAULT_EXPERIMENT, EXPERIMENTS, FILE_DEF, INFERENCE, JOB_SCRIPT, MPMD, NAMELIST, NEMO_DIR,
                      NNEMO, NNEMO_BRANCH, NNEMO_MIRROR, NNEMO_REPO, PREFIX, REQUIRED_LINKS, RESTART_1_4DEG, RESTART_CONFIG, RESTART_LINK, STALE_LINKS,
                      aposteriori_dirs, experiment_paths, read_json, write_json)


def set_namelist(path, values):
    """Set the uncommented `key = value` entries of the namelist `path` to `values` {key: value}, as launch.sh does with sed.

    Raises if a key is not exactly once in the file.
    """
    with open(path) as f:
        text = f.read()
    for key, value in values.items():
        text, n = re.subn(rf'^( *{re.escape(key)} *= *).*$', rf'\g<1>{value}', text, flags=re.MULTILINE)
        if n != 1:
            raise ValueError(f'{key} found {n} times in {path}')
    with open(path, 'w') as f:
        f.write(text)


def file_groups(path):
    """Output configuration [{id, output_freq, enabled, files}] of the <file_group> tags of the XIOS file_def `path`."""
    return [dict(id=g.get('id'), output_freq=g.get('output_freq'), enabled=g.get('enabled'),
                 files=[f.get('name_suffix') for f in g.iter('file')]) for g in ElementTree.parse(path).iter('file_group')]


def missing_inputs(run_dir):
    """Paths read by the job of `run_dir` that do not exist : targets of RESTART_LINK and REQUIRED_LINKS, and the
    executables of jobs/mpmd.conf (relative to `run_dir`)."""
    paths = [f'{run_dir}/{name}' for name in [RESTART_LINK, *REQUIRED_LINKS]]
    with open(f'{run_dir}/{MPMD}') as f:
        for line in f:
            fields = line.split()
            if fields and not fields[0].startswith('#'):
                paths.append(os.path.join(run_dir, fields[1]))
    return sorted({p for p in paths if not os.path.exists(p)})


def link(target, name):
    """Make `name` a symlink to `target`, replacing an existing link or file."""
    if os.path.lexists(name):
        os.remove(name)
    os.symlink(target, name)


def run(command, cwd=None, env=None):
    """Run `command` in `cwd` with the environment variables `env` added. Returns its stdout, exits with its stdout and
    stderr if it fails."""
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, env=env and {**os.environ, **env})
    if result.returncode != 0:
        sys.exit(f'FAILED ({result.returncode}) : {" ".join(command)}\n{result.stdout}\n{result.stderr}')
    return result.stdout


def mirror_env():
    """git settings, as environment variables, that clone NNEMO_REPO from NNEMO_MIRROR ; None if there is no mirror."""
    if not os.path.isdir(NNEMO_MIRROR):
        return None
    return {'GIT_CONFIG_COUNT': '1', 'GIT_CONFIG_KEY_0': f'url.file://{NNEMO_MIRROR}.insteadOf', 'GIT_CONFIG_VALUE_0': NNEMO_REPO}


def launch(constraint, folder, experiment, nsteps, submit):
    """Create the NEMO run `experiment` of the converted generation in `folder` (aposteriori/) and submit it. Dry run if
    not `submit`."""
    setup, paths = EXPERIMENTS[experiment], experiment_paths(experiment)
    dest = f'{folder}/{paths["nemo_run"]}'
    run_dir = f'{dest}/{NEMO_DIR}'
    conversion = read_json(f'{folder}/{CONVERSION}')
    restart = os.path.abspath(f'{folder}/{RESTART_1_4DEG}')
    nn_it000 = round(float(conversion['time_counter'])) + 1
    namelist = {'nn_it000': nn_it000, 'nn_itend': nn_it000 + nsteps - 1}
    name = f'aposteriori_{constraint}{setup["suffix"]}'
    sbatch = ['sbatch', f'--job-name={name}', f'--output={name}.out', *setup['sbatch'], JOB_SCRIPT]

    print(f'{constraint} ({experiment}) : {run_dir}\n'
          f'  bash {NNEMO} {dest} {NNEMO_BRANCH}' + (f' (from {NNEMO_MIRROR})' if mirror_env() else '') + '\n'
          + (f'  copy {setup["file_def"]} onto {FILE_DEF}\n' if setup['file_def'] else '') +
          f'  link {RESTART_LINK} -> {restart}, remove {STALE_LINKS}\n'
          f'  {RESTART_CONFIG} : cn_ocerst_in={RESTART_LINK}, {NAMELIST} : {namelist}\n'
          f'  cd {run_dir} && {" ".join(sbatch)}')
    if not submit:
        return

    print(run(['bash', NNEMO, dest, NNEMO_BRANCH], env=mirror_env()))
    if setup['file_def']:
        shutil.copyfile(f'{run_dir}/{setup["file_def"]}', f'{run_dir}/{FILE_DEF}')
    link(restart, f'{run_dir}/{RESTART_LINK}')
    for k in STALE_LINKS:
        if os.path.islink(f'{run_dir}/{k}'):
            os.remove(f'{run_dir}/{k}')
    with open(f'{run_dir}/{RESTART_CONFIG}', 'w') as f:
        f.write(f'cn_ocerst_in={RESTART_LINK}\n')
    set_namelist(f'{run_dir}/{NAMELIST}', namelist)
    outputs = file_groups(f'{run_dir}/{FILE_DEF}')

    missing = missing_inputs(run_dir)
    if missing:
        sys.exit(f'{constraint} : not submitted (fix, then remove {dest} to start over), missing inputs of {run_dir} :\n  ' + '\n  '.join(missing))
    # jobs/launch.sh calls `module`, a shell function the job inherits from the submitting shell : submit from a login
    # shell, where it is defined, even when this script runs from a non-interactive ssh command.
    out = run(['bash', '-lc', shlex.join(sbatch)], cwd=run_dir)
    job = re.search(r'Submitted batch job (\d+)', out)
    write_json(f'{folder}/{paths["launch"]}', dict(experiment=experiment, run_dir=run_dir, job_name=name, job_id=job and job.group(1), namelist=namelist, outputs=outputs,
                                          nsteps=nsteps, restart=restart, sbatch=' '.join(sbatch),
                                          sbatch_output=out.strip(), created=time.strftime('%Y-%m-%d %H:%M:%S')))


parser = argparse.ArgumentParser(description='Create and submit the NEMO runs of the converted generations')
parser.add_argument('--inference', default=INFERENCE, help='inference folder of the model, default from $OCEANDATA')
parser.add_argument('--prefix', default=PREFIX, help='sweep folder in the inference folder')
parser.add_argument('--constraints', nargs='*', default=None, help='constraints to launch, default all')
parser.add_argument('--experiment', default=DEFAULT_EXPERIMENT, choices=list(EXPERIMENTS), help='experiment to run')
parser.add_argument('--nsteps', type=int, default=None, help='number of time steps of the run, default that of the experiment')
parser.add_argument('--submit', action='store_true', help='execute, dry run otherwise')
args = parser.parse_args()

dirs = aposteriori_dirs(args.inference, args.prefix, required=CONVERSION, constraints=args.constraints)
if not dirs:
    sys.exit(f'No {args.inference}/{args.prefix}/*/*/aposteriori/{CONVERSION}, run step 2 first')

paths = experiment_paths(args.experiment)
nsteps = args.nsteps or EXPERIMENTS[args.experiment]['nsteps']
for constraint, folder in dirs.items():
    if os.path.exists(f'{folder}/{paths["launch"]}'):
        print(f'{constraint} : already launched ({folder}/{paths["launch"]}), skip')
    elif os.path.lexists(f'{folder}/{paths["nemo_run"]}'):
        print(f'{constraint} : {folder}/{paths["nemo_run"]} exists without {paths["launch"]}, check it by hand (remove it to start over), skip')
    else:
        launch(constraint, folder, args.experiment, nsteps, args.submit)
