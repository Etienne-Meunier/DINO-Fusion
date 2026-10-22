#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paths and cluster constants shared by the a posteriori steps, and the helpers to find their `aposteriori/` folders.

Every step works under ``<INFERENCE>/<PREFIX>/<constraint>/<timestamp>/aposteriori/``. INFERENCE is built from
$OCEANDATA, so the same scripts run locally or on Jean-Zay, where the generation tree is mirrored under JZ_OCEANDATA.
RUN / PREFIX follow the PAPER_RUN / PAPER_PREFIX environment variables, as `paper/common.py`.
"""
import json
import os
from datetime import datetime
from glob import glob

RUN = os.environ.get('PAPER_RUN', 'tav0h83b')
PREFIX = os.environ.get('PAPER_PREFIX', 'optimal')
MODEL_REL = f'models/dino-fusion/{RUN}'
INFERENCE = f'{os.environ["OCEANDATA"]}/{MODEL_REL}/inference'

APOSTERIORI = 'aposteriori'
DATA_RUN = 'data'                          # folder of the training states of step 0, next to the constraint folders
DATA_STATES = 'states.npy'                 # its states, not named generated.npy so the generation loaders skip them
DATA_BATCH = 16                            # as the batch of a generation
SELECTION = 'selection.json'
CONVERSION = 'conversion.json'
LAUNCH = 'launch.json'
COLLECT = 'collect.json'
RESTART_1DEG = 'restart_1deg.nc'
RESTART_1_4DEG = 'restart_1_4deg.nc'
NEMO_RUN = 'nemo_run'
RESULTS = 'results'

# ---------------------------------------------------------------------------------------------------------------------
# Jean-Zay constants. Everything of the project lives under JZ_ROOT; JZ_OCEANDATA mirrors the local $OCEANDATA layout
# (set OCEANDATA to it on JZ).
# ---------------------------------------------------------------------------------------------------------------------
JZ_HOST = 'jzjulie'                        # ssh alias used by rsync
JZ_ROOT = '/lustre/fswork/projects/rech/omr/romr004/etienne/'
JZ_OCEANDATA = f'{JZ_ROOT}oceandata/'
# Reference runs of `Results/a_posteriori/convert_state.py` (BASE_REFERENCES), relative to $OCEANDATA.
REFERENCES_REL = 'Dino-Fusion-Data/Restart-format'

# `nnemo.sh <dest> <branch>` (https://github.com/Etienne-Meunier/nnemo) runs `rm -rf <dest>`, shallow-clones the branch
# of Dinonline into <dest> and runs <dest>/restart0/scripts/setup_links.sh. The diffusion branch (from 944ff64) sets the
# XIOS outputs, the climatology link and the slurm options (job name, time, mails) of the a posteriori runs itself.
NNEMO = f'{JZ_ROOT}nnemo.sh'
NNEMO_BRANCH = 'diffusion'
NNEMO_REPO = 'https://github.com/Etienne-Meunier/Dinonline.git'  # REPO_URL of nnemo.sh
# Bare clone of NNEMO_REPO on JZ, cloned from instead of GitHub when it exists (GitHub is not always reachable from the
# login nodes). Refresh it from a machine that reaches GitHub, see README.
NNEMO_MIRROR = f'{JZ_ROOT}code/Dinonline.git'
NEMO_DIR = 'restart0'                      # run directory inside <dest>, sbatch'ed from there

RESTART_CONFIG = 'restart.config'          # single line `cn_ocerst_in=<restart>`, copied into namelist_cfg by the job
NAMELIST = 'namelist_cfg'
JOB_SCRIPT = 'jobs/launch.sh'
MPMD = 'jobs/mpmd.conf'                    # `<ranks> <executable>` lines run by srun --multi-prog
FILE_DEF = 'file_def_nemo-oce.xml'         # XIOS outputs, recorded in launch.json
RESTART_LINK = 'restart_1_4deg.nc'         # link to the 1/4° restart in NEMO_DIR, read with cn_ocerst_indir = '.'
REQUIRED_LINKS = ['emp_climatology_1_4deg.nc']  # links of setup_links.sh read by the run, checked before sbatch
STALE_LINKS = ['diffusion_states']         # committed link to a purged /lustre/fsn1 folder, unused

NSTEPS = 432000                            # 10 years at rn_Dt = 720 s (~170 min with the layout of jobs/launch.sh)

# NEMO experiments run from the same restart. Each has its own run directory, launch record and results folder in
# aposteriori/ (`experiment_paths`), its length, the XIOS output template of Dinonline restart0 copied onto FILE_DEF
# (None keeps FILE_DEF) and sbatch options added to those of jobs/launch.sh.
EXPERIMENTS = {
    '10y': dict(suffix='', nsteps=NSTEPS, file_def=None, sbatch=[]),
    # 2 months (60 days of 120 steps), daily means of the 3D fields, ~12 GB of outputs per run
    'daily_2m': dict(suffix='_daily_2m', nsteps=7200, file_def='file_def_nemo-oce_daily.xml',
                     sbatch=['--qos=qos_cpu-dev', '--time=01:00:00']),
}
DEFAULT_EXPERIMENT = '10y'

EXP = 'DINO'                               # cn_exp of namelist_cfg, prefix of the NEMO files
REBUILD = 'rebuild_nemo.exe'               # in NEMO_DIR, reads its arguments from the namelist file `nam_rebuild` of the cwd
# Environment of jobs/launch.sh, needed by rebuild_nemo.exe and ncdump.
MODULES = ('module purge; module load intel-compilers/19.0.4 intel-mpi/19.0.4 hdf5/1.10.5-mpi netcdf/4.7.2-mpi '
           'netcdf-fortran/4.5.2-mpi')
REBUILD_THREADS = 4                        # OpenMP threads of rebuild_nemo.exe
REBUILD_SBATCH = ['--account=omr@cpu', '--qos=qos_cpu-dev', '--ntasks=1', f'--cpus-per-task={REBUILD_THREADS}',
                  '--hint=nomultithread', '--time=00:30:00']
REBUILD_DONE = 'NEMO rebuild completed successfully'  # last line printed by a successful rebuild_nemo.exe
NEMO_END = 'AAAAAAAA'                      # last line of ocean.output of a NEMO run that ended normally
RESTART_DIMS = {'y': 797, 'x': 242, 'nav_lev': 36}


def timestamp(path):
    """Date of a generation folder `.../<ddmmyy-HHMMSS>`, as named by `Diffusion_Model/generate.py`."""
    return datetime.strptime(os.path.basename(os.path.normpath(path)), '%d%m%y-%H%M%S')


def aposteriori_dirs(inference=INFERENCE, prefix=PREFIX, required=SELECTION, constraints=None):
    """``<inference>/<prefix>/<constraint>/<timestamp>/aposteriori/`` folders holding the file `required`, keyed by constraint.

    `constraints` restricts the result to these names. A constraint with several timestamps keeps the latest.
    """
    found = {}
    for path in glob(f'{inference}/{prefix}/*/*/{APOSTERIORI}/{required}'):
        generation = os.path.dirname(os.path.dirname(path))
        constraint = os.path.basename(os.path.dirname(generation))
        if constraints is None or constraint in constraints:
            found.setdefault(constraint, []).append(generation)
    return {c: f'{max(g, key=timestamp)}/{APOSTERIORI}' for c, g in sorted(found.items())}


def experiment_paths(experiment):
    """Names, inside aposteriori/, of the run directory, launch record and results folder of `experiment` (EXPERIMENTS key)."""
    suffix = EXPERIMENTS[experiment]['suffix']
    return dict(nemo_run=f'{NEMO_RUN}{suffix}', launch=f'{os.path.splitext(LAUNCH)[0]}{suffix}.json', results=f'{RESULTS}{suffix}')


def read_json(path):
    """Content of the json file `path`."""
    with open(path) as f:
        return json.load(f)


def write_json(path, content):
    """Write `content` in the json file `path`, indented."""
    with open(path, 'w') as f:
        json.dump(content, f, indent=2, default=str)
    print(f'Saved {path}')
