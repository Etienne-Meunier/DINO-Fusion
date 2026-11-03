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

# Restart protocols of Results/a_posteriori/restart_initialisation.md: their files in aposteriori/ and the namelist_cfg
# values their runs need. Step 2 writes only 'minimal'; the 'template' files (spun-up 1/4° restart with the generated
# fields replaced) are those of the 10y and daily_2m runs. Both use the calendar of the 1/4° reference restart (date
# 0249/12/01 at the first step) unless selection.json has a start_day (minimal only). The DINO forcing phase follows kt
# only, (kt / 120) mod 360 days, so the reference kt = 10800000 starts the forcing on day 0 (1 January).
PROTOCOLS = {
    'template': dict(restart_1deg=RESTART_1DEG, restart_1_4deg=RESTART_1_4DEG, conversion=CONVERSION, namelist={}),
    # NEMO reads tb, ub, sshb and the *_b forcing fields unless ln_1st_euler. Without sshb, NEMO 4.2 forces it with a
    # warning (domain.F90): set explicitly so the run does not rely on that fallback.
    'minimal': dict(restart_1deg='restart_1deg_minimal.nc', restart_1_4deg='restart_1_4deg_minimal.nc',
                    conversion='conversion_minimal.json', namelist={'ln_1st_euler': '.true.'}),
}

# NEMO experiments. Each has its restart protocol, its own run directory, launch record and results folder in
# aposteriori/ (`experiment_paths`), its length, the XIOS output template of Dinonline restart0 copied onto FILE_DEF
# (None keeps FILE_DEF) and sbatch options added to those of jobs/launch.sh.
EXPERIMENTS = {
    '10y': dict(protocol='template', suffix='', nsteps=NSTEPS, file_def=None, sbatch=[]),
    # 2 months (60 days of 120 steps), daily means of the 3D fields, ~12 GB of outputs per run
    'daily_2m': dict(protocol='template', suffix='_daily_2m', nsteps=7200, file_def='file_def_nemo-oce_daily.xml',
                     sbatch=['--qos=qos_cpu-dev', '--time=01:00:00']),
    '10y_minimal': dict(protocol='minimal', suffix='_minimal', nsteps=NSTEPS, file_def=None, sbatch=[]),
    'daily_2m_minimal': dict(protocol='minimal', suffix='_daily_2m_minimal', nsteps=7200,
                             file_def='file_def_nemo-oce_daily.xml', sbatch=['--qos=qos_cpu-dev', '--time=01:00:00']),
    # 2 weeks (14 days of 120 steps); a 2-month daily run took 3.5 min
    'daily_2w_minimal': dict(protocol='minimal', suffix='_daily_2w_minimal', nsteps=1680,
                             file_def='file_def_nemo-oce_daily.xml', sbatch=['--qos=qos_cpu-dev', '--time=00:30:00']),
    # 2 years (720 days of 120 steps), 7-day means of the 3D fields and daily means of the 2D fields, ~25 GB per run
    'weekly_2y_minimal': dict(protocol='minimal', suffix='_weekly_2y_minimal', nsteps=86400,
                              file_def='file_def_nemo-oce_weekly.xml', sbatch=['--qos=qos_cpu-dev', '--time=02:00:00']),
}
DEFAULT_EXPERIMENT = '10y_minimal'

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
    """Names, inside aposteriori/, of the run directory, launch record and results folder of `experiment` (EXPERIMENTS key),
    and of the 1/4° restart and conversion record of its restart protocol."""
    setup = EXPERIMENTS[experiment]
    protocol = PROTOCOLS[setup['protocol']]
    return dict(nemo_run=f'{NEMO_RUN}{setup["suffix"]}', launch=f'{os.path.splitext(LAUNCH)[0]}{setup["suffix"]}.json',
                results=f'{RESULTS}{setup["suffix"]}', restart_1_4deg=protocol['restart_1_4deg'],
                conversion=protocol['conversion'])


def read_json(path):
    """Content of the json file `path`."""
    with open(path) as f:
        return json.load(f)


def write_json(path, content):
    """Write `content` in the json file `path`, indented."""
    with open(path, 'w') as f:
        json.dump(content, f, indent=2, default=str)
    print(f'Saved {path}')
