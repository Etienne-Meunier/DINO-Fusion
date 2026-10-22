#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 4 (Jean-Zay) : check the NEMO runs of step 3, merge their final restart and gather the results.

For every <generation folder>/aposteriori/ holding the launch record of the experiment (step 3, --experiment, default
DEFAULT_EXPERIMENT), in its run directory <nemo_run>/restart0 :
    1. the run is finished if its job is no longer in squeue, time.step reached nn_itend, ocean.output has NEMO_END and
       no 'E R R O R', and the slurm log has the `real` line of `time srun` (elapsed time); otherwise nothing is touched
    2. the split final restart <EXP>_<nn_itend>_restart_XXXX.nc (as many pieces as its DOMAIN_number_total) is merged
       into <EXP>_<nn_itend>_restart.nc by restart0/rebuild_nemo.exe, as a qos_cpu-dev job by default (run this script
       again once it is done; rebuild_<constraint>.json holds its job id) or on the login node with --login. Its output
       goes to rebuild_<constraint>.out, which must end with REBUILD_DONE: this is the proof the merge is complete,
       a failed merge is reported and not retried until rebuild_<constraint>.{json,out} are removed
    3. the merged restart is checked with ncdump (RESTART_DIMS)
    4. the merged restart and the XIOS outputs of the launch record are moved to aposteriori/<results>/, the slurm,
       rebuild logs and ocean.output copied there, with <results>/collect.json (status, elapsed, rebuild proof, file sizes)
<nemo_run>, the launch record and <results> are named by `experiment_paths` (nemo_run, launch.json, results for 10y).
The restart pieces are kept, unless --clean-pieces (removed once the merge is proven and the merged restart checked).

Dry run by default (prints the actions), --run to execute them. Only the python standard library is needed.
"""
import argparse
import glob
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from settings import (COLLECT, DEFAULT_EXPERIMENT, EXP, EXPERIMENTS, INFERENCE, MODULES, NEMO_DIR, NEMO_END, PREFIX, REBUILD,
                      REBUILD_DONE, REBUILD_SBATCH, REBUILD_THREADS, RESTART_DIMS, aposteriori_dirs, experiment_paths,
                      read_json, write_json)


def output_files(outputs):
    """Names of the files written by the enabled XIOS groups `outputs` of launch.json (file_def name `@expname@_@freq@`)."""
    return [f'{EXP}_{g["output_freq"].replace("mo", "m")}{suffix}.nc' for g in outputs
            if g['enabled'] == '.TRUE.' for suffix in g['files']]


def read(path):
    """Content of the text file `path`, '' if it does not exist."""
    return open(path, errors='replace').read() if os.path.exists(path) else ''


def in_queue(job_id):
    """True if the slurm job `job_id` is pending or running, False if not, None if squeue is unavailable or failed."""
    try:
        result = subprocess.run(['squeue', '-h', '-j', str(job_id)], capture_output=True, text=True)
    except FileNotFoundError:
        return None
    # squeue fails with 'Invalid job id' once a finished job has left the queue.
    if result.returncode != 0 and 'Invalid job id' not in result.stderr:
        return None
    return bool(result.stdout.strip())


def run_status(run_dir, log, job_id):
    """State of the run of `run_dir` : queue, last time step, NEMO_END closing ocean.output, its 'E R R O R' count, elapsed time."""
    step = read(f'{run_dir}/time.step').strip()
    ocean = read(f'{run_dir}/ocean.output')
    elapsed = re.findall(r'^real\s+(\S+)', read(f'{run_dir}/{log}'), flags=re.MULTILINE)
    # NEMO also prints NEMO_END as a section separator during the run: only a last line equal to it marks the end.
    lines = ocean.split()
    return dict(in_queue=in_queue(job_id), last_step=int(step) if step.isdigit() else None,
                ended=bool(lines) and lines[-1] == NEMO_END,
                errors=ocean.count('E R R O R'), elapsed=elapsed[-1] if elapsed else None)


def shell(command, cwd):
    """Run the bash `command` in `cwd` in a login shell with MODULES loaded. Returns (returncode, stdout + stderr)."""
    result = subprocess.run(['bash', '-lc', f'{MODULES}; {command}'], cwd=cwd, capture_output=True, text=True)
    return result.returncode, result.stdout + result.stderr


def header(path, cwd):
    """`ncdump -h` of the netcdf file `path`, None if it cannot be read."""
    code, out = shell(f'ncdump -h {shlex.quote(path)}', cwd)
    if code != 0:
        print(f'  ncdump -h {path} failed :\n{out}')
        return None
    return out


def dims(text):
    """Dimensions {name: size} of an `ncdump -h` output."""
    return {k: int(v) for k, v in re.findall(r'^\s*(\w+) = (?:UNLIMITED ; // \()?(\d+)', text.split('variables:')[0],
                                             flags=re.MULTILINE)}


def rebuild(constraint, run_dir, base, pieces, args):
    """Merge `pieces` into `<base>.nc` in `run_dir`, as a qos_cpu-dev job unless `args.login`. Dry run if not `args.run`.

    Returns the proof {mode, log, job_id} of a completed merge, None while the merge is not proven.
    """
    log, marker = f'{run_dir}/rebuild_{constraint}.out', f'{run_dir}/rebuild_{constraint}.json'
    job = read_json(marker) if os.path.exists(marker) else {}
    if REBUILD_DONE in read(log):
        return dict(mode=job.get('mode', 'login'), log=log, job_id=job.get('job_id'))
    if job and job.get('job_id') and in_queue(job['job_id']) is not False:
        print(f'  rebuild job {job["job_id"]} pending or running (or squeue unavailable), skip')
        return None
    if job or os.path.exists(log):
        print(f'  rebuild failed, no "{REBUILD_DONE}" in {log} :\n{read(log)[-2000:]}'
              f'  remove {marker} and {log} to retry, skip')
        return None

    if not os.path.exists(f'{run_dir}/{REBUILD}'):
        print(f'  no {REBUILD} in {run_dir}, skip')
        return None
    if not pieces:
        print(f'  no {base}_XXXX.nc to merge, skip')
        return None
    text = header(pieces[0], run_dir)
    total = re.search(r':DOMAIN_number_total = (\d+)', text or '')
    if total is None or int(total.group(1)) != len(pieces):
        print(f'  {len(pieces)} pieces but DOMAIN_number_total {total and total.group(1)} in {pieces[0]}, skip')
        return None

    command = (f'export OMP_NUM_THREADS={REBUILD_THREADS}; '
               f'printf "&nam_rebuild\\nfilebase=\\"{base}\\"\\nndomain={len(pieces)}\\n/\\n" > nam_rebuild && ./{REBUILD}')
    print(f'  merge {len(pieces)} pieces -> {base}.nc {"on this node" if args.login else "with a qos_cpu-dev job"} : '
          f'{command}')
    if not args.run:
        return None
    if os.path.exists(f'{run_dir}/{base}.nc'):
        print(f'  remove {base}.nc, merged without proof of success')
        os.remove(f'{run_dir}/{base}.nc')
    if args.login:
        code, out = shell(command, run_dir)
        with open(log, 'w') as f:
            f.write(out)
        if code != 0 or REBUILD_DONE not in out:
            if os.path.exists(f'{run_dir}/{base}.nc'):
                os.remove(f'{run_dir}/{base}.nc')
            print(f'  rebuild failed ({code}), see {log}, skip')
            return None
        write_json(marker, dict(mode='login', job_id=None, created=time.strftime('%Y-%m-%d %H:%M:%S')))
        return dict(mode='login', log=log, job_id=None)
    sbatch = ['sbatch', f'--job-name=rebuild_{constraint}', f'--output={os.path.basename(log)}', *REBUILD_SBATCH,
              f'--wrap=bash -lc {shlex.quote(f"{MODULES}; {command}")}']
    result = subprocess.run(sbatch, cwd=run_dir, capture_output=True, text=True)
    job_id = re.search(r'Submitted batch job (\d+)', result.stdout)
    if result.returncode != 0 or job_id is None:
        print(f'  sbatch failed ({result.returncode}) :\n{result.stdout}{result.stderr}')
        return None
    write_json(marker, dict(mode='sbatch', job_id=job_id.group(1), created=time.strftime('%Y-%m-%d %H:%M:%S')))
    print(f'  rebuild job {job_id.group(1)} submitted, run this script again once it is done')
    return None


def collect(constraint, folder, paths, args):
    """Check, merge and gather the run of `folder` (aposteriori/) named by `paths` (`experiment_paths`). Dry run if not
    `args.run`."""
    run_dir, results = f'{folder}/{paths["nemo_run"]}/{NEMO_DIR}', f'{folder}/{paths["results"]}'
    launch = read_json(f'{folder}/{paths["launch"]}')
    itend = launch['namelist']['nn_itend']
    log = f'{launch["job_name"]}.out'
    status = run_status(run_dir, log, launch['job_id'])
    print(f'{constraint} : {run_dir}\n  job {launch["job_id"]} in queue {status["in_queue"]}, time step '
          f'{status["last_step"]} / {itend}, ended {status["ended"]}, {status["errors"]} errors, elapsed {status["elapsed"]}')
    if (status['in_queue'] is not False or status['last_step'] != itend or not status['ended'] or status['errors']
            or status['elapsed'] is None):
        print('  not finished or failed, skip')
        return

    base = f'{EXP}_{itend:08d}_restart'
    merged = f'{base}.nc'
    pieces = sorted(glob.glob(f'{run_dir}/{base}_[0-9][0-9][0-9][0-9].nc'))
    proof = rebuild(constraint, run_dir, base, pieces, args)
    if proof is None:
        return

    outputs = output_files(launch['outputs'])
    moved = [merged, *outputs]
    missing = [n for n in moved if not os.path.exists(f'{run_dir}/{n}') and not os.path.exists(f'{results}/{n}')]
    if missing:
        print(f'  missing {missing}, skip')
        return
    copied = [log, 'ocean.output', os.path.basename(proof['log'])]
    print(f'  merge proven ({proof["mode"]}), move {moved} and copy {copied} to {results}'
          + (f', remove {len(pieces)} pieces' if args.clean_pieces else ''))
    if not args.run:
        return

    text = header(f'{run_dir}/{merged}' if os.path.exists(f'{run_dir}/{merged}') else f'{results}/{merged}', run_dir)
    found = dims(text) if text else None
    if found is None or any(found.get(k) != v for k, v in RESTART_DIMS.items()):
        print(f'  {merged} has dimensions {found}, expected {RESTART_DIMS}, skip')
        return
    os.makedirs(results, exist_ok=True)
    for name in moved:
        if os.path.exists(f'{run_dir}/{name}'):
            shutil.move(f'{run_dir}/{name}', f'{results}/{name}')
    for name in copied:
        if os.path.exists(f'{run_dir}/{name}'):
            shutil.copy2(f'{run_dir}/{name}', f'{results}/{name}')
    if args.clean_pieces:
        for piece in pieces:
            os.remove(piece)
    write_json(f'{results}/{COLLECT}', dict(
        status='complete', job_id=launch['job_id'], nn_itend=itend, ndomain=len(pieces) or None, **status,
        rebuild=dict(**proof, proof=REBUILD_DONE), restart_dims=found, pieces_removed=args.clean_pieces,
        run_dir=run_dir, files={n: os.path.getsize(f'{results}/{n}') for n in sorted(os.listdir(results)) if n != COLLECT},
        created=time.strftime('%Y-%m-%d %H:%M:%S')))


parser = argparse.ArgumentParser(description='Check the NEMO runs, merge their final restart and gather the results')
parser.add_argument('--inference', default=INFERENCE, help='inference folder of the model, default from $OCEANDATA')
parser.add_argument('--prefix', default=PREFIX, help='sweep folder in the inference folder')
parser.add_argument('--constraints', nargs='*', default=None, help='constraints to collect, default all')
parser.add_argument('--experiment', default=DEFAULT_EXPERIMENT, choices=list(EXPERIMENTS), help='experiment to collect')
parser.add_argument('--login', action='store_true', help='merge the restart on this node instead of a qos_cpu-dev job')
parser.add_argument('--clean-pieces', action='store_true', help='remove the restart pieces once the merge is proven')
parser.add_argument('--run', action='store_true', help='execute, dry run otherwise')
args = parser.parse_args()

paths = experiment_paths(args.experiment)
dirs = aposteriori_dirs(args.inference, args.prefix, required=paths['launch'], constraints=args.constraints)
if not dirs:
    sys.exit(f'No {args.inference}/{args.prefix}/*/*/aposteriori/{paths["launch"]}, run step 3 first')

for constraint, folder in dirs.items():
    if os.path.exists(f'{folder}/{paths["results"]}/{COLLECT}'):
        print(f'{constraint} : already collected ({folder}/{paths["results"]}/{COLLECT}), skip')
    else:
        collect(constraint, folder, paths, args)
