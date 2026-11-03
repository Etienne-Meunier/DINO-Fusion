"""
Copies existing generations into `<model>/inference/october/<name>/<ddmmyy-HHMMSS>/`, the layout `generation_dirs`
reads (prefix='october'). Each run gets its generated.npy and an args.json holding the sampler (noise, land), the
constraint, the batch, the seed and its source. The timestamp is the source file's modification time. No generation
is run : the batch is the largest available for each setting, except the C7 eta series, all batch 8 to compare across eta, and C_NL_b8, its unconstrained batch 8 reference.

Names : C = centred noise (no C : raw noise), B0 = BorderZero at every step, NL = noisy land, zm = zero_mean, sTS = surface T,S mean.
"""
from pathlib import Path
import json, os, shutil, time
import numpy as np

HERE = Path(__file__).resolve().parent
MODEL = Path(os.environ['OCEANDATA']) / 'models/dino-fusion/tav0h83b'
OUT = MODEL / 'inference/october'
RUNS = {'none' : ('optimal/none/151026-221621/generated.npy', 'raw', 'model', 'none'),
        'C' : ('traj_raw_call_x.npy', 'centred', 'model', 'none'),
        'C_B0' : ('traj_border0_call_x.npy', 'centred', 'B0', 'none'),
        'C_NL' : ('centred_none_b32', 'centred', 'NL', 'none'),
        'C_NL_zm0.1' : ('centred_zm_e0.1', 'centred', 'NL', 'zero_mean eta 0.1 eps 0.01'),
        'C_NL_C7030_sTS' : ('centred_surf_c7030_b32', 'centred', 'NL', 'C7 eta 30 + surface T,S mean'),
        'C_NL_b8' : ('centred_none', 'centred', 'NL', 'none'),
        'C_NL_C7010' : ('centred_c7010', 'centred', 'NL', 'C7 eta 10'),
        'C_NL_C7030' : ('centred_c7030', 'centred', 'NL', 'C7 eta 30'),
        'C_NL_C7050' : ('centred_c7050', 'centred', 'NL', 'C7 eta 50'),
        'C_NL_C7100' : ('centred_c7100', 'centred', 'NL', 'C7 eta 100'),
        'NL' : ('nl_none', 'raw', 'NL', 'none'),
        'NL_C7010' : ('nl_c7010', 'raw', 'NL', 'C7 eta 10'),
        'NL_C7030' : ('nl_c7030', 'raw', 'NL', 'C7 eta 30'),
        'NL_C7050' : ('nl_c7050', 'raw', 'NL', 'C7 eta 50'),
        'NL_C7100' : ('nl_c7100', 'raw', 'NL', 'C7 eta 100')}


def source(name) :
    """Path of the generated batch and of its run args (None for traj_test outputs)."""
    if name.startswith('optimal/') :
        f = MODEL / 'inference' / name
        return f, f.parent / 'args.json'
    if name.endswith('.npy') :
        return HERE / 'runs' / name, None
    return HERE / 'runs' / name / 'generated.npy', HERE / 'runs' / name / 'args.json'


for name, (src, noise, land, constraint) in RUNS.items() :
    f, a = source(src)
    if not f.exists() :
        print(f'{name} : {f} missing, skipped')
        continue
    stamp = time.strftime('%d%m%y-%H%M%S', time.localtime(f.stat().st_mtime))
    dest = OUT / name / stamp
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(f, dest / 'generated.npy')
    args = {'noise' : noise, 'land' : land, 'constraint' : constraint, 'batch' : len(np.load(f, mmap_mode='r')),
            'seed' : 0, 'source' : str(f),
            'run_args' : json.loads(a.read_text()) if a else None}
    args['eta'] = float(np.atleast_1d(args['run_args']['eta'])[0]) if args['run_args'] else 0.0
    (dest / 'args.json').write_text(json.dumps(args, indent=2))
    print(name, dest)
