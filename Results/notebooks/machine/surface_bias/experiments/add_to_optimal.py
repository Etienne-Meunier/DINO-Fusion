"""
Copies the repo-sampler runs of C7 with the surface T,S mean constraint (raw noise, BorderZero at every step, b16, seed 0,
`run.py --variant c7_surf`) into `<model>/inference/optimal/<name>/<ddmmyy-HHMMSS>/`, with an args.json in the format of
the optimal runs plus the source and the run args. The timestamp is the source file's modification time.
"""
from pathlib import Path
import json, os, shutil, time

HERE = Path(__file__).resolve().parent
MODEL = Path(os.environ['OCEANDATA']) / 'models/dino-fusion/tav0h83b'
RUNS = {'negative_grad_density_huber_surface_border0' : 'b16_surf_c7010',
        'negative_grad_density_huber_surface_border0_30' : 'b16_surf_c7030'}

for name, src in RUNS.items() :
    f, run_args = HERE / 'runs' / src / 'generated.npy', json.loads((HERE / 'runs' / src / 'args.json').read_text())
    dest = MODEL / 'inference/optimal' / name / time.strftime('%d%m%y-%H%M%S', time.localtime(f.stat().st_mtime))
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(f, dest / 'generated.npy')
    args = {'loss' : 'negative_grad_density_huber+surface_mean', 'eps' : run_args['eps'], 'batch' : run_args['batch'],
            'eta' : run_args['eta'], 'mode' : run_args['mode'], 'projection' : 'border_zero', 'seed' : run_args['seed'],
            'lbd0' : run_args['lbd0'], 'inf_steps' : run_args['steps'], 'prefix' : 'optimal',
            'constraint' : 'C7 + surface T,S mean', 'source' : str(f), 'run_args' : run_args}
    (dest / 'args.json').write_text(json.dumps(args, indent=2))
    print(name, dest)
