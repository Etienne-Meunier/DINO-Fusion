"""
Sample one experimental run, seed 0, BorderZero, 1000 steps, paired with the C7 eta sweep and none_border0_b8.
Outputs in runs/<name>/ : generated.npy (b, 37, 208, 64), lbd.npy, trace.npy (steps, 2, m) mean lbd and loss, args.json.

    python run.py --variant profile --eta 30 --eps 0 --name profile_eta30
    python run.py --variant c7_surf --eta 100 1e-5 --eps 8.6e-8 9 --lbd0 1 0 --name c7surf_eta100
    python run.py --variant c7_anchor --eta 100 1e-4 --eps 8.6e-8 1 --lbd0 1 0 --mode raw sqrt --name c7anchor
With an anchor variant and --anchor parallel, the unconstrained sample of the same seed is saved as ref.npy.
"""
from exp import *
import argparse, json, time
from accelerate import Accelerator
from configs.base_config import TrainingConfig
from pipelines.pipeline_tensor import DDPMPipeline_Tensor
from pipelines.State import Observer
from pipelines.losses import BorderZero, ZeroMean

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--variant', choices=['none', 'c7', 'c7_anchor', 'c7_anchor_rho', 'c7skip', 'c7skip_anchor', 'profile_c7skip', 'c7_T', 'proftop_T', 'c7_anchor_prof', 'proftop_anchor', 'profile', 'topfree', 'c7_surf', 'profile_topfree', 'profile_c7', 'c7_zm0', 'zero_mean', 'zero_mean_c7'], required=True)
parser.add_argument('--eta', type=float, nargs='+', required=True)
parser.add_argument('--eps', type=float, nargs='+', required=True)
parser.add_argument('--lbd0', type=float, nargs='+', default=[1.0])
parser.add_argument('--mode', choices=['raw', 'sqrt'], nargs='+', default=['raw'])
parser.add_argument('--t_on', type=float, nargs='+', default=None, help='per constraint, active only for t < t_on')
parser.add_argument('--anchor', choices=['parallel', 'final'], default='parallel',
                    help='reference of the anchor variants : unconstrained trajectory sampled alongside, or the final none run (seed 0)')
parser.add_argument('--centred', action='store_true',
                    help='centred noise and noisy land cells along the trajectory (exp.ddpm_step), BorderZero at the end')
parser.add_argument('--raw_noise', action='store_true',
                    help='with --centred : noisy land cells only, the noise is not centred')
parser.add_argument('--centre_grad', action='store_true',
                    help='guidance step with its volume-weighted spatial mean per channel removed (exp.remove_level_mean)')
parser.add_argument('--dT', type=float, default=0.1)
parser.add_argument('--dS', type=float, default=0.01)
parser.add_argument('--drho', type=float, default=0.1)
parser.add_argument('--K', type=int, default=12, help='levels of c7_anchor_prof')
parser.add_argument('--batch', type=int, default=8)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--steps', type=int, default=1000)
parser.add_argument('--name', type=str, required=True)
args = parser.parse_args()

device = Accelerator().device
pipeline = DDPMPipeline_Tensor.from_pretrained(MODEL).to(device)
pipeline.scheduler.set_timesteps(args.steps)
obs = Observer('data', TrainingConfig(), device=device)
NONE = f'{MODEL}/inference/sweep_baselines/none_border0_b8/151026-145216/generated.npy'
ref = Reference(torch.from_numpy(np.load(NONE)).to(device) if args.anchor == 'final' else None)
ocean = obs.normalizer.padder((~obs.normalizer.masker.mask).float()).to(device)
volume = obs.cell_volume('strided').to(device)
grad_weights = obs.normalizer.padder(torch.cat([volume, volume, volume[:1]])) * ocean
zero_mean = lambda : OceanOnly(ZeroMean(observer=obs), ocean) if args.centred else ZeroMean(observer=obs)
loss_c = {'none' : lambda : None,
          'zero_mean' : zero_mean,
          'zero_mean_c7' : lambda : Stack([zero_mean(), NegativeGradDensityHuber(observer=obs)]),
          'c7' : lambda : NegativeGradDensityHuber(observer=obs),
          'c7_anchor' : lambda : Stack([NegativeGradDensityHuber(observer=obs), SurfaceAnchor(ref, args.dT, args.dS, observer=obs)]),
          'c7_anchor_rho' : lambda : Stack([NegativeGradDensityHuber(observer=obs),
                                            SurfaceAnchorRho(ref, args.drho, observer=obs)]),
          'c7skip' : lambda : HuberSkipTop(observer=obs),
          'c7_T' : lambda : Detach(NegativeGradDensityHuber(observer=obs)),
          'proftop_T' : lambda : Stack([Detach(ProfileHinge(observer=obs)), Detach(HuberTopFree(observer=obs))]),
          'profile_c7skip' : lambda : Stack([ProfileHinge(observer=obs), HuberSkipTop(observer=obs)]),
          'c7skip_anchor' : lambda : Stack([HuberSkipTop(observer=obs), SurfaceAnchor(ref, args.dT, args.dS, observer=obs)]),
          'c7_anchor_prof' : lambda : Stack([NegativeGradDensityHuber(observer=obs),
                                             ProfileAnchor(ref, args.K, dT=args.dT, dS=args.dS, observer=obs)]),
          'proftop_anchor' : lambda : Stack([ProfileHinge(observer=obs), HuberTopFree(observer=obs),
                                             SurfaceAnchor(ref, args.dT, args.dS, observer=obs)]),
          'profile' : lambda : ProfileHinge(observer=obs),
          'topfree' : lambda : HuberTopFree(observer=obs),
          'c7_surf' : lambda : Stack([NegativeGradDensityHuber(observer=obs), SurfaceMean(observer=obs)]),
          'profile_topfree' : lambda : Stack([ProfileHinge(observer=obs), HuberTopFree(observer=obs)]),
          'c7_zm0' : lambda : Stack([NegativeGradDensityHuber(observer=obs), SurfaceZeroMean(observer=obs)]),
          'profile_c7' : lambda : Stack([ProfileHinge(observer=obs), NegativeGradDensityHuber(observer=obs)])}[args.variant]()
one = lambda v : v[0] if len(v) == 1 else v

t0 = time.time()
x, lbd, trace, x_ref = sample(pipeline, args.batch, loss_c=loss_c, eta=one(args.eta), eps=one(args.eps),
                              projection=BorderZero(obs), seed=args.seed,
                              lbd0=torch.tensor(one(args.lbd0), device=device), mode=one(args.mode), t_on=args.t_on,
                              reference=ref if 'anchor' in args.variant else None,
                              ocean=ocean if args.centred else None, grad_weights=grad_weights if args.centre_grad else None,
                              centre_noise=not args.raw_noise)
out = HERE / 'runs' / args.name
out.mkdir(parents=True, exist_ok=True)
np.save(out / 'generated.npy', x.cpu().numpy())
if lbd is not None :
    np.save(out / 'lbd.npy', lbd.cpu().numpy())
    np.save(out / 'trace.npy', trace)
if x_ref is not None :
    np.save(out / 'ref.npy', x_ref.cpu().numpy())
json.dump(dict(vars(args), minutes=(time.time() - t0) / 60), open(out / 'args.json', 'w'), indent=2)
print(f'Saved in {out} ({(time.time() - t0) / 60:.1f} min)')
