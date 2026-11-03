"""
Task 3 : order-of-magnitude check of the accumulation mechanism. The surface warming of the final state is predicted as
    dT0 = 3 sigma_T0 * sum_t lbd_t * DC_T0(t) * c_t
with DC_T0(t) the DPS step on channel T level 0 at lbd = 1 (p3, NONE, V0 base, interpolated in t), lbd_t growing linearly
from lbd0 = 1 to the final lbd of the run, and c_t = 1 (push kept as is in x_t) or 1 / sqrt(abar_t) (push kept in the
Tweedie estimate, DC-blind denoiser). Compared with the paired surface warming measured by agent A.
Output : p4_accumulation.txt
"""
from common import *
import pandas as pd
from diffusers import DDPMScheduler

p3 = pd.read_csv(HERE / 'p3_remedies.csv')
b = p3[(p3.set == 'NONE') & (p3.variant == 'V0 base')].sort_values('t')
sch = DDPMScheduler.from_pretrained(MODEL, subfolder='scheduler')
t = np.arange(1000)
dc = np.interp(t, b.t.values, b.T0.values)
abar = sch.alphas_cumprod.numpy()
obs = get_observer()
std = obs.normalizer.normaliser.normalization.std
s3 = 3 * float(std.flatten()[0])
# agent A : eta -> (final mean lbd, paired dT level 0 in C)
runs = {10 : (98.7, 1.603), 30 : (188.9, 3.054), 50 : (262.5, 4.183), 80 : (356.7, 5.647)}
out = [f'3 sigma_T level 0 = {s3:.2f} C']
for eta, (lf, dT) in runs.items() :
    lbd = np.linspace(lf, 1.0, 1000)            # t = 999 first : lbd_t grows as t decreases
    p1 = s3 * (lbd * dc).sum()
    p2 = s3 * (lbd * dc / np.sqrt(abar)).sum()
    out.append(f'eta {eta:4d} final lbd {lf:6.1f} : predicted dT0 {p1:5.2f} C (c=1), {p2:6.2f} C (c=1/sqrt(abar)), observed {dT:5.2f} C')
open(HERE / 'p4_accumulation.txt', 'w').write('\n'.join(out) + '\n'); print('\n'.join(out))
