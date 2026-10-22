"""Level-mean T of the training-time eval samples epoch_*.npy (8 states each, generated during training on the cluster)."""
import numpy as np, sys
sys.argv = ['x']
exec(open('stats_my.py').read().split('pools = {}')[0])
M = '/Volumes/LoCe/oceandata/models/dino-fusion/tav0h83b'
allt = []
for e in list(range(3000, 5000, 50)):
    t = lm(np.load(f'{M}/epoch_{e}.npy')); allt.append(t)
    if e % 250 == 0 or e == 4950: row(f'epoch_{e}', t)
row('POOL epochs 3000-4950', np.concatenate(allt)); row('POOL epochs 4500-4950', np.concatenate(allt[-10:]))
np.save(f'{OUT}/epoch_level_T.npy', np.stack(allt))
