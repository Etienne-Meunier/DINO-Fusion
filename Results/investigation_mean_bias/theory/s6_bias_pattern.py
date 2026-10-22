"""Pattern correlation of the ensemble-mean offset field (state minus data climatology, ocean cells) between the
current run and the training-time samples of each epoch: which checkpoint does the local unet behave like?"""
from load import *
from glob import glob
data = load_data(200); clim = data.mean(0)
def offset(x): return (x.mean(0) - clim)[ocean_n]
cur = offset(np.load(RUNS['cur_none']))
cur_b = offset(np.load(RUNS['cur_none_border0']))
old = np.load(RUNS['old200_a'], mmap_mode='r')
refs = {'old200[:8]': offset(np.array(old[:8])), 'old200[8:16]': offset(np.array(old[8:16])), 'old200_all': offset(np.array(old)), 'data[100:108]': offset(data[100:108])}
eps = sorted(glob(f'{os.path.dirname(INF)}/epoch_*.npy'), key=lambda s: int(s.split('_')[-1][:-4]))
for p in eps:
    e = int(p.split('_')[-1][:-4])
    if e % 250 == 0 or e >= 3400 and e <= 4000 or e >= 4800: refs[f'epoch_{e}'] = offset(np.load(p))
cc = lambda a, b: np.corrcoef(a, b)[0, 1]
print('ref                corr(offset_ref, offset_cur_none)  corr(.., cur_none_border0)  rms(offset_ref)')
for k, v in refs.items():
    print(f'{k:18s} {cc(v, cur):7.3f} {cc(v, cur_b):7.3f}   {np.sqrt((v**2).mean()):.3f}')
print('cur_none vs cur_none_border0 offset corr %.3f; rms cur %.3f' % (cc(cur, cur_b), np.sqrt((cur**2).mean())))
c1, c2 = np.load(RUNS['cur_none'])[:8], np.load(RUNS['cur_none'])[8:]
print('cur first8 vs last8 offset corr %.3f (independent noise, same weights)' % cc(offset(c1), offset(c2)))
