from glob import glob
import numpy as np
import matplotlib as mpl
mpl.rcParams['image.origin'] = 'lower'
path = '/Volumes/LoCe/oceandata/Dino-Fusion/dino_1_4_degree_coarse_130924/'

ims = glob(path + '*.npy')

for im in ims :
    a = np.load(im)
    if a.ndim == 3 :
        a = a[0]
    plt.imshow(a)
    plt.axis('off')
    plt.savefig(im.replace('.npy', '.png'))
