import xarray as xr
import numpy as np
from glob import glob
from tqdm import tqdm


restarts = glob('/gpfsstore/rech/omr/uym68qx/nemo_output/DINO/Dinoffusion/1_4degree/restart*')
save_path = '/gpfsstore/rech/gzi/ufk69pe/DINO-Fusion-Data/1_4_degree'

f = open(save_path + '_files.txt')
idx = 0
for restart in restarts :
    try :
        data_TS = xr.open_mfdataset(restart + '/DINO_10d_grid_T_3D.nc')
        data_SSH = xr.open_mfdataset(restart + '/DINO_10d_grid_T_2D.nc', decode_times=False)
    except Exception as e : 
        print(restart, e)
        continue

    data = {}
    infos = {}
    data['toce.npy'] = data_TS.toce_inst.values
    data['soce.npy'] = data_TS.soce_inst.values
    data['ssh.npy'] = data_SSH.ssh_inst.values
    for i in tqdm(range(len(data['toce']))) :
        for key in data.keys() :
            name=f'{idx:05d}.{key}'    
            np.save(save_path + name, data[key][i])
            f.writelines(name+'\n')
            idx += 1
    del data_TS, data_SSH, data
f.close()

d = data['toce.npy']
x_mean = np.nanmean(d, axis=(0,2,3), keepdims=False)
x_std  = np.nanstd(d, axis=(0,2,3), keepdims=False)
