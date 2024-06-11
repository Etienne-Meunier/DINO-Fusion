import xarray as xr
import numpy as np
from glob import glob
from tqdm import tqdm


restarts = glob('/gpfsstore/rech/omr/uym68qx/nemo_output/DINO/Dinoffusion/1_4degree/restart*')
save_path = '/gpfsstore/rech/gzi/ufk69pe/DINO-Fusion-Data/1_4_degree/'

idx = 0
for restart in restarts :
    try :
        data_TS = xr.open_mfdataset(restart + '/DINO_10d_grid_T_3D.nc')
        data_SSH = xr.open_mfdataset(restart + '/DINO_10d_grid_T_2D.nc', decode_times=False)
        toce = data_TS.toce_inst.values
        soce = data_TS.soce_inst.values
        ssh = data_SSH.ssh_inst.values
        for i in range(len(toce)) :
            np.save(toce[i], save_path + f'toce_{idx}.npy')
            np.save(soce[i], save_path + f'soce_{idx}.npy')
            np.save(ssh[i], save_path + f'ssh_{idx}.npy')
            idx += 1
    except Exception as e : 
        print(restart, e)

