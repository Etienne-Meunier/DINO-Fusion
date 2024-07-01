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
    infos = Infos()
    data['toce.npy'] = data_TS.toce_inst.values
    data['soce.npy'] = data_TS.soce_inst.values
    data['ssh.npy'] = data_SSH.ssh_inst.values
    for i in tqdm(range(len(data['toce']))) :
        for key in data.keys() :
            name=f'{idx:05d}.{key}'    
            np.save(save_path + name, data[key][i])
            
            infos[key]['mean'] += np.nanmean(data[key][i], axis=(1,2), keepdims=True)[None]
            infos[key]['std'] += np.nanstd(data[key][i], axis=(1,2), keepdims=True)[None]
            f.writelines(name+'\n')
            idx += 1
    del data_TS, data_SSH, data
f.close()


class Infos : 
    
    def __init__(self, keys) : 
        self.infos = {}
        for key in keys :
            self.infos[key] = {'mean' :  0, 'std' : 0, 'counter' : 0}

    def update(self, key, mean, std) : 
        self.infos[key]['mean'] += mean
        self.infos[key]['std'] += std
        self.infos[key]['counter'] += 1

    def normalise(self) : 
        for key in self.infos.keys : 
            self.infos[key]['mean'] /= self.infos[key]['counter']
            self.infos[key]['std'] /= self.infos[key]['counter']


    


