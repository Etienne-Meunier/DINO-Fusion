import xarray as xr
import numpy as np
from glob import glob
import json
from tqdm import tqdm

class Infos : 
    
    def __init__(self, keys) : 
        self.infos = {}
        self.global_counter = 0
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

    def save(self, save_path) : 
        for key in self.infos.keys() :
            assert self.global_counter == self.infos[key]['counter'], f'Counter error {key} : {self.global_counter} != {self.infos[key]["counter"]}'
            with open(f'{save_path}/infos.{key}', 'w+') as j  : 
                json.dump(self.infos[key], j)
        


def convert_nc(restart_path, save_path, file_names, infos) : 
    data_TS = xr.open_mfdataset(restart_path + '/DINO_10d_grid_T_3D.nc')
    data_SSH = xr.open_mfdataset(restart_path + '/DINO_10d_grid_T_2D.nc', decode_times=False)

    data = {}
    data['toce.npy'] = data_TS.toce_inst.values
    data['soce.npy'] = data_TS.soce_inst.values
    data['ssh.npy'] = data_SSH.ssh_inst.values
    assert len(data['toce.npy']) == data['soce.npy'] == data['ssh.npy'], 'Inequal length'
    for i in tqdm(range(len(data['toce.npy']))) :
        for key in data.keys() :
            name=f"{infos['key']['counter']:05d}.{key}"
            np.save(save_path + name, data[key][i])
            infos.update(key,
                         np.nanmean(data[key][i], axis=(1,2)),
                         np.nanstd(data[key][i], axis=(1,2)))
            file_names.writelines(name+'\n')
        infos.global_counter += 1
    return counter


if __name__ == '__main__' : 
    restarts = glob('/gpfsstore/rech/omr/uym68qx/nemo_output/DINO/Dinoffusion/1_4degree/restart*')
    save_path = '/gpfsstore/rech/gzi/ufk69pe/DINO-Fusion-Data/1_4_degree'

    counter, infos = 0, Infos(keys=['toce.npy', 'soce.npy', 'ssh.npy'])

    with open(save_path + '_files.txt', 'w+') as f:
        for restart in restarts :
            try : 
                convert_nc(restart, save_path, f, infos)
            except Exception as e : 
                print(restart, e)

