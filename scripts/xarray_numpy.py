import xarray as xr
import numpy as np
from glob import glob
import json
from tqdm import tqdm
from pathlib import Path

class Infos : 
    
    def __init__(self, keys) : 
        self.infos = {}
        self.global_counter = 0
        for key in keys :
            self.infos[key] = {'mean' :  None, 'std' : None, 'mask' : None, 'counter' : 0}


    @staticmethod
    def s_plus(old, new) : 
        if old is None : 
            old = new * 0.0
        old += new
        return old 
        

    def update(self, key, mean, std, mask) : 
        self.infos[key]['mean'] = Infos.s_plus(self.infos[key]['mean'], mean)
        self.infos[key]['std']  =  Infos.s_plus(self.infos[key]['std'], std)
        if self.infos[key]['mask'] is None : 
            self.infos[key]['mask'] = mask
        self.infos[key]['counter'] += 1

    def normalise(self) : 
        for key in self.infos.keys : 
            self.infos[key]['mean'] /= self.infos[key]['counter']
            self.infos[key]['std'] /= self.infos[key]['counter']
            

    def save(self, save_path, file_names) : 
    

        Path(save_path + '/infos/').mkdir(exist_ok=True, parents=True)
        for key in self.infos.keys() :
            assert self.global_counter == self.infos[key]['counter'], f'Counter error {key} : {self.global_counter} != {self.infos[key]["counter"]}'
            write_file(self.infos[key]['mean'], f'infos/mean.{key}\n', save_path, file_names)
            write_file(self.infos[key]['std'], f'infos/std.{key}\n', save_path, file_names)
            write_file(self.infos[key]['mask'], f'infos/mask.{key}\n', save_path, file_names)
        
def write_file(array, name, save_path, file_names) : 
    np.save(save_path + name, array)
    file_names.writelines(name+'\n')

def fill_na(array, value=0) : 
    array[array == value] = np.nan
    return array

def convert_nc(restart_path, save_path, file_names, infos) : 
    data_TS = xr.open_mfdataset(restart_path + '/DINO_10d_grid_T_3D.nc')
    data_SSH = xr.open_mfdataset(restart_path + '/DINO_10d_grid_T_2D.nc', decode_times=False)

    data = {}
    data['toce.npy'] = fill_na(data_TS.toce_inst.values)
    data['soce.npy'] = fill_na(data_TS.soce_inst.values)
    data['ssh.npy'] = fill_na(data_SSH.ssh_inst.values)
    assert len(data['toce.npy']) == len(data['soce.npy']) == len(data['ssh.npy']), 'Inequal length'

    for i in tqdm(range(len(data['toce.npy']))) :
        for key in data.keys() :
            name=f"{infos.infos[key]['counter']:05d}.{key}"
            write_file(data[key][i], name, save_path, file_names)
            infos.update(key,
                         np.nanmean(data[key][i], axis=(-1,-2), keepdims=True),
                         np.nanstd(data[key][i], axis=(-1,-2), keepdims=True),
                         (data[key][i] == np.nan))
        infos.global_counter += 1
    return counter


if __name__ == '__main__' : 
    restarts = glob('/gpfsstore/rech/omr/uym68qx/nemo_output/DINO/Dinoffusion/1_4degree/restart*')
    save_path = '/gpfsstore/rech/gzi/ufk69pe/DINO-Fusion-Data/1_4_degree/'

    counter, infos = 0, Infos(keys=['toce.npy', 'soce.npy', 'ssh.npy'])

    with open(save_path + 'list_files.txt', 'w+') as f:
        for restart in restarts :
            try : 
                convert_nc(restart, save_path, f, infos)
            except Exception as e : 
                print(restart, e)
        infos.save(save_path, f)

