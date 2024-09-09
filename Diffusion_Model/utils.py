from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
import webdataset as wds
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data.distributed import DistributedSampler
from ipdb import set_trace
import einops
import io
import tarfile
import collections
import types


def save_images(images, output_path) :
    """
    XXX
    """
    images = images.cpu()
    Path(output_path).parent.mkdir(exist_ok=True)

    np.save(str(output_path).replace(".png",".npy"),images)
    try :

        fig, axs = plt.subplots(3, min(len(images), 8), figsize=(15,15))
        for ci, c in enumerate([(0, 'surface_toce'), (8, 'surface_soce'), (-1, 'ssh')]) :
            for b in range(min(len(images), 8)) :
                axs[ci, b].imshow(images[b, c[0]], origin='lower')

                if b == 0 :
                    axs[ci, b].set_title(c[1])
                else :
                    axs[ci, b].axis('off')

        fig.savefig(str(output_path), bbox_inches='tight')
    except Exception as e :
        print(f'Error drawing figure {e}')


class TransformFields :

        def __init__(self, info_file, fields) :
            self.fields = fields
            self.get_infos(info_file)
            self.data_shape = None
            self.padding =  {'xup' : 3, 'xdown' : 3, 'yup' : 1, 'ydown' : 2, 'val' : 0}


        def __call__(self, sample) :
            dico = {}
            sample = {key.replace('.npy', '') : val for key, val in sample.items()}
            for feature in ["soce","toce","ssh"]:
                #1. standardize
                data = self.standardize_4D(sample,feature)
                #2. replace padding and edges by 0
                data = self.replaceEdges(data,feature,val=0)

                if data.ndim == 2 :
                    data = data[None]
                #3. pad data
                data = self.padData(data, **self.padding)
                dico[feature] = data
            #set_trace()
            data = self.stride_concat(dico)
            return data

        def uncall(self, data) :
            sample = self.un_stride_concat(data)
            for feature in ["soce","toce","ssh"]:
                array = sample[feature]
                array = self.unpadData(array,**self.padding)

                array = self.replaceEdges(array, feature, val=np.nan)

                array = self.unstandardize_4D(array, feature)
                sample[feature] =  array
            sample = {key+'.npy' : val for key, val in sample.items()}
            return sample


        def stride_concat(self, sample) :
            return np.concatenate([sample[key][sl] for key, sl in self.fields.items()])

        def get_data_shape(self) :
            fake_data = {key : np.random.rand(*(self.infos[key]['shape'])) for key in self.fields.keys()}
            fake_data = self.stride_concat(fake_data)
            fake_data = self.padData(fake_data, **self.padding)

            return fake_data.shape

        def un_stride_concat(self, data) :
            assert self.step == 2, 'Only works for step=2 for now'
            assert (len(data.shape) == 4), 'Only works for arrays like (b c h w) for now'
            ranges = {'toce' : slice(0, 18), 'soce' : slice(18, 36), 'ssh' : slice(36,None)}
            sample =  {key : data[:, val] for key, val in ranges.items()}
            sample['toce'] = self.interpolate_double(sample['toce'])
            sample['soce'] = self.interpolate_double(sample['soce'])
            return sample


        def interpolate_double(self, array) :
            """
            Take array with size (b, c, w, h) and return
            array with (b, c*2, w, h) using interpolation to fill the
            gaps.
            """
            r = (array[:, 0:-1] + array[:, 1:]) / 2
            interleaved_tensor = einops.rearrange([array[:, :-1], r], 'd b c h w -> b (c d) h w', d=2)
            return np.concatenate([interleaved_tensor, array[:, -1:]], axis=1)


        def get_infos(self, info_file):
            print(f'Reading infos in {info_file}')
            tar = tarfile.open(info_file)

            target_path='infos/'
            max_return = 9

            self.infos = collections.defaultdict(dict)
            while max_return > 0 :
                member = tar.next()
                if member.path.startswith(target_path):
                    feature, metric, _ = member.name.replace('infos/', '').split('.')
                    self.infos[feature][metric] = np.load(io.BytesIO(tar.extractfile(member).read()))
                    max_return -= 1

            self.infos['soce']['shape'] = (36, 797, 242)
            self.infos['toce']['shape'] = (36, 797, 242)
            self.infos['ssh']['shape'] = (1, 797, 242)

        def standardize_4D(self,sample,feature):
            """
                Standardize the data given a mean and a std
            """
            return (sample[f'{feature}'] - self.infos['mean'][feature]) / (2*self.infos['std'][feature] + 1e-8)

        def unstandardize_4D(self,sample,feature):
            """
                Standardize the data given a mean and a std
            """
            return (sample * (2*self.infos['std'][feature])) + self.infos['mean'][feature]

        def replaceEdges(self,data,feature,val):
            """
                Replace edges by a values. Default is 0
                data : batch, depth, x, y
            """
            data[self.infos['mask'][feature]] = val

            # Temporar : remove when using new tar
            data = np.nan_to_num(data, val)
            return data

        def padData(self,dataset,xup,xdown,yup,ydown,val):
            """
                pad data on axis x and y
            """
            return np.pad(dataset, ((0, 0), (yup, ydown), (xup, xdown)), mode='constant',constant_values=val)

        def unpadData(self,dataset,xup,xdown,yup,ydown):
            return dataset[:, :,yup:-ydown, xup:-xdown]

def get_data_shape(self) :
    return self.dataset.pipeline[-1].args[0].transforms[0].get_data_shape()

def get_dataloader(tar_file, fields, batch_size=5) :
    transform = TransformFields(info_file=tar_file, fields=fields)
    composed = transforms.Compose([transform])
    dataset = wds.WebDataset(tar_file).select(lambda x : 'infos' not in x['__key__']).shuffle(100).decode().map(composed)
    dl = DataLoader(dataset=dataset, batch_size=batch_size)
    dl.get_data_shape = types.MethodType(get_data_shape, dl)
    return dl

if __name__ == '__main__' :
    from configs.base_config import *
    config = TrainingConfig()
    train_dataloader = get_dataloader(config.data_file, batch_size=config.train_batch_size, fields=config.fields)
    config.data_shape = train_dataloader.get_data_shape()
    idt = iter(train_dataloader)
    next(idt)
