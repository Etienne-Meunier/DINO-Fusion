import webdataset as wds
from torch.utils.data import DataLoader
import types, tarfile, collections, io, torch
from torchvision import transforms
from tensordict.tensordict import TensorDict
import numpy as np
from TransformFields import TransformationPipeline
import re

@staticmethod
def get_infos(info_file):
        print(f'Reading infos in {info_file}')
        tar = tarfile.open(info_file)

        target_path='infos/'
        max_return = 15

        infos = collections.defaultdict(dict)
        while max_return > 0 :
            member = tar.next()
            if member.path.startswith(target_path):
                feature, metric, _ = member.name.replace('infos/', '').split('.')
                data = np.load(io.BytesIO(tar.extractfile(member).read()))
                if metric == 'ssh' : 
                    data = data[None] # Add leading dimension for SSH 
                if feature == 'mask':
                    infos[feature][metric] = torch.tensor(data, dtype=torch.bool)
                else: 
                    infos[feature][metric] = torch.tensor(data, dtype=torch.float32)

                max_return -= 1

        infos['shape']['soce'] = infos['mask']['toce'].shape
        infos['shape']['toce'] = infos['mask']['soce'].shape
        infos['shape']['ssh'] = infos['mask']['ssh'].shape

        infos = TensorDict(infos, batch_size=[])
        return infos

def strip_ext(sample):
    new_sample = {}
    for k, v in sample.items():
        if k.startswith("__"):
            new_sample[k] = v
        else:
            # remove extension like .npy
            new_key = re.sub(r"\.npy$", "", k)
            new_sample[new_key] = v
    return new_sample

def drop_meta(sample):
    return {k: v for k, v in sample.items() if not k.startswith("__")}

def get_dataloader(tar_file, transform=None, batch_size=5, shuffle=True) :
    dataset = wds.WebDataset(tar_file).select(lambda x : 'infos' not in x['__key__'])

    if shuffle :
        dataset=dataset.shuffle(1000)

    dataset = dataset.decode().map(strip_ext)
    dataset = dataset.map(lambda s: {k: s[k] for k in ["toce", "ssh", "soce"] if k in s})

    if transform is not None:
        composed = transforms.Compose([transform])
        dataset = dataset.map(composed)


    dl = DataLoader(dataset=dataset, batch_size=batch_size)

    if transform is not None:
        dl.get_transform = types.MethodType(lambda _ : transform, dl)
        dl.get_data_shape = types.MethodType(transform.get_output_shape, dl)
    return dl


if __name__ == '__main__' :
    from configs.base_config import *
    config = TrainingConfig()
    tr = TransformationPipeline(get_infos(config.data_file), config.fields, config.normalisation, device='mps')
    train_dataloader = get_dataloader(config.data_file, tr)

    idt = iter(train_dataloader)
    b = next(idt)
    print(f'{b.shape=}')