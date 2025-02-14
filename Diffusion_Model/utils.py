from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn.functional import interpolate
import webdataset as wds
import io
import tarfile
import collections
import types
import einops


def save_images(images, output_path):
    """
    Save tensor images as both .npy and .png files.

    Args:
        images (torch.Tensor): Batch of images to save
        output_path (str or Path): Path where to save the images

    The function creates a figure with 3 rows and up to 8 columns,
    showing surface_toce, surface_soce, and ssh channels.
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


class TransformFields:
    """
    A class to handle data transformations for oceanographic data fields.

    This class provides methods to:
    - Standardize/normalize data
    - Handle padding and edge cases
    - Transform and untransform data between different formats

    Args:
        info_file (str): Path to the info file containing statistics
        fields (dict): Dictionary of fields to process
        normalisation (str): Type of normalization to use ('std', 'min-max', etc.)
    """

    def __init__(self, info_file, fields, normalisation='std') :

            VALID_NORMALIZATIONS = {'std', '7-std', '3-std', 'min-max', 'global-min-max'}

            def _validate_inputs(self):
                """Validate initialization parameters"""
                if self.normalisation not in self.VALID_NORMALIZATIONS:
                    raise ValueError(f"Normalization must be one of: {self.VALID_NORMALIZATIONS}")
                if not self.fields:
                    raise ValueError("Fields dictionary cannot be empty")
            self.fields = fields
            self.get_infos(info_file)
            self.data_shape = None
            self.padding =  {'xup' : 1, 'xdown' : 1, 'yup' : 5, 'ydown' : 4, 'val' : 0}
            self.normalisation = normalisation

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

            if array.shape[0] == 1 :
                array = array[0]

            array = self.replaceEdges(array, feature, val=np.nan)

            array = self.unstandardize_4D(array, feature)
            sample[feature] =  array
        sample = {key+'.npy' : val for key, val in sample.items()}
        return sample


    def stride_concat(self, sample) :
        return np.concatenate([sample[key][sl] for key, sl in self.fields.items()])

    def un_stride_concat(self, data, interpolation=True) :
        idx = 0
        sample = {}
        for key in self.fields.keys() :
            oz = self.infos['shape'][key][0] # original z
            levels = len(np.arange(oz)[self.fields[key]])
            field =  data[idx:levels+idx]
            idx += levels
            if interpolation :
                sample[key] = interpolate(field[None, None], size=(oz, field.shape[1], field.shape[2]), mode='trilinear')[0,0]
            else :
                sample[key] = field
        return sample


    def get_data_shape(self) :
        fake_data = {key : np.random.rand(*(self.infos['shape'][key])) for key in self.fields.keys()}
        fake_data = self.stride_concat(fake_data)
        fake_data = self.padData(fake_data, **self.padding)

        return fake_data.shape


    def interpolate_double(self, array) :
        """
        Take array with size (b, c, w, h) and return
        array with (b, c*2, w, h) using interpolation to fill the
        gaps.
        """
        r = (array[:, 0:-1] + array[:, 1:]) / 2
        interleaved_tensor = einops.rearrange([array[:, :-1], r], 'd b c h w -> b (c d) h w', d=2)
        return np.concatenate([interleaved_tensor, array[:, -1:]], axis=1)


    def get_infos(self, info_file: str) -> None:
        """
        Load and parse information from the tar file.

        Args:
            info_file: Path to the tar file containing dataset information

        Raises:
            FileNotFoundError: If info_file doesn't exist
            tarfile.ReadError: If tar file is corrupted
        """
        print(f'Reading infos in {info_file}')
        try:
            tar = tarfile.open(info_file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Info file not found: {info_file}")
        except tarfile.ReadError:
            raise tarfile.ReadError(f"Error reading tar file: {info_file}")

        target_path='infos/'
        max_return = 15

        self.infos = collections.defaultdict(dict)
        while max_return > 0 :
            member = tar.next()
            if member.path.startswith(target_path):
                feature, metric, _ = member.name.replace('infos/', '').split('.')
                self.infos[feature][metric] = np.load(io.BytesIO(tar.extractfile(member).read()))
                max_return -= 1

        self.infos['shape']['soce'] = self.infos['mask']['toce'].shape
        self.infos['shape']['toce'] = self.infos['mask']['soce'].shape
        self.infos['shape']['ssh'] = self.infos['mask']['ssh'][None].shape

    def standardize_4D(self,sample,feature):
        """
            Standardize the data given a mean and a std
        """
        if self.normalisation == 'std' :
            return (sample[f'{feature}'] - self.infos['mean'][feature]) / (self.infos['std'][feature] + 1e-8)

        elif self.normalisation == '7-std' :
                        return (sample[f'{feature}'] - self.infos['mean'][feature]) / (7*self.infos['std'][feature] + 1e-8)

        elif self.normalisation == '3-std' :
                        return (sample[f'{feature}'] - self.infos['mean'][feature]) / (3*self.infos['std'][feature] + 1e-8)

        elif self.normalisation == 'min-max' :
            return 2*(sample[f'{feature}'] - self.infos['min'][feature]) / (self.infos['max'][feature] - self.infos['min'][feature]) - 1

        elif self.normalisation == 'global-min-max' :
            min, max = self.infos['min'][feature].min(), self.infos['max'][feature].max()
            return 2*(sample[f'{feature}'] - min) / (max - min) - 1

    def unstandardize_4D(self, sample, feature):
        """
        Unstandardize the data based on the normalization type used
        Args:
            sample: The normalized data to be unstandardized
            feature: The feature name/key
        Returns:
            Unstandardized data
        """
        if self.normalisation == 'std':
            return (sample * (self.infos['std'][feature])) + self.infos['mean'][feature]

        elif self.normalisation == '7-std' :
                        return (sample * (7*self.infos['std'][feature])) + self.infos['mean'][feature]

        elif self.normalisation == '3-std' :
                        return (sample * (3*self.infos['std'][feature])) + self.infos['mean'][feature]

        elif self.normalisation == 'min-max':
            return (sample + 1) * (self.infos['max'][feature] - self.infos['min'][feature]) / 2 + self.infos['min'][feature]

        elif self.normalisation == 'global-min-max' :
            min, max = self.infos['min'][feature].min(), self.infos['max'][feature].max()
            return (sample + 1) * (max - min) / 2 + min

    def replaceEdges(self,data,feature,val):
        """
            Replace edges by a values. Default is 0
            data : batch, depth, x, y
        """
        data[self.infos['mask'][feature]] = val
        return data

    def padData(self,dataset,xup,xdown,yup,ydown,val):
        """
            pad data on axis x and y
        """
        return np.pad(dataset, ((0, 0), (yup, ydown), (xup, xdown)), mode='constant',constant_values=val)

    def unpadData(self,dataset,xup,xdown,yup,ydown,val):
        return dataset[:,yup:-ydown, xup:-xdown]

def get_data_shape(self) :
    return self.get_transform().get_data_shape()

def get_transform(self) :
    return self.dataset.pipeline[-1].args[0].transforms[0]


def get_dataloader(tar_file, fields=None, normalisation=None, batch_size=5, transform=True, shuffle=True):
    """
    Create a DataLoader for the dataset with specified transformations.

    Args:
        tar_file (str): Path to the WebDataset tar file
        fields (dict, optional): Dictionary specifying fields to load
        normalisation (str, optional): Type of normalization to apply
        batch_size (int, default=5): Batch size for the dataloader
        transform (bool, default=True): Whether to apply transformations
        shuffle (bool, default=True): Whether to shuffle the dataset

    Returns:
        DataLoader: PyTorch DataLoader object with additional methods
    """
    dataset = wds.WebDataset(tar_file).select(lambda x : 'infos' not in x['__key__'])

    if shuffle :
        dataset=dataset.shuffle(1000)

    dataset = dataset.decode()

    if transform :
        tr = TransformFields(info_file=tar_file, fields=fields, normalisation=normalisation)
        composed = transforms.Compose([tr])
        dataset = dataset.map(composed)

    dl = DataLoader(dataset=dataset, batch_size=batch_size)
    if transform :
        dl.get_transform = types.MethodType(get_transform, dl)
        dl.get_data_shape = types.MethodType(get_data_shape, dl)
    return dl

if __name__ == '__main__' :
    from configs.base_config import *
    config = TrainingConfig()
    train_dataloader = get_dataloader(config.data_file, batch_size=config.train_batch_size, fields=config.fields, normalisation=config.normalisation)
    config.data_shape = train_dataloader.get_data_shape()
    idt = iter(train_dataloader)
    b = next(idt)
