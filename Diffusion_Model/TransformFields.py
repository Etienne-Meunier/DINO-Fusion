import torch
from einops import rearrange
import torch.nn.functional as F 
import re, tarfile, collections, io
from copy import deepcopy
from tensordict.tensordict import TensorDict
import numpy as np
import collections
from ipdb import set_trace

class Strider :

    def __init__(self, stride_pattern, field_len, interpolation=True, name='', axis=0) : 
        """
            Stride field on given axis
            stride_pattern (slide) : slide to get from the field
            field_len (int) : len along the axis to stride
            interpolation (bool) : Use interpolation for un-striding
            name (str) : optional strider name
            axis (int) : axis to stride on
        """
        assert axis == 0, NotImplemented('Stride is only implemented along the first axis')
        self.stride_pattern = stride_pattern
        self.interpolation = interpolation
        self.name = name
        self.field_len = field_len
        self.strided_field_len = self._get_strided_len(field_len, stride_pattern)

    def _get_strided_len(self, field_len, stride_pattern) : 
        rfield = torch.zeros((field_len,))
        strided_field = rfield[stride_pattern]
        return strided_field.shape[0]

    def __call__(self, field):
        assert field.shape[0] == self.field_len, f'Before Stride ({self.name=}) : {field.shape[0]=} != {self.field_len=}'
        strided_field = field[self.stride_pattern]
        assert strided_field.shape[0] == self.strided_field_len,f'After Stride ({self.name=}) : {strided_field.shape[0]=} != {self.strided_field_len=}'
        return strided_field
    
    def __uncall__(self, strided_field) : 
        assert self.interpolation == True, NotImplementedError(f'Interpolation {self.interpolation} not implemented')
        assert strided_field.shape[0] == self.strided_field_len, f'Before Unstride ({self.name=}) : {strided_field.shape[0]=} != {self.strided_field_len=}'

        field = F.interpolate(rearrange(strided_field, 'z x y -> x y z'), size=(self.field_len), mode='linear')
        field = rearrange(field, 'x y z -> z x y')
        assert field.shape[0] == self.field_len, f'After Unstride ({self.name=}) : {field.shape[0]=} != {self.field_len=}'
        return field

    def __str__(self):
        return f'Strider ({self.name}) - {self.field_shape=} {self.stride_pattern=} - {self.field_len=} -> {self.strided_field_len=}'
    
class Normalisation :
    def __init__(self) : 
        pass

    def __call__(self, field) :
        return field 

class StdNorm(Normalisation) :
    def __init__(self, mean, std, factor):
        self.mean = mean
        self.std = std
        self.factor = factor

    def __call__(self, field):
        return (field - self.mean) / (3*self.std + 1e-8)
    
    def __uncall__(self, normalised_field) : 
        return (normalised_field * (3*self.std)) + self.mean

    @staticmethod
    def parse_x_std(s: str):
        """
        Returns (is_std: bool, x: int|None)
        """
        _pattern = re.compile(r'^(\d+)-std$')
        m = _pattern.match(s or '')
        if not m:
            return False, None
        return True, int(m.group(1))


class Normaliser : 
    def __init__(self, infos, normalisation_type) :
        """
            infos (TensorDict) : infos dictionnary with 'mean', 'std', 'min', 'max'
            normalisation_type (str) : type of normalisation to use
        """
        self.infos = infos
        self.normalization = self.get_normalization(normalisation_type)

    def get_normalization(self, normalisation_type) :
        if normalisation_type == '' :
            return Normalisation
        elif StdNorm.parse_x_std(normalisation_type)[0] :
            return StdNorm(self.infos['mean'], self.infos['std'], StdNorm.parse_x_std(normalisation_type)[1])
        else : 
            raise NotImplementedError(f'{normalisation_type=} not implemented')
        
    def __call__(self, field):
        return self.normalization.__call__(field)
    
    def __uncall__(self, normalised_field) : 
        return self.normalization.__uncall__(normalised_field)
    

class Masker : 
    def __init__(self, mask, val_mask=0, val_unmask=torch.nan): 
        """
            mask (tensor) : values to mask
            val_mask (float) : value to use for masking
            val_unmaks (float) : value to use for unmasking
        """
        self.mask = mask
        self.val_mask = val_mask
        self.val_unmask = val_unmask

    def __call__(self, field) : 
        assert field.shape == self.mask.shape, f'Masker call : {field.shape=} != {self.mask.shape=}'        
        return torch.where(self.mask, torch.full_like(field, self.val_mask), field)

    def __uncall__(self, masked_field) : 
        assert masked_field.shape == self.mask.shape, f'Masker uncall : {masked_field.shape=} != {self.mask.shape=}'
        return torch.where(self.mask, torch.full_like(masked_field, self.val_unmask), masked_field)


class Concatener : 
    def __init__(self, concat_lens, axis=0, name='') : 
        """
        Concatenate tensor
        concat_lens ({str : (int)}) : order of the fields + shapes along axis 
        axis (int) : axis to concatenate on (default = 0)
        name (str) : optional name
        """
        self.concat_lens = concat_lens
        self.axis = axis

    def __call__(self, fields_dict) :
        """
        fields_dict (dict) : dictionnary of tensor to concatenate
        """ 
        concatenation = []
        for c, s in self.concat_lens.items() : 
            f = fields_dict[c]
            assert f.shape[self.axis] == s, 'Error in shapes during concatenation'
            concatenation.append(f)
        concatenation = torch.concat(concatenation, axis=self.axis)
        return concatenation
    

    def __uncall__(self, concatenation) : 
        fields_list = torch.split(concatenation, list(self.concat_lens.values()), dim=self.axis)
        fields_dict = {}
        for i, (k, v) in enumerate(self.concat_lens.items()) : 
            assert v == fields_list[i].shape[0], f'{v=}!={fields_list[i].shape[0]=}'
            fields_dict[k] = fields_list[i]
        return TensorDict(fields_dict, batch_size=[])
    

class Padder : 
      def __init__(self, paddings=(1, 1, 5, 4), value=0):
          """
          paddings : (xup, xdown, yup, ydown)
          """
          self.paddings = paddings
          self.value = value
        
      def __call__(self, field):
          return F.pad(field, self.paddings, mode='constant', value=self.value) #torch version
          
      def __uncall__(self, padded_field) : 
          xup, xdown, yup, ydown = self.paddings
          return padded_field[:,yup:-ydown, xup:-xdown]
      

class DimensionCheck :

    def __call__(self, field) : 
        if not torch.is_tensor(field) :
            return field
        elif field.ndim == 3 : 
          return field
        elif field.ndim == 2 : 
          return field[None]
        else :
            raise Exception(f'Dim check call : error in number of dimension {field.ndim}')
        
    def __uncall__(self, field) : 
        if not torch.is_tensor(field) :
            return field
        assert field.ndim == 3, f'Dim check uncall : error in number of dimension {field.ndim}'
        if field.shape[0] == 1 :
            return field[0]
        return field


class TransformationPipeline : 

    """
        file -> strided -> concatenated -> normalized -> input
    """

    def __init__(self, infos, stride_patterns, normalisation_type, device='cpu'):
        self.device = device
        self.infos = infos.to(device)

        # Dimension check 
        self.dimcheck = DimensionCheck()

        # Striders 
        self.stride_patterns = stride_patterns
        self.striders = self.build_striders()

        # Concatenaters
        concat_lens = {k : s.strided_field_len for k, s in self.striders.items()}
        self.concatener = Concatener(concat_lens)

        # Infos
        self.strided_infos, self.concatenated_infos = self.prepare_infos(self.infos)


        # Normaliser 
        self.normaliser = Normaliser(self.concatenated_infos, normalisation_type=normalisation_type)
    
        # Masker 
        self.masker = Masker(self.concatenated_infos['mask'])

        # Pad
        self.padder = Padder()


    
    def prepare_infos(self, infos) :
        # Stride infos
        strided_infos = deepcopy(infos)
        concatenated_infos = {}
        for m in ['mask', 'mean', 'std'] : 
            for k, v in infos[m].items() : 
                strided_infos[m][k] = self.striders[k](v)
            concatenated_infos[m] = self.concatener(strided_infos[m])
        return strided_infos, TensorDict(concatenated_infos, batch_size=[])    
        
    def build_striders(self) :
        return {k : Strider(sp, field_len=self.infos['shape'][k][0], name=k) for k, sp in self.stride_patterns.items() }             

    def __call__(self, fields_dict, result='input'):
        #set_trace()
        del fields_dict['__key__']

        if result == 'file' : return f

        # Dim check
        f = TensorDict(fields_dict, batch_size=[]).apply(self.dimcheck)

        # Stride separately each modality
        f = TensorDict({k : s(f[k]) for k, s in self.striders.items()}, batch_size=[]).to(self.device)

        if result == 'strided' : return f
        
        # Concatenate modalities 
        f = self.concatener(f)

        if result == 'concatened' : return f
    
        # Normalise volume
        f = self.normaliser(f)

        if result == 'normalised' : return f

        # Replace Edges for the volume
        f = self.masker(f)

        # Pad volume
        f = self.padder(f)
        return f

    def __uncall__(self, concatenated_fields, result='file') : 
        # Unpad
        f = self.padder.__uncall__(concatenated_fields.to(self.device))

        if result == 'masked' : return f

        # Restore Edges
        f = self.masker.__uncall__(f)

        if result == 'normalized' : return f

        # Normalise volume
        f = self.normaliser.__uncall__(f)

        if result == 'concatenated' : return f

        # Unconcat 
        f = self.concatener.__uncall__(f)

        if result == 'strided' : return f

        # Unstride
        f = TensorDict({k : s.__uncall__(f[k]) for k, s in self.striders.items()}, batch_size=[])

        f = f.apply(self.dimcheck.__uncall__)
        return f
    
    def fake_data(self) : 
        fake_data = TensorDict({k  : torch.randn(*self.infos['shape'][k]) for k in self.stride_patterns.keys()}, batch_size=[])
        return fake_data

    def get_output_shape(self) : 
        fake_data = self.fake_data()
        fake_data = self.__call__(fake_data)
        return fake_data.shape




if __name__ == '__main__' : 
    from configs.base_config import *
    from tensordict.tensordict import TensorDict
    import torch

    config = TrainingConfig()
    tr = TransformationPipeline(config.data_file, config.fields, config.normalisation)

    print('Pipeline output shape :', tr.get_output_shape())


    shape = (36, 199, 62)
    tdict = TensorDict({'soce' : torch.rand(*shape), 'toce' : torch.rand(*shape), 'ssh' : torch.rand((1, shape[1], shape[2]))}, batch_size=[])

    tdict_transformed = tr(tdict)
    tdict_reversed = tr.__uncall__(tdict_transformed)