import torch
from ipdb import set_trace
import numpy as np

class DiffusionConstraint:
    def __init__(self):
        pass

    def apply(self, x):
        raise NotImplementedError

class ZeroMeanConstraint(DiffusionConstraint):
    def __init__(self, dims=[-1, -2]):
        super().__init__()
        self.dims = dims

    def apply(self, x, t=None):
        mean = x.mean(dim=self.dims, keepdim=True)
        #print(f'apply constraint : {mean} {mean.shape}')
        return x - mean

class BorderZeroConstraint(DiffusionConstraint) :
    def __init__(self, mask='pipelines/border_mask.npy') :
        self.mask =  torch.tensor(np.load(mask))

    def apply(self, x, t=None) :
        x[:, self.mask] = 0.0
        return x


class ConditionalGeneration_SSHLow(DiffusionConstraint) :
    def __init__(self, field='pipelines/ssh_low.npy') :
        self.field =  torch.tensor(np.load(field))

    def apply(self, x, t=None) :
        x[:, -1] = self.field
        return x

class ConditionalGeneration_SSHHigh(DiffusionConstraint) :
    def __init__(self, field='pipelines/ssh_high.npy') :
        self.field =  torch.tensor(np.load(field))

    def apply(self, x, t=None) :
        x[:, -1] = self.field
        return x


class ConditionalGeneration_STempLow(DiffusionConstraint) :
    def __init__(self, field='pipelines/surface_temp_low.npy') :
        self.field =  torch.tensor(np.load(field))

    def apply(self, x, t=None) :
        x[:, 0] = self.field
        return x

class ConditionalGeneration_STempHigh(DiffusionConstraint) :
    def __init__(self, field='pipelines/surface_temp_high.npy') :
        self.field =  torch.tensor(np.load(field))

    def apply(self, x, t=None) :
        x[:, 0] = self.field
        return x


class Beta:
    def __init__(self, beta, beta_type='constant'):
        self.beta_type = beta_type
        self.beta = beta

    def get_beta(self, t, k=20, l=40):
        if self.beta_type == 'constant':
            return self.beta
        elif self.beta_type == 'decreasing':
            # Example: stronger constraint at the end of sampling
            return self.beta + l*self.beta * torch.exp((1-t/1000) * k - k)
        else:
            raise ValueError(f"Unknown beta type: {self.beta_type}. Available beta type: constant, decreasing ")


class GradientZeroMeanConstraint(DiffusionConstraint):
    def __init__(self, beta, beta_type='constant', dims=[-1, -2]):
        super().__init__()
        self.beta = Beta(beta=beta, beta_type=beta_type)
        self.dims = dims

    def apply(self, x, t=None, interior=None): #(slice(5, -5), slice(5, -5))
        # Handle interior region indexing for older Python versions
        if interior:
            region = x[..., interior[0], interior[1]]
            grad = region.mean(dim=self.dims, keepdim=True)
        else:
            grad = x.mean(dim=self.dims, keepdim=True)

        beta = self.beta.get_beta(t) if t is not None else self.beta
        print(f'apply constraint {beta}: {grad[0, :, 0,0]}')

        if interior:
            x[..., interior[0], interior[1]] -= beta * grad
        else:
            x -= beta * grad
        return x

