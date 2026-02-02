from __init__ import PRP; import sys


from Diffusion_Model.pipelines.State import Observer
from collections.abc import Callable
import torch
import numpy as np
import abc
from ShapeChecker import ShapeCheck
import einops
from ipdb import set_trace

## --- STRUCTURE ----

class Beta:
    def __init__(self, beta_init, params_beta={}, beta_type='constant'):
        self.beta_type = beta_type
        self.value = beta_init
        self.prb = params_beta

    def update(self, t, esp=None):
        match self.beta_type : 
            case 'constant' : 
                pass # Never change beta if constant
            case 'decreasing':
                l, k, n = self.prb['l'], self.prb['k'], self.prb['max_iter']
                self.value += l*self.beta * torch.exp(((1-t/n) - 1)*k)
            case 'eq' :
                self.value += self.prb['eta'] * esp
            case 'ineq':
                self.value = (self.value + self.prb['eta'] * esp).clamp(0.0)
            case _:
                raise NameError(f'{self.beta_type} not known')

class FunctionGradient() : 
    def __init__(self):
        pass

    @abc.abstractmethod
    def __call__(field : torch.tensor) -> torch.tensor :
        pass

class Constraint() :
    def __call__(self, field : torch.tensor, t : int)-> torch.tensor :
        """
            Takes the normlized field and time and return 
            and increment on the normlized field
        """
        return torch.zeros_like(field)

class Projection() : 
    def __call__(self, field : torch.tensor)-> torch.tensor :
        """
            Takes the normalized field and return 
            a projection of it
        """
        return field


class GradientConstraint(Constraint) : 
    def __init__(self, beta : Beta, gradient_f : FunctionGradient):
        super().__init__()
        self.beta = beta
        self.gradient_f = gradient_f
    
    def __call__(self, field, t) : 
        """
        Take as input state, compute the correction 
        and return the corrected state.normalized
        """
        grad = self.gradient_f(field)
        self.beta.update(t, esp=torch.mean(torch.nansum(grad**2, dim=-1)))
        return self.beta.value * grad


class Loss : 
    def __init__(self, observer : Observer, dims=(-2,-1)) : 
        self.observer = observer
        self.dims = dims
        pass

    def __call__(self, field : torch.tensor) : 
        pass

## --- CONSTRAINTS ----

class ZeroMean(FunctionGradient) : 
    def __init__(self, dims=(-2, -1)):
        super().__init__() 
        self.dims = dims

    def __call__(self, field : torch.tensor) : 
        grad = field.mean(dim=self.dims, keepdim=True)
        return grad * torch.ones_like(field)

    def __str__(self):
        return 'GradientZeroMean'
    
class LossAD(FunctionGradient) : 
    def __init__(self, loss : Loss):
        super().__init__() 
        self.loss = loss

    def __call__(self, field : torch.tensor) : 
        with torch.enable_grad() :
            field.requires_grad_(True)
            field.grad = None
            l = self.loss(field)
            l.backward()
            return field.grad
        

class OptimProjection(Projection) :
    def __init__(self, loss : Loss, eps=0.1, max_opt_steps = 10):
        self.loss = loss
        self.eps = eps # Threshold at which we consider optimization successfull
        self.max_opt_steps = max_opt_steps
        super().__init__()

    def __call__(self, field : torch.tensor):
        field_o = field.clone() # Avoid inplace modification
        with torch.enable_grad() :
            field_o.requires_grad_(True)
            field_o.grad = None
            opt = torch.optim.LBFGS([field_o], history_size=10, max_iter=4, line_search_fn='strong_wolfe')
            def closure() : 
                opt.zero_grad()
                loss = self.loss(field_o)
                loss.backward()
                return loss
            for i in range(self.max_opt_steps) :
                if self.loss(field_o) < self.eps :
                    break
                opt.step(closure)    
            print(self.loss(field_o).item(), i)
            field_o.requires_grad_(False)
        return field_o


## --- Losses (for AD) ----

class MeanDensityProfile(Loss) : 
    def __init__(self, profile=\
                        np.loadtxt(f'{PRP}/Results/analysis_scripts/mean_density_train.txt'),
                        result='file',
                        **kwargs) : 
        super().__init__(**kwargs)
        self.mean_density = torch.tensor(profile)
        self.result = result

    def __call__(self, field):
        density_profile = self.observer.density_profile(field, result=self.result)
        self.mean_density = self.mean_density.to(density_profile)
        loss = torch.nansum((self.mean_density[None, :] - density_profile)**2)
        return loss
    

class MeanGradDensityProfile(Loss) : 
    def __init__(self, profile=\
                        np.loadtxt(f'{PRP}/Results/analysis_scripts/mean_density_train.txt'),
                        **kwargs) : 
        super().__init__(**kwargs)
        mean_density = torch.tensor(profile)
        mean_density[-1] = torch.nan
        self.grad_density = mean_density[:-1] - mean_density[1:]
        self.grad_density[-1] = 0.0 # Bottom layer with only nans

    def __call__(self, field):
        grad_density = self.observer.density_gradient(field)
        grad_density[:, -1] = 0.0 # Bottom layer with no nans
        grad_density_profile = grad_density.nanmean(dim=self.dims)
        self.grad_density = self.grad_density.to(grad_density_profile)

        loss = torch.nansum((self.grad_density[None, :] - grad_density_profile)**2)
        return loss
    

class NegativeGradDensityProfile(Loss) : 
    def __init__(self, **kwargs) : 
        super().__init__(**kwargs)

    def __call__(self, field):
        grad_density = self.observer.density_gradient(field)
        grad_density[:, -1] = 0.0 # Bottom layer with no nans
        grad_density_profile = grad_density.nanmean(dim=self.dims)

        loss = torch.nansum((grad_density_profile.clamp(0.0))**2)
        return loss


## --- Projections ---

class BorderZero(Projection):
    def __init__(self, observer):
        self.observer = observer

    def __call__(self, field):
        g = self.observer.unnormalized(field, 'masked')
        g[:, self.observer.normalizer.masker.mask] = 0.0
        return self.observer.normalizer.padder(g)

    def __str__(self):
        return f'BorderZeroConstraint - mask : {self.mask_path}'

class MeanDensityProfileProjection(Projection) :
    """
        Project the density profile on the mean density profile
    """ 
    def __init__(self, profile=\
                        np.loadtxt(f'{PRP}/Results/analysis_scripts/mean_density_train_strided.txt'),
                        **kwargs) : 
        super().__init__(**kwargs)
        self.mean_density = torch.tensor(profile) # (Z)

    def __call__(self, density_profile) : 
        """
            density profile : (B, Z, *)  where * can be spatial (I, J) or even nothing
        """ 
        self.mean_density = self.mean_density.to(density_profile)
        return self.mean_density[None, :, *((density_profile.ndim-2)*[None])] * torch.ones_like(density_profile)
    

class IsotonicDensity(Projection) :
    """
        Project the density profile on the mean density profile
    """ 
    def __init__(self, **kwargs) : 
        super().__init__(**kwargs)
        from scipy.optimize import isotonic_regression
        self.isof = isotonic_regression

    def __call__(self, density_profile) : 
        """
            density profile : (B, Z, *)  where * can be spatial (I, J) or even nothing
        """ 
        r = np.apply_along_axis(lambda o : self.isof(o).x, axis=1, arr=density_profile.detach().cpu().numpy())
        return torch.tensor(r).to(density_profile)
    

    