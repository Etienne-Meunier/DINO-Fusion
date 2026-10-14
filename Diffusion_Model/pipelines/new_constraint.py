from __init__ import PRP; import sys


from pipelines.State import Observer
from collections.abc import Callable
import torch
import numpy as np
import abc
import einops

## --- STRUCTURE ----

class Beta:
    """
        Weight of the constraint gradient along sampling.
            constant    : beta = beta_init
            linear_up   : beta = 2 beta_init (1 - t/max_iter), strongest at the end of sampling
            linear_down : beta = 2 beta_init t/max_iter, strongest at the start of sampling
            eq          : beta <- beta + eta * C
            ineq        : beta <- max(beta + eta * (C - eps), 0), dual ascent of the constraint C <= eps
        The linear schedules have the same mean over sampling as the constant one.
    """
    def __init__(self, beta_init, params_beta={}, beta_type='constant'):
        self.beta_type = beta_type
        self.value = beta_init
        self.base = beta_init
        self.prb = params_beta

    def update(self, t, esp=None):
        match self.beta_type : 
            case 'constant' : 
                pass # Never change beta if constant
            case 'decreasing':
                l, k, n = self.prb['l'], self.prb['k'], self.prb['max_iter']
                self.value += l*self.value * torch.exp(((1-t/n) - 1)*k)
            case 'linear_up' :
                self.value = 2 * self.base * (1 - float(t) / self.prb['max_iter'])
            case 'linear_down' :
                self.value = 2 * self.base * float(t) / self.prb['max_iter']
            case 'eq' :
                self.value += self.prb['eta'] * esp
            case 'ineq':
                self.value = torch.as_tensor(self.value + self.prb['eta'] * (esp - self.prb.get('eps', 0.0))).clamp(0.0)
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
        grad, loss = self.gradient_f(field)
        self.beta.update(t, esp=loss)
        return self.beta.value * grad


class Loss : 
    def __init__(self, observer : Observer, dims=(-2,-1)) : 
        self.observer = observer
        self.dims = dims
        pass

    def __call__(self, field : torch.tensor) : 
        pass

## --- CONSTRAINTS ----
    
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
            return field.grad, l.detach()
        

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

class ZeroMean(Loss) : 
    def __init__(self, **kwargs) : 
        super().__init__(observer=None, **kwargs)
                            
    def __call__(self, field):
        mf = torch.nanmean(field, axis=(-2, -1))
        loss = torch.nansum(mf**2, axis=(0, 1))
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



class DistancetoIsotonic(Loss) : 
    """
        Squared distance of the density columns to the isotonic cone (density non-decreasing with depth).
        The projection is detached : the gradient of the squared distance to a convex set is 2 (x - P(x)).
    """
    def __init__(self, result='strided', **kwargs) : 
        super().__init__(**kwargs)
        self.isoton = IsotonicDensity()
        self.result = result

    def __call__(self, field):
        density = self.observer.density(field, result=self.result)
        isoton_density = self.isoton(density.detach())
        loss = torch.nansum((density - isoton_density)**2)
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
        return 'BorderZero'

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

    def __call__(self, density_profile) : 
        """
            density profile : (B, Z, *)  where * can be spatial (I, J) or even nothing
        Returns :
            (B, Z, *) non-decreasing along Z, closest in L2 to the input (columns with a NaN are NaN)
        Uses the min-max formula of the isotonic regression, x_i = max_{j<=i} min_{k>=i} mean(y_j..y_k), in O(Z^2) memory per column.
        """ 
        y = density_profile.detach().movedim(1, -1)                                       # (..., Z)
        z = y.shape[-1]
        c = torch.nn.functional.pad(y.cumsum(-1), (1, 0))                                 # (..., Z+1)
        j, k = torch.meshgrid(torch.arange(z, device=y.device), torch.arange(z, device=y.device), indexing='ij')
        means = (c[..., k + 1] - c[..., j]) / (k - j + 1).clamp(min=1)                     # (..., Z, Z) mean(y_j..y_k)
        means = torch.where(k >= j, means, torch.inf)
        # min_{k' >= k} mean(y_j..y_k') as a loop : cummin is not implemented on MPS
        upper = [means[..., -1]]
        for kk in range(z - 2, -1, -1) :
            upper.insert(0, torch.minimum(means[..., kk], upper[0]))
        upper = torch.stack(upper, dim=-1)
        lower = torch.where(k <= j, upper.transpose(-2, -1), -torch.inf)                  # [i, j] = min_{k >= i} mean(y_j..y_k), j <= i
        return lower.max(-1).values.movedim(-1, 1)
    

    