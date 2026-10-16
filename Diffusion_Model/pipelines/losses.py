"""
Constraint losses on normalised states (b, c, i, j) -> (b,), one value per sample, used by `sampling.sample`, and the
projections they rely on.
"""
from __init__ import PRP; import sys


from pipelines.State import Observer
import torch
import numpy as np


class Loss :
    """
        observer (Observer) : observation operators and grid metrics
        dims : spatial dimensions
        volume_weighted (bool) : weight the terms of the loss by their share of the ocean volume
    """
    def __init__(self, observer : Observer, dims=(-2,-1), volume_weighted=False) :
        self.observer = observer
        self.dims = dims
        self.volume_weighted = volume_weighted

    def __call__(self, field : torch.tensor) -> torch.tensor :
        """
            Takes the normalised field (b, c, i, j) and returns the loss of each sample (b,)
        """
        pass


class Projection() :
    def __call__(self, field : torch.tensor)-> torch.tensor :
        """
            Takes the normalized field and return
            a projection of it
        """
        return field


## --- Losses ----

class ZeroMean(Loss) :
    """
        Zero spatial mean of each normalised level. With volume_weighted, the levels are weighted by their share of the
        volume of their field (`Observer.level_volume`), each field (toce, soce, ssh) with a total weight of 1.
    """
    def __init__(self, observer=None, **kwargs) :
        super().__init__(observer=observer, **kwargs)

    def __call__(self, field):
        mf = torch.nanmean(field, axis=(-2, -1))
        if not self.volume_weighted :
            return torch.nansum(mf**2, axis=1)
        lens = list(self.observer.normalizer.concatener.concat_lens.values())
        volume = self.observer.level_volume('strided')
        level_w = torch.cat([volume / volume.sum() if n > 1 else torch.ones(1, device=volume.device) for n in lens])
        return torch.nansum(level_w * mf**2, axis=1)


class NegativeGradDensity(Loss) :
    def __init__(self, **kwargs) :
        super().__init__(**kwargs)

    def __call__(self, field):
        grad_density = self.observer.density_gradient(field)
        grad_density[:, -1] = 0.0 # Bottom layer with no nans
        if self.volume_weighted :
            w = self.observer.volume_w / self.observer.volume_w.sum()
            return torch.nanmean(w * grad_density.clamp(0.0)**2, axis=(1, 2, 3))
        return torch.nanmean((grad_density.clamp(0.0))**2, axis=(1, 2, 3))
 

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
        w = 1.0
        if self.volume_weighted :
            w = self.observer.level_volume(self.result)
            w = w / w.sum()
        return torch.nansum(w * (self.mean_density[None, :] - density_profile)**2, axis=1)


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
        w = 1.0
        if self.volume_weighted :
            w = self.observer.volume_w.sum(dim=(-2, -1))
            w = w / w.sum()
        return torch.nansum(w * (self.grad_density[None, :] - grad_density_profile)**2, axis=1)


class DistancetoIsotonic(Loss) :
    """
        Squared distance of the density columns to the isotonic cone (density non-decreasing with depth).
        The projection is detached : the gradient of the squared distance to a convex set is 2 (x - P(x)).
        With volume_weighted the projection stays unweighted : 2 W (x - P(x)) is then not the exact gradient of the loss,
        but still a descent direction of the loss and of the distance to the cone.
    """
    def __init__(self, result='strided', **kwargs) :
        super().__init__(**kwargs)
        self.isoton = IsotonicDensity()
        self.result = result

    def __call__(self, field):
        density = self.observer.density(field, result=self.result)
        isoton_density = self.isoton(density.detach())
        w = 1.0
        if self.volume_weighted :
            w = self.observer.cell_volume(self.result)
            w = w / w.sum()
        return torch.nanmean(w * (density - isoton_density)**2, axis=tuple(range(1, density.ndim)))


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


class OptimProjection(Projection) :
    """
        Projection without closed form : minimise `loss` from the field with LBFGS until every sample has loss < eps.
    """
    def __init__(self, loss : Loss, eps=0.1, max_opt_steps = 10):
        self.loss = loss
        self.eps = eps
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
                loss = self.loss(field_o).sum()
                loss.backward()
                return loss
            for i in range(self.max_opt_steps) :
                if (self.loss(field_o) < self.eps).all() :
                    break
                opt.step(closure)
            field_o.requires_grad_(False)
        return field_o


class IsotonicDensity(Projection) :
    """
        Project the density columns on the isotonic cone (density non-decreasing with depth)
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
        # Densities ~1e3 : in float32 the cumsum below rounds at ~1e-3, far above real inversions. Isotonic regression
        # commutes with a shift, so work on the anomaly to the surface value.
        offset = y[..., :1]
        y = y - offset
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
        return (lower.max(-1).values + offset).movedim(-1, 1)


LOSSES = ['mean_density', 'mean_grad_density', 'negative_grad_density', 'zero_mean', 'dist_isoton']

# Default eps per loss, approx max of the loss on data, indexed as eps_data[volume_weighted][name]
eps_data = {False : {'mean_density' : 0.02,
                     'mean_grad_density' : 0.002,
                     'negative_grad_density' : 3e-10,
                     'zero_mean' : 0.01,
                     'dist_isoton' : 2e-10},
            True : {'mean_density' : None,
                    'mean_grad_density' : None,
                    'negative_grad_density' : None,
                    'zero_mean' : None,
                    'dist_isoton' : None}}

def get_loss(name, observer, volume_weighted=False) :
    """
        Per-sample loss `name` (one of LOSSES) on the normalised state, (b, c, i, j) -> (b,).
        volume_weighted : see `Loss`
    """
    if name == 'mean_density' :
        return MeanDensityProfile(observer=observer, result='strided', volume_weighted=volume_weighted,
                                  profile=np.loadtxt(f'{PRP}/Results/analysis_scripts/mean_density_train_strided.txt'))
    elif name == 'mean_grad_density' :
        return MeanGradDensityProfile(observer=observer, volume_weighted=volume_weighted)
    elif name == 'negative_grad_density' :
        return NegativeGradDensity(observer=observer, volume_weighted=volume_weighted)
    elif name == 'zero_mean' :
        return ZeroMean(observer=observer, volume_weighted=volume_weighted)
    elif name == 'dist_isoton' :
        return DistancetoIsotonic(observer=observer, volume_weighted=volume_weighted)
    raise ValueError(f'Unknown loss {name}')
