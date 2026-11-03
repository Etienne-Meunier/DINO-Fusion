"""
Task 1 : gradient of NegativeGradDensityHuber through the loss alone (no UNet), on data and unconstrained generated states.

For the descent direction dx = -grad_x L (normalised state), reports per level :
    - descent on rho, T, S (file levels) : -dL/drho, -dL/dT, -dL/dS, horizontal mean over ocean cells (sum over cells / n_ocean)
    - descent on the normalised channels : -dL/dx per strided T / S channel, horizontal mean and DC fraction mean^2 / mean(x^2)
    - first-order change induced by the step dx on rho, T, S (jvp), horizontal ocean mean
All quantities are averaged over the states of a set, and are normalised by the total |dx| so sets can be compared.

Output : p1_levels_<set>.csv, p1_summary.txt
"""
from common import *
import pandas as pd
from metrics import get_density_at_surface_tensor


def rho_TS(x, obs) :
    """rho, T, S (b, 36, 199, 62) of the normalised state x, differentiable."""
    u = obs.unnormalized(x)
    T, S = u['toce'], u['soce']
    return get_density_at_surface_tensor(T, S), T, S


def huber(rho, loss) :
    """Same value as NegativeGradDensityHuber, from rho."""
    g = rho[:, 1:] - rho[:, :-1]
    g = torch.where(loss.valid, g, torch.zeros_like(g))
    h = torch.where(g < -loss.tau, -g - loss.tau / 2, g**2 / (2 * loss.tau))
    return (loss.w * torch.where(g < 0, h, torch.zeros_like(g))).sum(dim=(1, 2, 3))


def level_mean(f, tmask) :
    """Horizontal mean over ocean cells (b, k, i, j) -> (b, k), nan on land ignored."""
    f = torch.nan_to_num(f) * tmask
    return f.sum((-2, -1)) / tmask.sum((-2, -1)).clamp(min=1)


def analyse(x, obs, loss, grad_fn=None) :
    """
        Per-level diagnostics of the descent step for states x (b, 37, 208, 64).
        grad_fn (x -> dx) : descent direction, default -grad_x of the loss
    Returns : DataFrame per file level k (0..35) and per strided level, dict of scalars
    """
    tmask = obs.tmask
    x = x.clone().requires_grad_(True)
    rho, T, S = rho_TS(x, obs)
    for v in (rho, T, S) : v.retain_grad()
    L = huber(rho, loss)
    assert torch.allclose(L, loss(x), rtol=1e-4), (L, loss(x))
    L.sum().backward()
    gx = x.grad.detach()
    dx = -gx if grad_fn is None else grad_fn(x.detach(), gx)
    oc = ocean_mask_norm(obs)
    dx = dx * oc
    # Induced first-order change, forward mode
    _, (drho, dT, dS) = torch.autograd.functional.jvp(lambda z : rho_TS(z, obs), x.detach(), dx)
    scale = dx.pow(2).sum((1, 2, 3)).sqrt().view(-1, 1)          # per-state |dx|
    lm = lambda f : (level_mean(f, tmask) / scale).mean(0).numpy()
    rows = pd.DataFrame({'k' : np.arange(36),
                         'desc_rho' : lm(-rho.grad), 'desc_T' : lm(-T.grad), 'desc_S' : lm(-S.grad),
                         'ind_rho' : lm(drho), 'ind_T' : lm(dT), 'ind_S' : lm(dS)})
    occ = oc[None].float()
    m = (dx.sum((-2, -1)) / occ.sum((-2, -1))) / scale
    ms = ((dx**2).sum((-2, -1)) / occ.sum((-2, -1))) / scale**2
    strided = pd.DataFrame({'ks' : np.arange(NT),
                            'dxT_mean' : m[:, :NT].mean(0).numpy(), 'dxS_mean' : m[:, NT:2 * NT].mean(0).numpy(),
                            'dxT_DCfrac' : (m[:, :NT]**2 / ms[:, :NT]).mean(0).numpy(),
                            'dxS_DCfrac' : (m[:, NT:2 * NT]**2 / ms[:, NT:2 * NT]).mean(0).numpy(),
                            'dxT_energy' : ms[:, :NT].mean(0).numpy(), 'dxS_energy' : ms[:, NT:2 * NT].mean(0).numpy()})
    vol = obs.volume
    mass = ((torch.nan_to_num(drho) * vol).sum((1, 2, 3)) / scale[:, 0])
    # share of the induced rho change (volume-weighted, signed) by depth band
    vrho = (torch.nan_to_num(drho) * vol).sum((-2, -1)) / scale
    scal = dict(L=L.mean().item(), mass_change=mass.mean().item(),
                vrho_lev0=vrho[:, 0].mean().item(), vrho_lev1_5=vrho[:, 1:6].sum(1).mean().item(),
                vrho_lev6_34=vrho[:, 6:35].sum(1).mean().item(), vrho_lev35=vrho[:, 35].mean().item(),
                active_frac_lev0=((rho[:, 1] - rho[:, 0] < 0) & loss.valid[0]).float().sum((-2, -1)).mean().item() / loss.valid[0].sum().item())
    return rows, strided, scal


if __name__ == '__main__' :
    torch.set_grad_enabled(True)
    obs = get_observer()
    loss = NegativeGradDensityHuber(observer=obs)
    sets = {'DATA' : load_data(16), 'NONE' : load_none()}
    out = open(HERE / 'p1_summary.txt', 'w')
    pd.set_option('display.width', 250); pd.set_option('display.precision', 3)
    for name, x in sets.items() :
        rows, strided, scal = analyse(x, obs, loss)
        rows.to_csv(HERE / f'p1_levels_{name}.csv', index=False)
        strided.to_csv(HERE / f'p1_strided_{name}.csv', index=False)
        txt = f'=== {name} ===\n{scal}\n{rows.to_string()}\n{strided.to_string()}\n'
        print(txt); out.write(txt)
