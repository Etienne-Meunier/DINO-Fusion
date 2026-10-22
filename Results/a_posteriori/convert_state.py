# %% [markdown]
# # Convert numpy field to Nemo Restart file
#
# Step 1 : convert numpy file to 1° simulation -> `generated_restart_<generation_name>_1deg.nc`
#
# Step 2 : upsample 1° simulation to 1/4° simulation
#
# Step 3 : create new 1/4° restart from upsampled prediction -> `generated_restart_<generation_name>_1_4deg.nc`

# %%
from __init__ import PRP; import sys
sys.path.append(f'{PRP}/Results/a_posteriori/dinostics/')

import os
import numpy as np
from datetime import datetime
from pathlib import Path
from experiment import NemoExperiment
import grid_manipulation
from grid_manipulation import GridManipulation

# A dinostics checkout with the xesmf import commented out (to work without xesmf) cannot regrid: provide it here.
if not hasattr(grid_manipulation, 'Regridder'):
    from xesmf import Regridder
    grid_manipulation.Regridder = Regridder

# %%
BASE_REFERENCES = f'{os.environ["OCEANDATA"]}/Dino-Fusion-Data/Restart-format/'
REFERENCES = {'1deg' : '1deg_restart', '1_4deg' : 'restart25_arch'}


def load_member(generation_path, idx_batch=0):
    """Load member `idx_batch` of `<generation_path>/{toce,soce,ssh}.npy`, shapes (b, z, y, x) and (b, 1, y, x).

    Returns toce, soce (z, y, x) and ssh (1, y, x).
    """
    toce = np.load(f'{generation_path}/toce.npy')[idx_batch, :, :, :]
    soce = np.load(f'{generation_path}/soce.npy')[idx_batch, :, :, :]
    ssh = np.load(f'{generation_path}/ssh.npy')[idx_batch, :, :]
    return toce, soce, ssh


def open_references(base_references=BASE_REFERENCES, references=REFERENCES):
    """Open the reference 1° and 1/4° experiments, used only for their grid and restart template. Returns (exp_1deg, exp_1_4deg)."""
    exp_1deg = NemoExperiment(path=base_references, experiment_name=references['1deg'])
    exp_1_4deg = NemoExperiment(path=base_references, experiment_name=references['1_4deg'])
    return exp_1deg, exp_1_4deg


def potential_density(toce, soce, z, nml):
    """Potential density (kg/m3) of `toce`, `soce` (z, y, x) at depths `z` with the NEMO EOS coefficients of namelist block `nml` (nameos)."""
    return (
        - nml['rn_a0'] * (1. + 0.5 * nml['rn_lambda1'] * (toce - 10.) + nml['rn_mu1'] * z) * (toce - 10.)
        + nml['rn_b0'] * (1. - 0.5 * nml['rn_lambda2'] * (soce - 35.) - nml['rn_mu2'] * z) * (soce - 35.)
        - nml['rn_nu'] * (toce - 10.) * (soce - 35.)
    ) + 1026


def write_restart_1deg(toce, soce, ssh, exp_1deg, path_save):
    """Write `toce`, `soce` (z, y, x), `ssh` (1, y, x) and their rhop in a copy of the restart of `exp_1deg`, saved at `path_save`.

    Returns the restart dataset.
    """
    restart = exp_1deg.open_restart().load()
    rhop = potential_density(toce, soce, exp_1deg.mask.gdept_0.data, exp_1deg.namelist['nameos'])

    for fields, value in [(('tb', 'tn'), toce), (('sb', 'sn'), soce), (('rhop',), rhop)]:
        for field in fields:
            restart[field] = (('time_counter', 'nav_lev', 'y', 'x'), np.expand_dims(value, axis=0))
    for field in ['sshb', 'sshn']:
        restart[field] = (('time_counter', 'y', 'x'), ssh)

    restart.attrs['file_name'] = Path(path_save).name
    restart.attrs['TimeStamp'] = datetime.now().astimezone().strftime('%d/%m/%Y %H:%M:%S %z')
    Path(path_save).unlink(missing_ok=True)
    restart.to_netcdf(path_save, unlimited_dims='time_counter')
    print(f'Saved {path_save}')
    return restart


def write_restart_1_4deg(path_1deg, exp_1deg, exp_1_4deg, path_save):
    """Regrid the 1° restart `path_1deg` to the grid of `exp_1_4deg` and write it in a copy of its restart, saved at `path_save`.

    Velocities are set to zero by the regridding. Returns the restart dataset, its `time_counter` is `nn_it000 - 1` of the run.
    """
    restart_interpolated = GridManipulation(exp_1deg).regrid_restart(exp_1_4deg, restart_path=path_1deg)

    restart = exp_1_4deg.open_restart().load().copy()
    for field in ['tn', 'tb', 'sn', 'sb', 'un', 'ub', 'vn', 'vb', 'rhop']:
        restart[field] = (('time_counter', 'nav_lev', 'y', 'x'), restart_interpolated[field].data.astype(restart[field].dtype))
    for field in ['sshn', 'sshb']:
        restart[field] = (('time_counter', 'y', 'x'), restart_interpolated[field].data.astype(restart[field].dtype))

    restart = restart.fillna(0.)
    restart = restart.chunk({'time_counter': 1, 'y': 797, 'x': 242, 'nav_lev': 36}).compute()

    restart.attrs['file_name'] = Path(path_save).name
    restart.attrs['TimeStamp'] = datetime.now().astimezone().strftime('%d/%m/%Y %H:%M:%S %z')
    Path(path_save).unlink(missing_ok=True)
    restart.to_netcdf(path_save, unlimited_dims='time_counter')
    print(f'Saved {path_save} \n     Time counter (nn_it000 - 1) : {restart.time_counter.data[0]}')
    return restart


def convert(generation_path, path_1deg, path_1_4deg, idx_batch=0, base_references=BASE_REFERENCES, references=REFERENCES):
    """Member `idx_batch` of the npy in `generation_path` -> 1° restart `path_1deg` -> 1/4° restart `path_1_4deg`.

    Returns the 1/4° restart dataset, its `time_counter` is `nn_it000 - 1` of the run.
    """
    toce, soce, ssh = load_member(generation_path, idx_batch)
    print(f'{soce.shape=} {toce.shape=} {ssh.shape=}')
    exp_1deg, exp_1_4deg = open_references(base_references, references)
    write_restart_1deg(toce, soce, ssh, exp_1deg, path_1deg)
    return write_restart_1_4deg(path_1deg, exp_1deg, exp_1_4deg, path_1_4deg)


if __name__ == '__main__':
    base_path_generated = "/lustre/fsn1/projects/rech/omr/romr004/data/diffusion_states/"
    generation_name = 'C2'
    idx_batch = 4 # which generated field to run the simulation on

    path_1deg = f'{base_path_generated}/generated_restart_{generation_name}_1deg.nc'
    path_1_4deg = f'{base_path_generated}/generated_restart_{generation_name}_1_4deg.nc'
    convert(f'{base_path_generated}/{generation_name}', path_1deg, path_1_4deg, idx_batch=idx_batch)

    _, exp_1_4deg = open_references()
    generated = exp_1_4deg.open_restart(restart_path=path_1_4deg)
    generated.rhop.isel(nav_lev=0).plot(vmin=1022, vmax=1028)
