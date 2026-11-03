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
MINIMAL_CALENDAR = ['kt', 'ndastp', 'adatrj', 'ntime']
STEPS_PER_DAY = 120                        # rn_Dt = 720 s of the 1/4° runs
YEAR_DAYS, MONTH_DAYS = 360, 30


def start_calendar(kt_template, start_day):
    """Calendar of a restart whose run starts on forcing day of year `start_day` (1 to YEAR_DAYS - 1), in the forcing
    year of the template time step `kt_template`.

    The DINO seasonal forcing (usrdef_sbc.F90, compute_day_of_year) takes its phase from the time step only,
    (kt / STEPS_PER_DAY) mod YEAR_DAYS, so the date ndastp alone does not move it. Returns {kt, time_counter, adatrj,
    ndastp}: kt = time_counter = start of the template's forcing year + STEPS_PER_DAY * start_day, adatrj = kt /
    STEPS_PER_DAY days, ndastp = YYYYMMDD (3-digit year kt // year steps) of day `start_day` in 30-day months.
    """
    if not 1 <= start_day < YEAR_DAYS:
        raise ValueError(f'start_day {start_day} not in [1, {YEAR_DAYS})')
    year_steps = YEAR_DAYS * STEPS_PER_DAY
    kt = int(kt_template) // year_steps * year_steps + STEPS_PER_DAY * start_day
    month, day = (start_day - 1) // MONTH_DAYS + 1, (start_day - 1) % MONTH_DAYS + 1
    return dict(kt=kt, time_counter=kt, adatrj=kt / STEPS_PER_DAY, ndastp=kt // year_steps * 10000 + month * 100 + day)


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


def write_restart_1deg(toce, soce, ssh, exp_1deg, path_save):
    """Write `toce`, `soce` (z, y, x) and `ssh` (1, y, x) as tn, sn, sshn in a copy of the restart of `exp_1deg`, saved at `path_save`.

    The other variables keep the values of the reference run: the file only carries the generated state to the regridding
    (`write_restart_1_4deg`), which does not write them. Returns the restart dataset.
    """
    restart = exp_1deg.open_restart().load()
    restart['tn'] = (('time_counter', 'nav_lev', 'y', 'x'), np.expand_dims(toce, axis=0))
    restart['sn'] = (('time_counter', 'nav_lev', 'y', 'x'), np.expand_dims(soce, axis=0))
    restart['sshn'] = (('time_counter', 'y', 'x'), ssh)

    restart.attrs['file_name'] = Path(path_save).name
    restart.attrs['TimeStamp'] = datetime.now().astimezone().strftime('%d/%m/%Y %H:%M:%S %z')
    Path(path_save).unlink(missing_ok=True)
    restart.to_netcdf(path_save, unlimited_dims='time_counter')
    print(f'Saved {path_save}')
    return restart


def write_restart_1_4deg(path_1deg, exp_1deg, exp_1_4deg, path_save, start_day=None):
    """Regrid tn, sn, sshn of the 1° restart `path_1deg` to the grid of `exp_1_4deg` and write the minimal restart `path_save`.

    The restart holds only tn, sn, sshn (regridded), un, vn (zero), the calendar MINIMAL_CALENDAR and the coordinates of the
    restart of `exp_1_4deg`, with its dtypes: NEMO initialises everything else, which requires ln_1st_euler = .true.
    (see restart_initialisation.md). `start_day`, when given, replaces kt, time_counter, adatrj and ndastp of the
    template by `start_calendar` (dtypes kept, ntime unchanged), so that the run starts on that forcing day of year.
    Returns the restart dataset, its `time_counter` is `nn_it000 - 1` of the run.
    """
    restart_interpolated = GridManipulation(exp_1deg).regrid_restart(exp_1_4deg, restart_path=path_1deg)

    template = exp_1_4deg.open_restart().load()
    restart = template.drop_vars(set(template.data_vars) - set(MINIMAL_CALENDAR))
    for field in ['tn', 'sn', 'un', 'vn']:
        restart[field] = (('time_counter', 'nav_lev', 'y', 'x'), restart_interpolated[field].data.astype(template[field].dtype))
    restart['sshn'] = (('time_counter', 'y', 'x'), restart_interpolated['sshn'].data.astype(template['sshn'].dtype))
    if start_day is not None:
        calendar = start_calendar(template['kt'].values.squeeze(), start_day)
        for k in ['kt', 'adatrj', 'ndastp']:
            restart[k] = restart[k].copy(data=np.asarray(calendar[k], dtype=template[k].dtype))
        restart = restart.assign_coords(time_counter=restart.time_counter.copy(
            data=np.full(restart.sizes['time_counter'], calendar['time_counter'], dtype=template.time_counter.dtype)))

    restart = restart.fillna(0.)
    restart = restart.chunk({'time_counter': 1, 'y': 797, 'x': 242, 'nav_lev': 36}).compute()

    restart.attrs['file_name'] = Path(path_save).name
    restart.attrs['TimeStamp'] = datetime.now().astimezone().strftime('%d/%m/%Y %H:%M:%S %z')
    Path(path_save).unlink(missing_ok=True)
    restart.to_netcdf(path_save, unlimited_dims='time_counter')
    print(f'Saved {path_save} \n     Time counter (nn_it000 - 1) : {restart.time_counter.data[0]}, kt : {restart.kt.values}, '
          f'adatrj : {restart.adatrj.values}, ndastp : {restart.ndastp.values}')
    return restart


def convert(generation_path, path_1deg, path_1_4deg, idx_batch=0, base_references=BASE_REFERENCES, references=REFERENCES,
            start_day=None):
    """Member `idx_batch` of the npy in `generation_path` -> 1° restart `path_1deg` -> 1/4° restart `path_1_4deg`.

    `start_day`, when given, sets the calendar of the 1/4° restart to that forcing day of year (`start_calendar`), the
    template's otherwise.
    Returns the 1/4° restart dataset, its `time_counter` is `nn_it000 - 1` of the run.
    """
    toce, soce, ssh = load_member(generation_path, idx_batch)
    print(f'{soce.shape=} {toce.shape=} {ssh.shape=}')
    exp_1deg, exp_1_4deg = open_references(base_references, references)
    write_restart_1deg(toce, soce, ssh, exp_1deg, path_1deg)
    return write_restart_1_4deg(path_1deg, exp_1deg, exp_1_4deg, path_1_4deg, start_day=start_day)


if __name__ == '__main__':
    base_path_generated = "/lustre/fswork/projects/rech/omr/romr004/etienne/diffusion_states/"
    generation_name = 'C2'
    idx_batch = 4 # which generated field to run the simulation on

    path_1deg = f'{base_path_generated}/generated_restart_{generation_name}_1deg.nc'
    path_1_4deg = f'{base_path_generated}/generated_restart_{generation_name}_1_4deg.nc'
    convert(f'{base_path_generated}/{generation_name}', path_1deg, path_1_4deg, idx_batch=idx_batch)

    _, exp_1_4deg = open_references()
    generated = exp_1_4deg.open_restart(restart_path=path_1_4deg)
    generated.tn.isel(nav_lev=0).plot()
