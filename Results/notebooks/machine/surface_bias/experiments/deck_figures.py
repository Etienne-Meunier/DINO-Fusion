"""
Figures of the surface-bias presentation, saved as figures/deck_*.png on the deck background (#f7f6f2).

    deck_traj      T0 of x0_hat along the generation (runs/traj_*.npz) : BorderZero, BorderZero + centred noise, new sampler
    deck_tradeoff  density error E against the level-0 sigma_0 bias to the data, per method and strength (b8 seed 0)
    deck_push      batch-mean level-mean T minus the data batch (b32)
    deck_maps      batch-mean level-0 T minus the data mean (b32)
    deck_sections  batch-mean zonal sigma_0 sections with the 65-45S band profile as inset (paper fig. 1 style), 2 x 4 sources
    deck_profiles  65-45S band sigma_0 profiles of the same sources on one axis, two panels (repo sampler, centred sampler)
    deck_band      35-60N band : surface T against stratification T0 - T7, members over the 36-phase climatology (b32)
The (E, bias) points are cached in figures/deck_tradeoff.npz.
"""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'paper'))
from common import *
from metrics import compute_density_error

BG, INK, MUTED, GRID = '#f7f6f2', '#1d2430', '#5b6270', '#dcdad3'
BLUE, ORANGE, AQUA, VIOLET, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#6250d6', '#c98500'
plt.rcParams.update({'font.size': 15, 'axes.titlesize': 16, 'axes.labelsize': 15, 'xtick.labelsize': 13, 'ytick.labelsize': 13,
                     'legend.fontsize': 13, 'figure.facecolor': BG, 'axes.facecolor': BG, 'savefig.facecolor': BG,
                     'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'text.color': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
                     'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False})
INF = f'{os.environ["OCEANDATA"]}/models/dino-fusion/tav0h83b/inference'
SWEEP = f'{INF}/sweep_density_huber/negative_grad_density_huber_border0'
NONE_B8 = f'{INF}/sweep_baselines/none_border0_b8/151026-145216/generated.npy'
OUT = HERE / 'figures'
mask = load_mask()
ocean = mask.tmask.values.astype(bool)
volume = (mask.e1t * mask.e2t * mask.e3t_0 * mask.tmask).transpose('depth', 'nav_lat', 'nav_lon').values
depth = mask.gdept_1d.values
lat = mask.gphit.isel(nav_lon=10).values


def run(name) :
    return np.load(HERE / 'runs' / name / 'generated.npy')


def fields(x) :
    return {k : v.detach().cpu().numpy() for k, v in observer.unnormalized(torch.as_tensor(x)).items()}


def E_bias(x) :
    """Mean density error E and mean volume-weighted level-0 sigma_0 of a normalised batch."""
    f = fields(x)
    sig = compute_density(f, mask).transpose('batch', 'depth', 'nav_lat', 'nav_lon').values[:, 0] - 1000
    lev0 = np.nansum(np.where(ocean[0], sig, 0) * volume[0], axis=(1, 2)) / volume[0].sum()
    return float(np.mean(compute_density_error(f, mask))), float(lev0.mean())


def level_mean_T(x) :
    """Volume-weighted level means of T (b, z)."""
    T = np.nan_to_num(fields(x)['toce'])
    return (T * volume).sum((-2, -1)) / volume.sum((-2, -1))


def traj() :
    """deck_traj : T0 of x0_hat against the step, one panel per sampler, data T0 range shaded."""
    panels = [('border0', 'BorderZero (repo)'), ('border0_call', 'BorderZero + centred noise'), ('bnoise_call', 'New sampler')]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for ax, (tag, title) in zip(axes, panels) :
        d = np.load(HERE / 'runs' / f'traj_{tag}.npz')
        ax.axhspan(19.80, 20.18, color=AQUA, alpha=0.18, linewidth=0, label='data range')
        ax.plot(d['t'], d['T0'], color=BLUE, linewidth=1.4, alpha=0.8)
        ax.set_xlim(1000, 0)
        ax.set_ylim(12, 28)
        ax.set_title(f'{title}\nfinal T0 {d["final"].mean():.2f} ± {d["final"].std():.2f} °C', loc='left')
        ax.set_xlabel('diffusion step t')
    axes[0].set_ylabel('level-0 T of x0_hat [°C]')
    axes[0].legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(OUT / 'deck_traj.png', dpi=150)


def tradeoff() :
    """deck_tradeoff : E against the level-0 sigma_0 bias, one line per method, strength labels at the points."""
    cache = OUT / 'deck_tradeoff.npz'
    if cache.exists() :
        res = {k : v for k, v in np.load(cache, allow_pickle=True)['res'].item().items()}
    else :
        old = {0. : np.load(NONE_B8)}
        for d in glob(f'{SWEEP}/*/generated.npy') :
            old[json.load(open(f'{os.path.dirname(d)}/args.json'))['eta']] = np.load(d)
        series = {'data' : {0 : batch.numpy()},
                  'repo sampler, C7' : dict(sorted(old.items())),
                  'new sampler, C7' : {0 : run('centred_none')} | {e : run(f'centred_c7{e:03d}') for e in [10, 30, 50, 100]},
                  'new sampler, C7 + surface T,S mean' : {e : run(f'centred_surf_c7{e:03d}') for e in [10, 30, 100]},
                  'new sampler, zero_mean' : {e : run(f'centred_zm_e{e:g}') for e in [0.03, 0.1, 0.3, 1, 3]},
                  'new sampler, zero_mean + C7 10' : {e : run(f'centred_zm_e{e:g}_c7010') for e in [0.1, 1]}}
        res = {k : {e : E_bias(x) for e, x in v.items()} for k, v in series.items()}
        np.savez(cache, res=np.array(res, dtype=object))
    sig_data = res['data'][0][1]
    style = {'repo sampler, C7' : (ORANGE, 'o', 'C7 η '), 'new sampler, C7' : (BLUE, 'o', 'C7 η '),
             'new sampler, C7 + surface T,S mean' : (VIOLET, 's', 'C7 η '),
             'new sampler, zero_mean' : (AQUA, 'D', ''), 'new sampler, zero_mean + C7 10' : (YELLOW, '^', '')}
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.axvline(0, color=MUTED, linewidth=1, linestyle=':')
    for k, (c, m, prefix) in style.items() :
        pts = sorted(res[k].items())
        b = [v[1] - sig_data for _, v in pts]
        E = [v[0] for _, v in pts]
        ax.plot(b, E, f'{m}-', color=c, markersize=9, linewidth=2, label=k, markeredgecolor=BG, markeredgewidth=1.5)
        if k in ('repo sampler, C7', 'new sampler, C7') :
            for (e, _), bi, Ei in zip(pts, b, E) :
                if e in (0, 10, 30, 100, 300) :
                    ax.annotate('none' if e == 0 else f'{e:g}', (bi, Ei), textcoords='offset points', xytext=(8, 6),
                                fontsize=12, color=MUTED)
    ax.plot([0], [res['data'][0][0]], '*', color=INK, markersize=16, label='data')
    ax.set_xlabel('surface σ0 bias to the data [kg m⁻³]   (< 0 : surface too light)')
    ax.set_ylabel('density error E [% volume]')
    ax.set_yscale('log')
    ax.set_yticks([0.3, 0.5, 1, 2, 5, 10])
    ax.set_yticklabels(['0.3', '0.5', '1', '2', '5', '10'])
    ax.minorticks_off()
    ax.set_xlim(-2.6, 0.9)
    ax.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig(OUT / 'deck_tradeoff.png', dpi=150)


def push() :
    """deck_push : batch-mean level-mean T minus the data batch against depth."""
    ref = level_mean_T(batch.numpy()).mean(0)
    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.axvline(0, color=MUTED, linewidth=1, linestyle=':')
    for name, label, c in [('centred_none_b32', 'unconstrained', INK), ('centred_c7010_b32', 'C7 10', BLUE), ('centred_surf_c7030_b32', 'C7 30 + surface T,S', VIOLET),
                           ('centred_zm_e1_c7010_b32', 'zero_mean 1 + C7 10', YELLOW)] :
        dT = level_mean_T(run(name)) - ref
        ax.plot(dT.mean(0), depth, 'o-', color=c, linewidth=2, markersize=6, label=label, markeredgecolor=BG)
    ax.set_yscale('symlog', linthresh=100)
    ax.set_ylim(4000, 0)
    ax.set_yticks([0, 50, 100, 300, 1000, 3000])
    ax.set_yticklabels(['0', '50', '100', '300', '1000', '3000'])
    ax.minorticks_off()
    ax.set_xlabel('level-mean T minus data [°C]')
    ax.set_ylabel('depth [m]')
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(OUT / 'deck_push.png', dpi=150)


def maps() :
    """deck_maps : batch-mean level-0 T minus the data-batch mean, three runs."""
    T0d = np.nanmean(fields(batch)['toce'][:, 0], 0)
    runs = [('centred_none_b32', 'New sampler, unconstrained'), ('centred_c7010_b32', 'New sampler, C7 10'),
            ('centred_zm_e1_c7010_b32', 'New sampler, zero_mean 1 + C7 10')]
    fig, axes = plt.subplots(1, 3, figsize=(13, 7.2), sharey=True)
    for ax, (name, title) in zip(axes, runs) :
        d = np.where(ocean[0], np.nanmean(fields(run(name))['toce'][:, 0], 0) - T0d, np.nan)
        im = ax.pcolormesh(np.arange(d.shape[1]), lat, d, cmap='RdBu_r', vmin=-3, vmax=3, shading='auto')
        ax.set_title(f'{title}\nmean {np.nansum(np.nan_to_num(d) * volume[0]) / volume[0].sum():+.2f} °C', loc='left')
        ax.set_xticks([])
        ax.grid(False)
        ax.set_facecolor('#c9c6bd')
    axes[0].set_ylabel('latitude [°N]')
    fig.colorbar(im, ax=axes, label='level-0 T minus data [°C]', fraction=0.03, pad=0.02)
    fig.savefig(OUT / 'deck_maps.png', dpi=150, bbox_inches='tight')


def band() :
    """deck_band : 35-60N band T0 against T0 - T7, the climatology loop coloured by phase, members of four sources."""
    K = 14
    area = (mask.e1t * mask.e2t * mask.tmask).transpose('depth', 'nav_lat', 'nav_lon').values[:K]
    T = np.load(OUT / 'seasonality_3d.npz')['T']
    ph = np.arange(len(T)) % 36
    clim = np.stack([T[ph == p].mean(0) for p in range(36)])
    sel = (lat > 35) & (lat < 60)
    feats = lambda Tz : (Tz[:, 0, sel].mean(-1), Tz[:, 0, sel].mean(-1) - Tz[:, 7, sel].mean(-1))

    def zonal(x) :
        t = np.nan_to_num(fields(x)['toce'][:, :K])
        return (t * area).sum(-1) / np.maximum(area.sum(-1), 1e-9)

    sources = [('Data (64 states)', T[np.random.default_rng(0).choice(len(T), 64, replace=False)], MUTED),
               ('Unconstrained', zonal(run('centred_none_b32')), INK),
               ('C7 10', zonal(run('centred_c7010_b32')), BLUE),
               ('zero_mean 1 + C7 10', zonal(run('centred_zm_e1_c7010_b32')), YELLOW)]
    f0, fs = feats(clim)
    fig, axes = plt.subplots(1, 4, figsize=(17, 5.2), sharex=True, sharey=True)
    for ax, (title, Tz, c) in zip(axes, sources) :
        loop = ax.scatter(f0, fs, c=np.arange(36), cmap='twilight', s=40, zorder=2)
        ax.plot(np.r_[f0, f0[:1]], np.r_[fs, fs[:1]], color=GRID, linewidth=1.5, zorder=1)
        a, s = feats(Tz)
        ax.scatter(a, s, color=c, s=46, alpha=0.85, edgecolor=BG, linewidth=1, zorder=3)
        ax.set_title(title, loc='left')
        ax.set_xlabel('band surface T [°C]')
    axes[0].set_ylabel('T(5 m) − T(87 m) [°C]')
    axes[0].set_xlim(13.4, 17.8)
    cb = fig.colorbar(loop, ax=axes, fraction=0.02, pad=0.01, ticks=[0, 9, 18, 27, 35])
    cb.set_label('phase of the year (36 = 1 yr)')
    fig.savefig(OUT / 'deck_band.png', dpi=150, bbox_inches='tight')


SOURCES = [('data', 'Data', MUTED), ('optimal:none', 'No projection (repo)', '#8a8f99'),
           ('optimal:none_border0', 'BorderZero (repo)', ORANGE), ('optimal:negative_grad_density_huber_border0', 'C7 10 (repo)', BLUE),
           ('centred_none_b32', 'Centred, no constraint', INK), ('centred_c7010_b32', 'Centred, C7 10', BLUE),
           ('centred_surf_c7030_b32', 'Centred, C7 30 + surface', VIOLET), ('centred_zm_e1_c7010_b32', 'Centred, zm + C7 10', YELLOW)]
BAND, LEVELS, DEPTH_POWER = (-65, -45), np.arange(22., 28.51, 0.5), 0.625
DEPTH_TICKS = [0, 100, 250, 500, 1000, 2000, 3000]


def load(name) :
    """Normalised batch of `data`, `optimal:<constraint>` (paper b16) or a runs/ folder."""
    if name == 'data' :
        return batch
    if name.startswith('optimal:') :
        return torch.from_numpy(np.load(f'{generation_dirs()[name[8:]]}/generated.npy'))
    return torch.from_numpy(run(name))


def mean_sections() :
    """Batch-mean zonal sigma_0 section (z, y) of every SOURCES entry, cached in figures/deck_sections.npz."""
    cache = OUT / 'deck_sections.npz'
    if cache.exists() :
        return dict(np.load(cache))
    out = {name : np.nanmean(zonal_sections(load(name), mask), 0) for name, _, _ in SOURCES}
    np.savez(cache, **out)
    return out


def band_profile(section) :
    return np.nanmean(section[:, (lat >= BAND[0]) & (lat < BAND[1])], axis=1)


def sections() :
    """deck_sections : 2 x 4 contour sections, band profile inset (black) over the data profile (dashed)."""
    import cmocean.cm as cm
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    sec = mean_sections()
    z = depth[:-1] ** DEPTH_POWER
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.4), sharex=True, sharey=True)
    for ax, (name, title, _) in zip(axes.flat, SOURCES) :
        ax.grid(False)
        cf = ax.contourf(lat, z, sec[name], levels=LEVELS, cmap=cm.deep_r, extend='both')
        ax.contour(lat, z, sec[name], levels=LEVELS, colors='k', linewidths=0.5)
        for b in BAND :
            ax.axvline(b, color='gray', linestyle='--', linewidth=1)
        ax.set_title(title, loc='left')
        ax.set_ylim(3500 ** DEPTH_POWER, 0)
        ax.set_yticks(np.asarray(DEPTH_TICKS) ** DEPTH_POWER)
        ax.set_yticklabels([str(d) for d in DEPTH_TICKS])
        ins = inset_axes(ax, width='18%', height='100%', loc='right', borderpad=0)
        ins.plot(band_profile(sec[name]), z, color='k', linewidth=2)
        ins.plot(band_profile(sec['data']), z, color='k', linewidth=1, linestyle='--')
        ins.set_facecolor('gray'); ins.patch.set_alpha(0.4)
        ins.set_xlim(25, 28.5); ins.set_ylim(ax.get_ylim())
        ins.set_xticklabels([]); ins.set_yticklabels([]); ins.tick_params(length=0)
    for ax in axes[1] :
        ax.set_xlabel('latitude [°N]')
    for ax in axes[:, 0] :
        ax.set_ylabel('depth [m]')
    cb = fig.colorbar(cf, ax=axes, fraction=0.02, pad=0.01, label='σ0 [kg m⁻³ − 1000]')
    cb.ax.invert_yaxis()
    fig.savefig(OUT / 'deck_sections.png', dpi=150, bbox_inches='tight')


def profiles() :
    """deck_profiles : band sigma_0 profiles, repo sources left, centred sources right, data dashed on both."""
    sec = mean_sections()
    fig, axes = plt.subplots(1, 2, figsize=(13, 7.5), sharey=True)
    for ax, group, title in [(axes[0], SOURCES[1:4], 'Repo sampler'), (axes[1], SOURCES[4:], 'Centred sampler')] :
        ax.plot(band_profile(sec['data']), depth[:-1], color=MUTED, linewidth=3, linestyle='--', label='Data')
        for name, label, c in group :
            ax.plot(band_profile(sec[name]), depth[:-1], color=c, linewidth=2.2, label=label.replace(' (repo)', '').replace('Centred, ', ''))
        ax.set_yscale('symlog', linthresh=100)
        ax.set_ylim(3500, 0)
        ax.set_yticks([0, 50, 100, 300, 1000, 3000]); ax.set_yticklabels(['0', '50', '100', '300', '1000', '3000'])
        ax.minorticks_off()
        ax.set_xlim(25.6, 28.0)
        ax.set_title(title, loc='left')
        ax.set_xlabel('σ0 [kg m⁻³ − 1000], 65–45°S')
        ax.legend(loc='lower left')
    axes[0].set_ylabel('depth [m]')
    fig.tight_layout()
    fig.savefig(OUT / 'deck_profiles.png', dpi=150)


for f in sys.argv[1:] or ['traj', 'tradeoff', 'push', 'maps', 'band', 'sections', 'profiles'] :
    globals()[f]()
