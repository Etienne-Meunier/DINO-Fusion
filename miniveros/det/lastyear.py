"""Last-year mean temperature of every run: the target of the deterministic emulator.

    python -m det.lastyear --raw-dir ../full_state/mnk965ig --out data/veros_acc_lastyear.npz

Reads only the last year of ``temp`` from each raw run (by seeking inside the zip, as extract_data.py does) and
writes one small npz: ``temp`` (R, Z, Y, X) float32, the mean of the 30-day outputs of the last model year
(12 outputs, 360 days; land = 0); ``run_ck``, ``run_eps``, ``run_names``; ``mask_land`` (Z, Y, X); ``zt``; ``n_snap``.
Salinity is 35 in every water cell of every run's last year, to within 2e-8 psu (exactly 35 in float32), so it is
not carried.
"""
from __future__ import annotations

import argparse
import glob
import os
import time as clock
import zipfile

import numpy as np

from extract_data import SECONDS_PER_YEAR, parse_run_name, read_small, read_time_block, to_zyx_interior


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", required=True, help="folder with ck*_eps*.npz")
    p.add_argument("--out", default="data/veros_acc_lastyear.npz")
    p.add_argument("--years", type=float, default=1.0, help="averaging window at the end of each run (years of 360 days)")
    a = p.parse_args(argv)

    files = sorted(glob.glob(os.path.join(a.raw_dir, "ck*_eps*.npz")))
    if not files:
        raise SystemExit(f"no run files in {a.raw_dir}")
    t0 = clock.time()
    temp, names, cks, epss, land, zt, n_snap = [], [], [], [], None, None, None
    for f in files:
        ck, eps = parse_run_name(f)
        with zipfile.ZipFile(f) as zf:
            t = read_small(zf, "time.npy")
            first = int(np.searchsorted(t, t[-1] - a.years * SECONDS_PER_YEAR, side="right"))
            blk = to_zyx_interior(read_time_block(zf, "temp.npy", first, len(t)))      # (n, Z, Y, X)
            if zt is None:
                zt, n_snap = read_small(zf, "zt.npy"), len(t) - first
            assert len(t) - first == n_snap, f"{f}: {len(t) - first} snapshots in the window, first run had {n_snap}"
        m = (blk == 0.0).all(0)
        assert np.array_equal(m, (blk == 0.0).any(0)), f"{f}: zero cells vary in time"
        land = m if land is None else land
        assert np.array_equal(land, m), f"{f}: land mask differs from the first run"
        temp.append(blk.astype(np.float64).mean(0).astype(np.float32))
        names.append(os.path.basename(f)[:-4]); cks.append(ck); epss.append(eps)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, temp=np.stack(temp), run_ck=np.array(cks), run_eps=np.array(epss), run_names=np.array(names),
             mask_land=land, zt=zt, n_snap=n_snap, window_years=a.years)
    print(f"wrote {a.out}: {len(files)} runs, mean of the last {n_snap} outputs, land cells {int(land.sum())}/{land.size}, "
          f"{os.path.getsize(a.out) / 1e6:.1f} MB, {clock.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
