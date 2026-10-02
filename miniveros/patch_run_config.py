"""One-off retrofit: append the mini-veros run configuration to an existing dataset written by extract_data.py.

The raw sweep files (mnk965ig) carry only c_k / c_eps, not the configuration they were run with. This rebuilds that
configuration with the current mini-veros (checked to reproduce mnk965ig to round-off over one model-year from a cold
start, see the ``verification`` field) and appends to the dataset npz, without rewriting it:

    run_config_common    JSON: setup, flat StaticConfig / Parameters (dotted keys, as accepted by Overridable.override)
                         minus the varying keys, run length, code version, provenance
    config_varying_keys  JSON: dotted parameter key -> dataset array holding its per-run value
    run_ck_exact         (n_runs,) float64, c_k actually run (read from the raw npz, not parsed from the file name)
    run_eps_exact        (n_runs,) float64, c_eps actually run

``run_ck`` / ``run_eps`` are left untouched: they are the (file-name rounded) values the model was conditioned on.

    python patch_run_config.py --data-file .../veros_acc_TS.npz --raw-dir .../full_state/mnk965ig \\
        --mini-veros-repo .../MiniVeros-Autodiff
"""
from __future__ import annotations

import argparse
import dataclasses
import io
import json
import os
import subprocess
import sys
import zipfile

import numpy as np
from numpy.lib import format as npf

NEW_KEYS = ("run_config_common", "config_varying_keys", "run_ck_exact", "run_eps_exact")
VARYING = {"tke_closure.c_k": "run_ck_exact", "tke_closure.c_eps": "run_eps_exact"}


def flat_dump(obj, prefix: str = "") -> dict:
    """Flatten a mini-veros dataclass into {dotted field: python scalar}; sub-dataclasses recurse and record their
    class under ``<field>.__class__`` (override can change their fields, not their type)."""
    out = {}
    for f in dataclasses.fields(obj):
        v = getattr(obj, f.name)
        key = prefix + f.name
        if dataclasses.is_dataclass(v):
            out[key + ".__class__"] = type(v).__name__
            out.update(flat_dump(v, key + "."))
        else:
            a = np.asarray(v)
            out[key] = a.item() if a.ndim == 0 else a.tolist()
    return out


def as_overrides(flat: dict) -> dict:
    return {k: v for k, v in flat.items() if not k.endswith(".__class__")}


def git(repo: str, *args: str) -> str:
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, check=True).stdout.strip()


def read_scalar(zf: zipfile.ZipFile, name: str) -> float:
    return float(np.load(io.BytesIO(zf.read(name + ".npy"))))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-file", required=True)
    p.add_argument("--raw-dir", required=True)
    p.add_argument("--mini-veros-repo", required=True, help="MiniVeros-Autodiff checkout (parent of mini-veros/)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    sys.path.insert(0, os.path.join(args.mini_veros_repo, "test", "ck_ceps_100y_sweep"))
    import common                                   # the sweep's own module: same build call, same run length
    from mini_veros.setups.acc import full

    with np.load(args.data_file, allow_pickle=False) as ds:
        present = set(ds.files)
        run_names = [str(r) for r in ds["run_names"]]
        run_ck, run_eps = np.asarray(ds["run_ck"]), np.asarray(ds["run_eps"])
    clash = present.intersection(NEW_KEYS)
    if clash:
        raise SystemExit(f"{args.data_file} already has {sorted(clash)}")

    # ---- exact per-run values from the raw files
    ck_exact, eps_exact, raw_meta = np.empty(len(run_names)), np.empty(len(run_names)), set()
    for i, name in enumerate(run_names):
        with zipfile.ZipFile(os.path.join(args.raw_dir, name + ".npz")) as zf:
            ck_exact[i], eps_exact[i] = read_scalar(zf, "c_k"), read_scalar(zf, "c_eps")
            raw_meta.add(tuple(read_scalar(zf, k) for k in ("n_years", "n_steps", "log_every_days")))
    assert len(raw_meta) == 1, f"raw files disagree on run length: {raw_meta}"
    assert np.allclose(ck_exact, run_ck, rtol=5e-4) and np.allclose(eps_exact, run_eps, rtol=5e-4), \
        "raw c_k/c_eps do not match the dataset's run_ck/run_eps"
    n_years, n_steps, log_every_days = raw_meta.pop()
    assert (int(n_steps), int(log_every_days)) == (common.N_STEPS, common.LOG_EVERY_DAYS)

    # ---- per-run configurations, split into common and varying parts
    flats = []
    for ck, eps in zip(ck_exact, eps_exact):
        model, _, _ = full.build(param_overrides={"tke_closure.c_k": ck, "tke_closure.c_eps": eps})
        flats.append({"config": flat_dump(model.config), "parameters": flat_dump(model.parameters)})
    for part in ("config", "parameters"):
        keys = set(flats[0][part])
        assert all(set(f[part]) == keys for f in flats)
        varying = {k for k in keys if any(f[part][k] != flats[0][part][k] for f in flats)}
        expected = set(VARYING) if part == "parameters" else set()
        assert varying == expected, f"{part}: varying keys {sorted(varying)}, expected {sorted(expected)}"
    config = flats[0]["config"]
    params = {k: v for k, v in flats[0]["parameters"].items() if k not in VARYING}

    # ---- round trip: the full dump fed back as overrides rebuilds the same model
    m_ref, _, _ = full.build(param_overrides={"tke_closure.c_k": ck_exact[0], "tke_closure.c_eps": eps_exact[0]})
    m_rt, _, _ = full.build(config_overrides=as_overrides(config),
                            param_overrides={**as_overrides(params),
                                             "tke_closure.c_k": ck_exact[0], "tke_closure.c_eps": eps_exact[0]})
    assert flat_dump(m_rt.config) == flat_dump(m_ref.config)
    assert flat_dump(m_rt.parameters) == flat_dump(m_ref.parameters)

    mv = os.path.join(args.mini_veros_repo, "mini-veros")
    run_config_common = {
        "schema": 1,
        "setup": "mini_veros.setups.acc.full",
        "build": "full.build(config_overrides=as_overrides(config), param_overrides=as_overrides(parameters) + "
                 "{k: run value for k in config_varying_keys}); keys ending in '.__class__' are not overrides",
        "config": config,
        "parameters": params,
        "run": {"init": "cold_start", "n_years": n_years, "n_steps": int(n_steps), "log_every_steps": common.LOG_EVERY,
                "log_every_days": int(log_every_days), "dt_tracer": config["dt_tracer"], "jax_enable_x64": True,
                "year_note": "model year = 365 days; the dataset's `year` key uses 360-day years"},
        "source": {"raw_dir": os.path.abspath(args.raw_dir), "wandb_sweep_id": os.path.basename(os.path.normpath(args.raw_dir)),
                   "sweep_yaml": "test/sweep/ck_ceps_100y.yaml"},
        "code": {"reconstructed": True,
                 "mini_veros_commit": git(mv, "rev-parse", "HEAD"),
                 "mini_veros_dirty": bool(git(mv, "status", "--porcelain", "--", "mini_veros")),
                 "repo_commit": git(args.mini_veros_repo, "rev-parse", "HEAD"),
                 "repo_dirty_files": git(args.mini_veros_repo, "status", "--porcelain", "--", "test/ck_ceps_100y_sweep").splitlines(),
                 "generation_code_note": "the sweep itself ran before 2026-09-22 from a Unison-synced working tree "
                                         "(mini-veros 1ead197 or f7ded9e era, old full.build({'c_k', 'c_eps'}) API, "
                                         "same TKE c_k / c_eps); exact version not recorded"},
        "verification": "cold-start rerun of ck0.126_eps0.8819 (c_k=0.12599, c_eps=0.88194) on CPU with the code "
                        "above, 12 monthly snapshots: max|diff|/max|ref| <= 2.2e-9 (temp), 5.1e-10 (salt), "
                        "<= 7.5e-8 (u, v, psi, tke, eke) vs the GPU raw file; the same run with the file-name "
                        "values (0.126, 0.8819) differs by ~1e-6 (temp) to ~1e-4 (tke)",
        "patched_by": "miniveros/patch_run_config.py",
    }
    varying_keys = {"parameters." + k: v for k, v in VARYING.items()}
    print(json.dumps({k: v for k, v in run_config_common.items() if k not in ("config", "parameters")}, indent=1))
    print(f"{len(config)} config keys, {len(params)} common parameter keys, varying: {varying_keys}")
    print(f"max rel |exact - run_ck| = {np.max(np.abs(ck_exact / run_ck - 1)):.2e}, "
          f"|exact - run_eps| = {np.max(np.abs(eps_exact / run_eps - 1)):.2e}")
    if args.dry_run:
        return

    new = {"run_config_common": np.array(json.dumps(run_config_common)),
           "config_varying_keys": np.array(json.dumps(varying_keys)),
           "run_ck_exact": ck_exact, "run_eps_exact": eps_exact}
    with zipfile.ZipFile(args.data_file, mode="a", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        for k, v in new.items():
            with zf.open(k + ".npy", mode="w", force_zip64=True) as fh:
                npf.write_array(fh, np.asanyarray(v), allow_pickle=False)
    with np.load(args.data_file, allow_pickle=False) as ds:
        assert json.loads(str(ds["run_config_common"])) == json.loads(json.dumps(run_config_common))
        assert np.array_equal(ds["run_ck_exact"], ck_exact) and np.array_equal(ds["run_eps_exact"], eps_exact)
        assert set(ds.files) == present | set(NEW_KEYS)
    print(f"appended {list(new)} to {args.data_file}")


if __name__ == "__main__":
    main()
