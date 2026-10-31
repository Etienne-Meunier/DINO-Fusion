"""The deterministic emulator and the baselines it is compared with.

Every method maps z, the standardised log-parameters (log c_k, log c_eps), to the vector y of last-year
temperatures on the water cells (K): ``fit(Z, Y)``, then ``predict(Z)``. The emulator is ``MLP``: a small network
from z to the leading PCA coefficients of the training states, decoded with the fixed PCA basis.

Importing this module limits BLAS and OpenMP to one thread in the process (joblib workers import it too). L-BFGS
turns the last-bit differences between thread counts into ~1e-3 K differences in the fitted network, so one
thread everywhere makes a fit identical in the main process and in any worker.
"""
from __future__ import annotations

import numpy as np
from threadpoolctl import threadpool_limits

threadpool_limits(1)


class PCA:
    """Mean and k leading principal directions of the training states (rows of Y)."""

    def __init__(self, k: int = 20):
        self.k = k

    def fit(self, Y):
        self.mean = Y.mean(0)
        _, _, vt = np.linalg.svd(Y - self.mean, full_matrices=False)
        self.basis = vt[: min(self.k, len(Y) - 1)]
        return self

    def coefficients(self, Y):
        return (Y - self.mean) @ self.basis.T

    def field(self, A):
        return self.mean + A @ self.basis


class MLP:
    """net(z) -> k leading PCA coefficients of the training states; an ensemble of ``members`` nets.

    The coefficients are divided by one constant (the std of the first), so the network's MSE is the physical MSE
    of the projected field up to a factor. The members differ only in their initialisation (``random_state`` =
    ``seed`` + member index); the prediction is their mean, and ``predict(Z, spread=True)`` also returns their
    spread (std over members, mean over cells, K)."""

    def __init__(self, hidden=(64, 64), activation="relu", alpha=1e-2, k=20, members=5, max_iter=5000, seed=0):
        self.hidden, self.activation, self.alpha = tuple(hidden), activation, alpha
        self.k, self.members, self.max_iter, self.seed = k, members, max_iter, seed

    def fit(self, Z, Y):
        from sklearn.neural_network import MLPRegressor
        self.pca = PCA(self.k).fit(Y)
        A = self.pca.coefficients(Y)
        self.scale = A[:, 0].std()
        self.nets = [MLPRegressor(hidden_layer_sizes=self.hidden, activation=self.activation, alpha=self.alpha,
                                  solver="lbfgs", max_iter=self.max_iter, tol=1e-8, random_state=self.seed + i).fit(Z, A / self.scale)
                     for i in range(self.members)]
        return self

    def predict(self, Z, spread=False):
        F = self.pca.field(np.stack([n.predict(Z).reshape(len(Z), -1) for n in self.nets]) * self.scale)   # (M, n, cells)
        return (F.mean(0), F.std(0).mean(-1)) if spread else F.mean(0)


class TrainMean:
    """The mean of the training states, whatever the parameters."""

    def fit(self, Z, Y):
        self.mean = Y.mean(0)
        return self

    def predict(self, Z):
        return np.repeat(self.mean[None], len(Z), axis=0)


class Nearest:
    """Copy the training run closest in z."""

    def fit(self, Z, Y):
        self.Z, self.Y = Z, Y
        return self

    def predict(self, Z):
        return self.Y[np.linalg.norm(Z[:, None] - self.Z[None], axis=-1).argmin(1)]


class RBF:
    """Thin-plate-spline interpolation of every cell, with a linear polynomial tail."""

    def fit(self, Z, Y):
        from scipy.interpolate import RBFInterpolator
        self.f = RBFInterpolator(Z, Y, kernel="thin_plate_spline", degree=1)
        return self

    def predict(self, Z):
        return self.f(Z)


class GP:
    """One Gaussian process per PCA coefficient (Matern 5/2, one length scale per input, white noise), with the
    hyper-parameters fitted by maximum marginal likelihood (no cross-validation). The noise may take up to the
    whole (normalised) variance, so coefficients that look like noise are fitted as noise.

    ``on="s"``: the input is the single coordinate s = w . z along which the first PCA coefficient of the training
    states varies linearly (unit w from a least-squares fit): the one-dimensional collapse of the two parameters."""

    def __init__(self, k: int = 20, on: str = "z"):
        self.k, self.on = k, on

    def _x(self, Z):
        return Z if self.on == "z" else (Z @ self.w)[:, None]

    def fit(self, Z, Y):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
        self.pca = PCA(self.k).fit(Y)
        A = self.pca.coefficients(Y)
        if self.on == "s":
            w = np.linalg.lstsq(np.c_[Z, np.ones(len(Z))], A[:, 0], rcond=None)[0][:2]
            self.w = w / np.linalg.norm(w)
        X = self._x(Z)
        kernel = ConstantKernel(1.0) * Matern(np.ones(X.shape[1]), (1e-2, 1e2), nu=2.5) + WhiteKernel(1e-6, (1e-10, 1.0))
        self.gps = [GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=2, random_state=0).fit(X, a)
                    for a in A.T]
        return self

    def predict(self, Z):
        X = self._x(Z)
        return self.pca.field(np.stack([g.predict(X) for g in self.gps], axis=1))


def select(make, grid, Z, Y, folds=5, repeats=3, seed=0, n_jobs=-1):
    """Hyper-parameters by repeated k-fold cross-validation on (Z, Y), the training runs of a split, and the
    one-standard-error rule: among the grid points whose CV RMSE (mean over runs) is within one standard error of
    the best, the most regularised one (largest alpha; ties by lower CV RMSE). The standard error is over runs, not
    folds: the std over the runs of the best point's per-run CV RMSE (averaged over the repeats), divided by
    sqrt(runs). With fewer runs than folds, the k-fold is leave-one-out and the repeats are identical. Returns the
    chosen point, the CV RMSE of every point and the standard error; the caller can also take the minimum. The
    partitions of the repeats are seeded seed, seed + 1, ..."""
    from joblib import Parallel, delayed
    from sklearn.model_selection import KFold

    parts = [p for rep in range(repeats) for p in KFold(min(folds, len(Z)), shuffle=True, random_state=seed + rep).split(Z)]

    def score(params, tr, va):
        P = make(**params).fit(Z[tr], Y[tr]).predict(Z[va])
        return va, np.sqrt(((P - Y[va]) ** 2).mean(1))

    res = Parallel(n_jobs=n_jobs)(delayed(score)(g, tr, va) for g in grid for tr, va in parts)
    err = np.zeros((len(grid), len(Z)))                          # per-run CV RMSE, averaged over the repeats
    for i, (va, e) in enumerate(res):
        err[i // len(parts), va] += e / repeats
    mean = err.mean(1)
    best = int(mean.argmin())
    se = err[best].std(ddof=1) / np.sqrt(len(Z))
    pick = max((i for i in range(len(grid)) if mean[i] <= mean[best] + se), key=lambda i: (grid[i]["alpha"], -mean[i]))
    return grid[pick], mean, se
