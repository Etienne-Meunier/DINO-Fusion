"""The deterministic emulator and the baselines it is compared with.

Every method maps z, the standardised log-parameters (log c_k, log c_eps), to the vector y of last-year
temperatures on the water cells (K): ``fit(Z, Y)``, then ``predict(Z)``. The emulator is ``MLP``: a small network
from z to the leading PCA coefficients of the training states, decoded with the fixed PCA basis.
"""
from __future__ import annotations

import numpy as np


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
    of the projected field up to a factor. The members differ only in their initialisation; the prediction is their
    mean, and ``predict(Z, spread=True)`` also returns their spread (std over members, mean over cells, K)."""

    def __init__(self, hidden=(64, 64), activation="relu", alpha=1e-2, k=20, members=5, max_iter=5000):
        self.hidden, self.activation, self.alpha = tuple(hidden), activation, alpha
        self.k, self.members, self.max_iter = k, members, max_iter

    def fit(self, Z, Y):
        from sklearn.neural_network import MLPRegressor
        self.pca = PCA(self.k).fit(Y)
        A = self.pca.coefficients(Y)
        self.scale = A[:, 0].std()
        self.nets = [MLPRegressor(hidden_layer_sizes=self.hidden, activation=self.activation, alpha=self.alpha,
                                  solver="lbfgs", max_iter=self.max_iter, tol=1e-8, random_state=i).fit(Z, A / self.scale)
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
    """One Gaussian process per PCA coefficient (Matern 5/2, one length scale per input, white noise).

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
        kernel = ConstantKernel(1.0) * Matern(np.ones(X.shape[1]), (1e-2, 1e2), nu=2.5) + WhiteKernel(1e-6, (1e-10, 1e-1))
        self.gps = [GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=2, random_state=0).fit(X, a)
                    for a in A.T]
        return self

    def predict(self, Z):
        X = self._x(Z)
        return self.pca.field(np.stack([g.predict(X) for g in self.gps], axis=1))


def select(make, grid, Z, Y, folds=5, n_jobs=-1):
    """The grid point with the lowest k-fold cross-validated RMSE (mean over runs) on (Z, Y), and all the scores.
    Called on the training runs of a split only."""
    from joblib import Parallel, delayed
    from sklearn.model_selection import KFold

    parts = list(KFold(min(folds, len(Z)), shuffle=True, random_state=0).split(Z))

    def score(params, tr, va):
        P = make(**params).fit(Z[tr], Y[tr]).predict(Z[va])
        return np.sqrt(((P - Y[va]) ** 2).mean(1)).sum()

    s = Parallel(n_jobs=n_jobs)(delayed(score)(g, tr, va) for g in grid for tr, va in parts)
    s = np.asarray(s).reshape(len(grid), len(parts)).sum(1) / len(Z)
    return grid[int(s.argmin())], s
