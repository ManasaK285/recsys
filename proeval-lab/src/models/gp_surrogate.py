from __future__ import annotations
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler

class GPSurrogate:
    """CPU-friendly GP surrogate with predictive mean and standard deviation."""
    def __init__(self, max_train_points: int = 250, random_state: int = 42):
        self.max_train_points = max_train_points
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if len(X) == 0:
            raise ValueError("Cannot fit GP with zero observations")
        if len(X) > self.max_train_points:
            rng = np.random.default_rng(self.random_state)
            keep = np.sort(rng.choice(len(X), self.max_train_points, replace=False))
            X, y = X[keep], y[keep]
        Xs = self.scaler.fit_transform(X)
        kernel = ConstantKernel(1.0, (0.1, 10.0)) * RBF(length_scale=1.0, length_scale_bounds=(.1, 10.0)) + WhiteKernel(noise_level=.03, noise_level_bounds=(1e-4, .2))
        self.model = GaussianProcessRegressor(
            kernel=kernel, normalize_y=True, n_restarts_optimizer=0,
            random_state=self.random_state, alpha=1e-5
        )
        self.model.fit(Xs, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray, return_std: bool = True):
        if not self.is_fitted:
            raise RuntimeError("Call fit before predict")
        Xs = self.scaler.transform(np.asarray(X, dtype=float))
        mean, std = self.model.predict(Xs, return_std=True)
        mean = np.clip(mean, 0.0, 1.0)
        std = np.maximum(std, 1e-6)
        return (mean, std) if return_std else mean
