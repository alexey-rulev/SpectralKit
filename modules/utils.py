from __future__ import annotations
from typing import Optional
import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter1d, gaussian_filter1d

def clip_nonneg(X: np.ndarray) -> np.ndarray:
    return np.clip(X, 0.0, None)

def normalize_rows(X: np.ndarray, method: Optional[str] = None) -> np.ndarray:
    if method is None or method == "none":
        return X
    X = X.astype(float)
    if method == "max":
        m = X.max(axis=1, keepdims=True); m[m==0]=1.0; return X/m
    if method == "l2":
        m = np.linalg.norm(X, axis=1, keepdims=True); m[m==0]=1.0; return X/m
    if method == "area":
        m = X.sum(axis=1, keepdims=True); m[m==0]=1.0; return X/m
    raise ValueError(f"Unknown normalization method: {method}")

def apply_smoothing(X: np.ndarray, method: str = "none", window: int = 9, poly: int = 2, sigma: float = 2.0) -> np.ndarray:
    if method in (None, "none"):
        return X
    n_feat = X.shape[1]
    out = np.empty_like(X, dtype=float)
    if method == "savgol":
        if window > n_feat: window = n_feat-1 if n_feat % 2 == 0 else n_feat
        if window % 2 == 0: window += 1
        if window < 3: return X
        for i in range(X.shape[0]):
            out[i] = savgol_filter(X[i], window_length=window, polyorder=min(poly, window-1))
        return out
    if method == "moving_average":
        size = max(1, min(int(window), n_feat))
        for i in range(X.shape[0]):
            out[i] = uniform_filter1d(X[i], size=size, mode="nearest")
        return out
    if method == "gaussian":
        s = max(0.1, float(sigma))
        for i in range(X.shape[0]):
            out[i] = gaussian_filter1d(X[i], sigma=s, mode="nearest")
        return out
    raise ValueError(f"Unknown smoothing method: {method}")


def apply_sample_smoothing(X: np.ndarray, method: str = "none", window: int = 9, poly: int = 2, sigma: float = 2.0) -> np.ndarray:
    """Smooth each grid value across adjacent uploaded samples (rows of ``X``)."""
    if X.ndim != 2:
        raise ValueError("Sample smoothing requires a 2D array.")
    return apply_smoothing(X.T, method=method, window=window, poly=poly, sigma=sigma).T
