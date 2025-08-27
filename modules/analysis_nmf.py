from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from sklearn.decomposition import NMF
from scipy.optimize import nnls

@dataclass
class NMFResult:
    W: np.ndarray
    H: np.ndarray
    X_hat: np.ndarray
    residuals: np.ndarray
    recon_error: float

def _nnls_W_given_H(X: np.ndarray, H: np.ndarray) -> np.ndarray:
    k, n_features = H.shape
    W = np.zeros((X.shape[0], k), dtype=float)
    A = H.T
    for i in range(X.shape[0]):
        w_i, _ = nnls(A, X[i])
        W[i] = w_i
    return W

def run_nmf(X: np.ndarray, n_components: int, init: str = "nndsvd", H_init: Optional[np.ndarray] = None,
            max_iter: int = 2000, l1_ratio: float = 0.0, alpha_W: float = 0.0, alpha_H: float = 0.0,
            random_state: Optional[int] = 0) -> NMFResult:
    X = np.asarray(X, dtype=float)
    if (X < 0).any():
        raise ValueError("X has negative entries. Clip or shift before NMF.")
    if init == "custom":
        if H_init is None:
            raise ValueError("H_init must be provided when init='custom'.")
        if H_init.shape[0] != n_components or H_init.shape[1] != X.shape[1]:
            raise ValueError(f"H_init shape {H_init.shape} incompatible with (k, n_features)=({n_components}, {X.shape[1]})")
        W_init = _nnls_W_given_H(X, H_init)
        model = NMF(n_components=n_components, init="custom", max_iter=max_iter, l1_ratio=l1_ratio,
                    alpha_W=alpha_W, alpha_H=alpha_H, random_state=random_state)
        W = model.fit_transform(X, W=W_init, H=H_init.copy())
        H = model.components_
    else:
        model = NMF(n_components=n_components, init=init, max_iter=max_iter, l1_ratio=l1_ratio,
                    alpha_W=alpha_W, alpha_H=alpha_H, random_state=random_state)
        W = model.fit_transform(X)
        H = model.components_
    X_hat = W @ H
    R = X - X_hat
    err = np.linalg.norm(R) / np.linalg.norm(X) if np.linalg.norm(X) > 0 else np.linalg.norm(R)
    return NMFResult(W=W, H=H, X_hat=X_hat, residuals=R, recon_error=err)
