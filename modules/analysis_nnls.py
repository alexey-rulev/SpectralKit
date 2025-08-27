from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import nnls

@dataclass
class NNLSResult:
    coeffs: np.ndarray
    X_hat: np.ndarray
    residuals: np.ndarray
    recon_error: float

def run_nnls(X: np.ndarray, B: np.ndarray) -> NNLSResult:
    X = np.asarray(X, dtype=float)
    B = np.asarray(B, dtype=float)
    if B.shape[1] != X.shape[1]:
        raise ValueError(f"Basis/features mismatch: B has n_features={B.shape[1]} but X has {X.shape[1]}")
    A = B.T
    coeffs = np.zeros((X.shape[0], B.shape[0]), dtype=float)
    for i in range(X.shape[0]):
        c_i, _ = nnls(A, X[i])
        coeffs[i] = c_i
    X_hat = coeffs @ B
    R = X - X_hat
    err = np.linalg.norm(R) / np.linalg.norm(X) if np.linalg.norm(X) > 0 else np.linalg.norm(R)
    return NNLSResult(coeffs=coeffs, X_hat=X_hat, residuals=R, recon_error=err)
