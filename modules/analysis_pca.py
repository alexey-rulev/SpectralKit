from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from sklearn.decomposition import PCA

@dataclass
class PCAResult:
    components: np.ndarray
    scores: np.ndarray
    mean_: np.ndarray
    explained_variance_ratio: np.ndarray
    X_hat: np.ndarray
    residuals: np.ndarray
    recon_error: float

def run_pca(X: np.ndarray, n_components: int, whiten: bool = False, random_state: Optional[int] = 0) -> PCAResult:
    X = np.asarray(X, dtype=float)
    model = PCA(n_components=n_components, whiten=whiten, random_state=random_state)
    scores = model.fit_transform(X)
    comps = model.components_
    mean_ = model.mean_
    X_hat = scores @ comps + mean_
    R = X - X_hat
    err = np.linalg.norm(R) / np.linalg.norm(X) if np.linalg.norm(X) > 0 else np.linalg.norm(R)
    return PCAResult(components=comps, scores=scores, mean_=mean_,
                     explained_variance_ratio=model.explained_variance_ratio_,
                     X_hat=X_hat, residuals=R, recon_error=err)
