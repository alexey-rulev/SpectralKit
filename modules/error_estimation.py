from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class ErrorEstimates:
    weight_std: np.ndarray
    component_std: np.ndarray


def _align_components(
    reference_components: np.ndarray,
    components: np.ndarray,
    weights: np.ndarray,
    nonnegative: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Match resampled components to the fitted ordering and scale."""
    reference_norms = np.linalg.norm(reference_components, axis=1, keepdims=True)
    component_norms = np.linalg.norm(components, axis=1, keepdims=True)
    reference_unit = reference_components / np.maximum(reference_norms, np.finfo(float).eps)
    component_unit = components / np.maximum(component_norms, np.finfo(float).eps)
    similarity = reference_unit @ component_unit.T
    if nonnegative:
        similarity = np.clip(similarity, 0.0, None)
    else:
        similarity = np.abs(similarity)

    row_indices, column_indices = linear_sum_assignment(-similarity)
    order = column_indices[np.argsort(row_indices)]
    components = components[order].copy()
    weights = weights[:, order].copy()

    for index in range(components.shape[0]):
        scale = np.dot(reference_components[index], components[index])
        scale /= max(np.dot(components[index], components[index]), np.finfo(float).eps)
        if nonnegative:
            scale = max(scale, np.finfo(float).eps)
        components[index] *= scale
        weights[:, index] /= scale
    return components, weights


def estimate_errors(
    X_hat: np.ndarray,
    residuals: np.ndarray,
    reference_weights: np.ndarray,
    reference_components: np.ndarray,
    fit: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    bootstrap_samples: int,
    random_state: Optional[int],
    nonnegative: bool = False,
    components_fixed: bool = False,
) -> ErrorEstimates:
    """Estimate one-standard-deviation errors by residual bootstrap resampling."""
    if bootstrap_samples < 2:
        raise ValueError("bootstrap_samples must be at least 2")

    rng = np.random.default_rng(random_state)
    weight_samples = np.empty((bootstrap_samples, *reference_weights.shape), dtype=float)
    component_samples = np.empty((bootstrap_samples, *reference_components.shape), dtype=float)

    for index in range(bootstrap_samples):
        sampled_residuals = residuals[rng.integers(0, residuals.shape[0], size=residuals.shape[0])]
        simulated_X = X_hat + sampled_residuals
        if nonnegative:
            simulated_X = np.clip(simulated_X, 0.0, None)
        weights, components = fit(simulated_X)
        if components_fixed:
            components = reference_components.copy()
        else:
            components, weights = _align_components(
                reference_components, components, weights, nonnegative=nonnegative
            )
        weight_samples[index] = weights
        component_samples[index] = components

    return ErrorEstimates(
        weight_std=np.std(weight_samples, axis=0, ddof=1),
        component_std=np.std(component_samples, axis=0, ddof=1),
    )
