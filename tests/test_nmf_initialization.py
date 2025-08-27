import os
import sys
import numpy as np

# Ensure the project root is on the Python path for imports during tests
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from modules.analysis_nmf import run_nmf


def test_partial_h_init_fills_missing_components():
    """``run_nmf`` should accept partial ``H_init`` and pad the rest randomly."""
    X = np.abs(np.random.rand(5, 10))
    H_init = np.abs(np.random.rand(2, 10))  # fewer rows than n_components

    res = run_nmf(X, n_components=3, init="custom", H_init=H_init, random_state=0)

    assert res.H.shape == (3, 10)
    assert np.all(res.H >= 0)
