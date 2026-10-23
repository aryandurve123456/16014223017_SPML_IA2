"""
Unit tests for Differential Privacy and Local Differential Privacy mathematical invariants.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from differential_privacy import DifferentiallyPrivateLogisticRegression
from dp_sgd import train_dp_sgd
from local_differential_privacy import (
    BinaryRandomizedResponse,
    DuchiMechanism,
    PiecewiseMechanism,
)


def test_binary_randomized_response_bounds():
    for eps in [0.5, 1.0, 2.0]:
        rr = BinaryRandomizedResponse(epsilon=eps, random_state=42)
        bound = rr.verify_privacy_bound()
        assert np.isclose(bound, eps, atol=1e-7)

        # Test output domain {0, 1}
        inputs = np.array([0, 1, 1, 0, 1] * 50)
        outputs = rr.randomize(inputs)
        assert set(np.unique(outputs)).issubset({0, 1})


def test_duchi_mechanism_unbiasedness_and_bounds():
    for eps in [1.0, 2.0]:
        duchi = DuchiMechanism(epsilon=eps, random_state=42)
        bound = duchi.verify_privacy_bound()
        assert np.isclose(bound, eps, atol=1e-7)

        # Output domain {-C, +C}
        arr = np.array([-0.5, 0.0, 0.5])
        rand_arr = duchi.randomize(arr)
        for v in rand_arr:
            assert np.isclose(abs(v), duchi.C, atol=1e-7)

        # Unbiasedness E[y] = x
        n_samples = 40000
        val = 0.3
        samples = np.full(n_samples, val)
        rand_samples = duchi.randomize(samples)
        assert np.isclose(np.mean(rand_samples), val, atol=0.04)


def test_piecewise_mechanism_domain_and_expectation():
    eps = 1.0
    pm = PiecewiseMechanism(epsilon=eps, random_state=42)
    C = pm.C

    x = np.array([-0.8, -0.2, 0.0, 0.3, 0.9])
    y = pm.randomize(x)
    assert np.all(y >= -C - 1e-7)
    assert np.all(y <= C + 1e-7)

    # Unbiasedness test
    n_samples = 40000
    val = -0.25
    samples = np.full(n_samples, val)
    rand_samples = pm.randomize(samples)
    assert np.isclose(np.mean(rand_samples), val, atol=0.04)


def test_dp_erm_sensitivity_bound():
    eps = 1.0
    lambda_reg = 0.02
    n_samples = 1000

    dp_clf = DifferentiallyPrivateLogisticRegression(
        epsilon=eps, lambda_reg=lambda_reg, random_state=42
    )

    X_dummy = np.random.uniform(-1, 1, size=(n_samples, 10))
    y_dummy = np.random.randint(0, 2, size=n_samples)
    dp_clf.fit(X_dummy, y_dummy)

    # Expected sensitivity: Delta_2 <= 2 * R / (n * lambda)
    expected_sens = 2.0 / (n_samples * lambda_reg)
    assert np.isclose(dp_clf.sensitivity_, expected_sens, atol=1e-7)
    assert dp_clf.noise_std_ > 0.0
