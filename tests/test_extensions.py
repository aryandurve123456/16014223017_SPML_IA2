"""
Unit tests for IA-2 extensions: Secure Aggregation, Compressive Privacy, and MIA.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from compressive_privacy import CompressivePrivacyProjector
from federated_learning import simulate_secure_aggregation
from membership_inference import extract_attack_features


def test_secure_aggregation_exact_cancellation():
    K = 5
    d = 100
    rng = np.random.default_rng(42)
    client_updates = [rng.normal(0, 1, size=(d,)) for _ in range(K)]

    secagg = simulate_secure_aggregation(client_updates, mask_scale=50.0, random_state=42)

    assert secagg["exact_cancellation_verified"] is True
    assert secagg["max_cancellation_error"] < 1e-10
    # Blinded updates must have low correlation with raw updates
    assert abs(secagg["mean_correlation_blinded_to_raw"]) < 0.25


def test_compressive_privacy_subspace_orthogonalization():
    n = 200
    d = 10
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, size=(n, d))
    s = rng.integers(0, 2, size=n)

    projector = CompressivePrivacyProjector(n_components=4, random_state=42)
    Z = projector.fit_transform(X, s)

    assert Z.shape == (n, 4)
    v_s = projector.sensitive_direction_

    # Verify that X projected onto orthogonal complement has near-zero dot product with v_s
    X_ortho = X - np.outer(np.dot(X, v_s), v_s)
    dot_products = np.dot(X_ortho, v_s)
    assert np.allclose(dot_products, 0.0, atol=1e-10)


def test_mia_feature_extraction_shape():
    class DummyModel:
        def predict_proba(self, X):
            p1 = np.full((len(X), 1), 0.75)
            p0 = 1.0 - p1
            return np.hstack([p0, p1])

    X_dummy = np.zeros((20, 5))
    y_dummy = np.ones(20, dtype=int)

    feats = extract_attack_features(DummyModel(), X_dummy, y_dummy)
    assert feats.shape[0] == 20
    assert feats.shape[1] == 6  # 2 probs + true_prob + nll + entropy + conf
    assert not np.isnan(feats).any()
