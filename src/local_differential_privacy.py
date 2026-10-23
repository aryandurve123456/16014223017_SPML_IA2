"""
Local Differential Privacy (LDP) Module for IA-2.
Implements Warner's Randomized Response, Duchi's Mechanism, Piecewise Mechanism,
multidimensional budget composition, and decision-boundary calibration to fix the 0-recall collapse of IA-1.
"""

import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from baseline import find_optimal_threshold


class BinaryRandomizedResponse:
    def __init__(self, epsilon: float, random_state: Optional[int] = None):
        if epsilon <= 0:
            raise ValueError(f"Epsilon must be strictly positive, got {epsilon}")
        self.epsilon = float(epsilon)
        self.rng = np.random.default_rng(random_state)
        self.p = np.exp(self.epsilon) / (np.exp(self.epsilon) + 1.0)
        self.q = 1.0 - self.p

    def randomize(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.int64)
        flip_mask = self.rng.random(size=arr.shape) > self.p
        return np.where(flip_mask, 1 - arr, arr)

    def verify_privacy_bound(self) -> float:
        return float(np.log(self.p / self.q))


class DuchiMechanism:
    def __init__(self, epsilon: float, random_state: Optional[int] = None):
        if epsilon <= 0:
            raise ValueError(f"Epsilon must be positive, got {epsilon}")
        self.epsilon = float(epsilon)
        self.rng = np.random.default_rng(random_state)
        self.C = (np.exp(self.epsilon) + 1.0) / (np.exp(self.epsilon) - 1.0)

    def randomize(self, x: np.ndarray) -> np.ndarray:
        arr = np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0)
        prob_pos = (arr / (2.0 * self.C)) + 0.5
        u = self.rng.random(size=arr.shape)
        return np.where(u < prob_pos, self.C, -self.C)

    def verify_privacy_bound(self) -> float:
        prob_max = (1.0 / (2.0 * self.C)) + 0.5
        prob_min = (-1.0 / (2.0 * self.C)) + 0.5
        return float(np.log(prob_max / prob_min))


class PiecewiseMechanism:
    def __init__(self, epsilon: float, random_state: Optional[int] = None):
        if epsilon <= 0:
            raise ValueError(f"Epsilon must be positive, got {epsilon}")
        self.epsilon = float(epsilon)
        self.rng = np.random.default_rng(random_state)
        exp_half = np.exp(self.epsilon / 2.0)
        self.C = (exp_half + 1.0) / (exp_half - 1.0)
        self.p_sample_inner = exp_half / (exp_half + 1.0)

    def randomize(self, x: np.ndarray) -> np.ndarray:
        arr = np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0)
        l_t = ((self.C + 1.0) / 2.0) * arr - ((self.C - 1.0) / 2.0)
        r_t = l_t + self.C - 1.0

        u_decision = self.rng.random(size=arr.shape)
        u_inner = self.rng.random(size=arr.shape)
        sample_inner = l_t + u_inner * (r_t - l_t)

        u_outer = self.rng.random(size=arr.shape)
        outer_length = self.C + 1.0
        left_length = l_t + self.C
        sample_outer_left = -self.C + (u_outer * outer_length)
        sample_outer_right = r_t + ((u_outer * outer_length) - left_length)
        sample_outer = np.where(u_outer * outer_length < left_length, sample_outer_left, sample_outer_right)

        return np.where(u_decision < self.p_sample_inner, sample_inner, sample_outer)


class MultidimensionalLDPPerturber:
    def __init__(
        self,
        epsilon_total: float,
        numerical_feature_count: int = 6,
        numerical_mechanism: str = "duchi",
        random_state: Optional[int] = None,
    ):
        self.epsilon_total = float(epsilon_total)
        self.numerical_feature_count = numerical_feature_count
        self.numerical_mechanism = numerical_mechanism.lower()
        self.random_state = random_state

    def perturb(self, X: np.ndarray) -> np.ndarray:
        n_samples, n_features = X.shape
        eps_per_feature = self.epsilon_total / float(n_features)
        rng = np.random.default_rng(self.random_state)
        X_priv = np.empty_like(X, dtype=np.float64)

        # 1. Numerical features
        n_num = self.numerical_feature_count
        if n_num > 0:
            if self.numerical_mechanism == "duchi":
                mech = DuchiMechanism(eps_per_feature, random_state=rng.integers(0, 10**8))
            elif self.numerical_mechanism == "piecewise":
                mech = PiecewiseMechanism(eps_per_feature, random_state=rng.integers(0, 10**8))
            else:
                raise ValueError(f"Unknown mechanism: {self.numerical_mechanism}")
            X_priv[:, :n_num] = mech.randomize(X[:, :n_num])

        # 2. Categorical features
        n_cat = n_features - n_num
        if n_cat > 0:
            rr = BinaryRandomizedResponse(eps_per_feature, random_state=rng.integers(0, 10**8))
            X_priv[:, n_num:] = rr.randomize(np.round(X[:, n_num:]).astype(np.int64)).astype(np.float64)

        return X_priv


def train_evaluate_ldp_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epsilon: float,
    numerical_feature_count: int = 6,
    numerical_mechanism: str = "duchi",
    random_state: int = 42,
) -> Dict:
    perturber = MultidimensionalLDPPerturber(
        epsilon_total=epsilon,
        numerical_feature_count=numerical_feature_count,
        numerical_mechanism=numerical_mechanism,
        random_state=random_state,
    )

    t0 = time.perf_counter()
    X_train_priv = perturber.perturb(X_train)
    t_pert = time.perf_counter() - t0

    # Train with balanced class weights to address noise + class imbalance
    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=random_state,
        solver="lbfgs",
    )
    t1 = time.perf_counter()
    model.fit(X_train_priv, y_train)
    t_train = time.perf_counter() - t1

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred_default = (y_prob >= 0.5).astype(np.int64)

    # Threshold calibration on privatized train data
    y_prob_tr = model.predict_proba(X_train_priv)[:, 1]
    opt_thresh = find_optimal_threshold(y_train, y_prob_tr)
    y_pred_calibrated = (y_prob >= opt_thresh).astype(np.int64)

    return {
        "model_name": f"LDP Logistic Regression (eps={epsilon}, {numerical_mechanism})",
        "paradigm": "Local Differential Privacy (Warner RR + " + numerical_mechanism.title() + ")",
        "epsilon": epsilon,
        "delta": 0.0,
        "perturbation_time_sec": round(t_pert, 4),
        "train_time_sec": round(t_train, 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "default_threshold_0.5": {
            "accuracy": round(float(accuracy_score(y_test, y_pred_default)), 4),
            "precision": round(float(precision_score(y_test, y_pred_default, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred_default, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, y_pred_default, zero_division=0)), 4),
            "confusion_matrix": confusion_matrix(y_test, y_pred_default).tolist(),
        },
        "calibrated_threshold": {
            "threshold_value": round(opt_thresh, 4),
            "accuracy": round(float(accuracy_score(y_test, y_pred_calibrated)), 4),
            "precision": round(float(precision_score(y_test, y_pred_calibrated, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred_calibrated, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, y_pred_calibrated, zero_division=0)), 4),
            "confusion_matrix": confusion_matrix(y_test, y_pred_calibrated).tolist(),
        },
    }


if __name__ == "__main__":
    from data_preparation import get_prepared_data

    X_tr, y_tr, X_te, y_te, _, _ = get_prepared_data()
    for eps in [0.5, 1.0, 5.0]:
        res = train_evaluate_ldp_model(X_tr, y_tr, X_te, y_te, epsilon=eps)
        print(f"LDP eps={eps}: ROC-AUC={res['roc_auc']}, Default F1={res['default_threshold_0.5']['f1']}, Calibrated F1={res['calibrated_threshold']['f1']} (thresh={res['calibrated_threshold']['threshold_value']})")
