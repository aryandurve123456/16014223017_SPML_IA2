"""
Central Differential Privacy Module for IA-2.
Implements Differentially Private Empirical Risk Minimization (DP-ERM) via Output Perturbation
(Chaudhuri, Monteleoni, Sarwate, JMLR 2011) with proven L2-sensitivity bounds,
calibrated Gaussian noise, and decision-threshold calibration.
"""

import time
from typing import Dict, Optional, Tuple

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


class DifferentiallyPrivateLogisticRegression:
    """
    (epsilon, delta)-Differentially Private Logistic Regression via Output Perturbation.
    
    Mathematical Derivation (Chaudhuri et al., 2011):
    Objective: min_w (1/n) * sum(loss_i(w)) + (lambda/2) * ||w||_2^2
    Inputs clipped to ||x_i||_2 <= R => Lipschitz constant L = R = 1.0.
    Strong convexity parameter = lambda.
    L2 global sensitivity: Delta_2(w*) <= (2 * R) / (n * lambda).
    Gaussian noise std: sigma = (Delta_2 * sqrt(2 * ln(1.25 / delta))) / epsilon.
    """

    def __init__(
        self,
        epsilon: float,
        delta: float = 1e-5,
        lambda_reg: float = 0.01,
        max_iter: int = 1000,
        random_state: Optional[int] = 42,
    ):
        if epsilon <= 0:
            raise ValueError(f"Epsilon must be strictly positive, got {epsilon}")
        if not (0 < delta < 1):
            raise ValueError(f"Delta must be in (0, 1), got {delta}")

        self.epsilon = float(epsilon)
        self.delta = float(delta)
        self.lambda_reg = float(lambda_reg)
        self.max_iter = max_iter
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[np.ndarray] = None
        self.sensitivity_: float = 0.0
        self.noise_std_: float = 0.0
        self.optimal_threshold_: float = 0.5

    def fit(
        self, X: np.ndarray, y: np.ndarray, calibrate_threshold: bool = True
    ) -> "DifferentiallyPrivateLogisticRegression":
        n_samples, n_features = X.shape

        # Augmented feature vector [x, 1] for unified weight and intercept sensitivity
        X_aug = np.hstack([X, np.ones((n_samples, 1))])
        norms = np.linalg.norm(X_aug, axis=1, keepdims=True)
        R = 1.0
        X_aug = X_aug * (R / np.maximum(norms, R))

        # Scikit-learn C parameter = 1 / (n * lambda)
        C_sklearn = 1.0 / (n_samples * self.lambda_reg)
        base_lr = LogisticRegression(
            C=C_sklearn,
            fit_intercept=False,
            max_iter=self.max_iter,
            random_state=self.random_state,
            solver="lbfgs",
        )
        base_lr.fit(X_aug, y)
        w_clean = base_lr.coef_.flatten()

        # Exact L2 global sensitivity bound
        self.sensitivity_ = (2.0 * R) / (n_samples * self.lambda_reg)

        # Calibrate Gaussian noise standard deviation
        self.noise_std_ = (
            self.sensitivity_ * np.sqrt(2.0 * np.log(1.25 / self.delta))
        ) / self.epsilon

        noise = self.rng.normal(loc=0.0, scale=self.noise_std_, size=w_clean.shape)
        w_private = w_clean + noise

        self.coef_ = w_private[:-1].reshape(1, -1)
        self.intercept_ = np.array([w_private[-1]])

        if calibrate_threshold:
            scores_tr = np.dot(X, self.coef_.T).ravel() + self.intercept_[0]
            probs_tr = 1.0 / (1.0 + np.exp(-np.clip(scores_tr, -30.0, 30.0)))
            self.optimal_threshold_ = find_optimal_threshold(y, probs_tr)

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("Model must be fitted before calling decision_function.")
        return np.dot(X, self.coef_.T).ravel() + self.intercept_[0]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        probs_pos = 1.0 / (1.0 + np.exp(-np.clip(scores, -30.0, 30.0)))
        return np.column_stack([1.0 - probs_pos, probs_pos])

    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        th = self.optimal_threshold_ if threshold is None else threshold
        probs = self.predict_proba(X)[:, 1]
        return (probs >= th).astype(np.int64)


def train_evaluate_dp_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epsilon: float,
    delta: float = 1e-5,
    lambda_reg: float = 0.01,
    random_state: int = 42,
) -> Tuple[DifferentiallyPrivateLogisticRegression, Dict]:
    dp_model = DifferentiallyPrivateLogisticRegression(
        epsilon=epsilon,
        delta=delta,
        lambda_reg=lambda_reg,
        random_state=random_state,
    )

    t0 = time.perf_counter()
    dp_model.fit(X_train, y_train, calibrate_threshold=True)
    t_train = time.perf_counter() - t0

    t1 = time.perf_counter()
    y_prob = dp_model.predict_proba(X_test)[:, 1]
    t_infer = time.perf_counter() - t1

    y_pred_default = (y_prob >= 0.5).astype(np.int64)
    y_pred_calibrated = (y_prob >= dp_model.optimal_threshold_).astype(np.int64)

    metrics = {
        "model_name": f"DP-ERM Logistic Regression (eps={epsilon}, delta={delta})",
        "privacy_mechanism": "Central DP Output Perturbation (Gaussian DP-ERM)",
        "epsilon": epsilon,
        "delta": delta,
        "train_time_sec": round(t_train, 4),
        "inference_time_sec": round(t_infer, 4),
        "theoretical_sensitivity": round(dp_model.sensitivity_, 8),
        "calibrated_noise_std": round(dp_model.noise_std_, 8),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "default_threshold_0.5": {
            "accuracy": round(float(accuracy_score(y_test, y_pred_default)), 4),
            "precision": round(float(precision_score(y_test, y_pred_default, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred_default, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, y_pred_default, zero_division=0)), 4),
            "confusion_matrix": confusion_matrix(y_test, y_pred_default).tolist(),
        },
        "calibrated_threshold": {
            "threshold_value": round(dp_model.optimal_threshold_, 4),
            "accuracy": round(float(accuracy_score(y_test, y_pred_calibrated)), 4),
            "precision": round(float(precision_score(y_test, y_pred_calibrated, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred_calibrated, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, y_pred_calibrated, zero_division=0)), 4),
            "confusion_matrix": confusion_matrix(y_test, y_pred_calibrated).tolist(),
        },
    }

    return dp_model, metrics


if __name__ == "__main__":
    from data_preparation import get_prepared_data

    X_tr, y_tr, X_te, y_te, _, _ = get_prepared_data()
    for eps in [0.5, 1.0, 5.0]:
        _, res = train_evaluate_dp_model(X_tr, y_tr, X_te, y_te, epsilon=eps)
        print(f"Central DP eps={eps}: ROC-AUC={res['roc_auc']}, Default F1={res['default_threshold_0.5']['f1']}, Calibrated F1={res['calibrated_threshold']['f1']} (thresh={res['calibrated_threshold']['threshold_value']})")
