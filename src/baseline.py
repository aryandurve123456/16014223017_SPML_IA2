"""
Baseline Classification Models Module for IA-2.
Implements non-private reference models (Logistic Regression and Multilayer Perceptron),
with both default (0.5) and calibrated decision thresholds to properly handle class imbalance.
"""

import time
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier


def find_optimal_threshold(y_true: np.ndarray, y_probs: np.ndarray) -> float:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    denom = precisions + recalls
    f1_scores = np.where(
        denom > 0,
        2.0 * (precisions * recalls) / np.maximum(denom, 1e-12),
        0.0,
    )
    # The last precision/recall value has no corresponding threshold in sklearn
    best_idx = int(np.argmax(f1_scores[:-1])) if len(thresholds) > 0 else 0
    best_thresh = float(thresholds[best_idx]) if len(thresholds) > 0 else 0.5
    return float(np.clip(best_thresh, 0.1, 0.9))


def train_evaluate_baseline_lr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    random_state: int = 42,
    C: float = 1.0,
    calibrate_threshold: bool = True,
) -> Tuple[LogisticRegression, Dict]:
    model = LogisticRegression(
        C=C,
        max_iter=1000,
        random_state=random_state,
        solver="lbfgs",
    )

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    t_train = time.perf_counter() - t0

    t1 = time.perf_counter()
    y_prob = model.predict_proba(X_test)[:, 1]
    t_infer = time.perf_counter() - t1

    # Standard default threshold evaluation
    y_pred_default = (y_prob >= 0.5).astype(np.int64)

    # Calibrated threshold evaluation
    optimal_thresh = 0.5
    if calibrate_threshold:
        y_prob_tr = model.predict_proba(X_train)[:, 1]
        optimal_thresh = find_optimal_threshold(y_train, y_prob_tr)

    y_pred_calibrated = (y_prob >= optimal_thresh).astype(np.int64)

    metrics = {
        "model_name": "Baseline Logistic Regression",
        "privacy_mechanism": "None (Non-Private)",
        "epsilon": None,
        "delta": None,
        "train_time_sec": round(t_train, 4),
        "inference_time_sec": round(t_infer, 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "default_threshold_0.5": {
            "accuracy": round(float(accuracy_score(y_test, y_pred_default)), 4),
            "precision": round(float(precision_score(y_test, y_pred_default, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred_default, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, y_pred_default, zero_division=0)), 4),
            "confusion_matrix": confusion_matrix(y_test, y_pred_default).tolist(),
        },
        "calibrated_threshold": {
            "threshold_value": round(optimal_thresh, 4),
            "accuracy": round(float(accuracy_score(y_test, y_pred_calibrated)), 4),
            "precision": round(float(precision_score(y_test, y_pred_calibrated, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred_calibrated, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, y_pred_calibrated, zero_division=0)), 4),
            "confusion_matrix": confusion_matrix(y_test, y_pred_calibrated).tolist(),
        },
    }

    return model, metrics


def train_evaluate_baseline_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    random_state: int = 42,
    hidden_layer_sizes: Tuple[int, ...] = (64, 32),
) -> Tuple[MLPClassifier, Dict]:
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=200,
        random_state=random_state,
        early_stopping=True,
    )

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    t_train = time.perf_counter() - t0

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(np.int64)

    metrics = {
        "model_name": "Baseline Multilayer Perceptron (Non-linear)",
        "privacy_mechanism": "None (Non-Private)",
        "train_time_sec": round(t_train, 4),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    return model, metrics


if __name__ == "__main__":
    from data_preparation import get_prepared_data

    X_tr, y_tr, X_te, y_te, _, _ = get_prepared_data()
    _, lr_res = train_evaluate_baseline_lr(X_tr, y_tr, X_te, y_te)
    print("Baseline Logistic Regression (Default vs Calibrated):")
    print("  ROC-AUC:", lr_res["roc_auc"])
    print("  Default 0.5:    Acc:", lr_res["default_threshold_0.5"]["accuracy"], "F1:", lr_res["default_threshold_0.5"]["f1"])
    print("  Calibrated:", lr_res["calibrated_threshold"]["threshold_value"], "Acc:", lr_res["calibrated_threshold"]["accuracy"], "F1:", lr_res["calibrated_threshold"]["f1"])
