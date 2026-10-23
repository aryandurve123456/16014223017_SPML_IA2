"""
Membership Inference Attack (MIA) Module for IA-2.
Implements a controlled shadow-model-based membership inference attack (Shokri et al., 2017; Salem et al., 2018)
with strict dataset separation (zero contamination between target and shadow universes).
Evaluates empirical privacy leakage across Baseline, Central DP, DP-SGD, and Local DP models.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MIA_RESULTS_FILE = RESULTS_DIR / "membership_inference_results.json"


def extract_attack_features(model: Any, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
    elif hasattr(model, "forward"):  # PyTorch model
        import torch
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(X, dtype=torch.float32))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
    else:
        raise ValueError("Unsupported model type for attack feature extraction.")

    # Prevent numerical log(0) issues
    probs_clipped = np.clip(probs, 1e-12, 1.0 - 1e-12)

    # 1. Probability assigned to true class
    n = len(y)
    true_class_probs = probs_clipped[np.arange(n), y]

    # 2. Negative log-likelihood (cross-entropy loss)
    nll_loss = -np.log(true_class_probs)

    # 3. Prediction entropy
    entropy = -np.sum(probs_clipped * np.log(probs_clipped), axis=1)

    # 4. Maximum predicted probability (confidence)
    confidence = np.max(probs_clipped, axis=1)

    # Combine into feature matrix
    attack_X = np.column_stack([probs_clipped, true_class_probs, nll_loss, entropy, confidence])
    return attack_X


def train_attack_model(
    shadow_model: Any,
    X_shadow_train: np.ndarray,
    y_shadow_train: np.ndarray,
    X_shadow_test: np.ndarray,
    y_shadow_test: np.ndarray,
    random_state: int = 42,
) -> LogisticRegression:
    # Shadow members (label = 1)
    feat_members = extract_attack_features(shadow_model, X_shadow_train, y_shadow_train)
    # Shadow non-members (label = 0)
    feat_nonmembers = extract_attack_features(shadow_model, X_shadow_test, y_shadow_test)

    # Balance shadow members and non-members for unbiased attack training (1:1 ratio)
    n_sh_per_class = min(len(feat_members), len(feat_nonmembers))
    feat_members = feat_members[:n_sh_per_class]
    feat_nonmembers = feat_nonmembers[:n_sh_per_class]

    label_members = np.ones(n_sh_per_class, dtype=np.int64)
    label_nonmembers = np.zeros(n_sh_per_class, dtype=np.int64)

    attack_X = np.vstack([feat_members, feat_nonmembers])
    attack_y = np.concatenate([label_members, label_nonmembers])

    attack_clf = LogisticRegression(max_iter=1000, random_state=random_state)
    attack_clf.fit(attack_X, attack_y)
    return attack_clf


def evaluate_membership_inference(
    attack_model: LogisticRegression,
    target_model: Any,
    X_target_train: np.ndarray,
    y_target_train: np.ndarray,
    X_target_test: np.ndarray,
    y_target_test: np.ndarray,
    model_name: str = "Target Model",
) -> Dict:
    # Target members (label = 1)
    feat_target_members = extract_attack_features(target_model, X_target_train, y_target_train)
    # Target non-members (label = 0)
    feat_target_nonmembers = extract_attack_features(target_model, X_target_test, y_target_test)

    # Balance evaluation set (1:1 ratio) to ensure fair chance baseline = 0.50
    n_eval_per_class = min(len(feat_target_members), len(feat_target_nonmembers))
    feat_target_members = feat_target_members[:n_eval_per_class]
    feat_target_nonmembers = feat_target_nonmembers[:n_eval_per_class]

    label_target_members = np.ones(n_eval_per_class, dtype=np.int64)
    label_target_nonmembers = np.zeros(n_eval_per_class, dtype=np.int64)

    eval_X = np.vstack([feat_target_members, feat_target_nonmembers])
    eval_y = np.concatenate([label_target_members, label_target_nonmembers])

    t0 = time.perf_counter()
    pred_probs = attack_model.predict_proba(eval_X)[:, 1]
    preds = (pred_probs >= 0.5).astype(np.int64)
    attack_time = time.perf_counter() - t0

    acc = float(accuracy_score(eval_y, preds))
    prec = float(precision_score(eval_y, preds, zero_division=0))
    rec = float(recall_score(eval_y, preds, zero_division=0))
    f1 = float(f1_score(eval_y, preds, zero_division=0))
    auc = float(roc_auc_score(eval_y, pred_probs))
    cm = confusion_matrix(eval_y, preds).tolist()

    # Advantage: True Positive Rate - False Positive Rate
    tn, fp, fn, tp = confusion_matrix(eval_y, preds).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    mia_advantage = float(tpr - fpr)

    return {
        "evaluated_model": model_name,
        "attack_accuracy": round(acc, 4),
        "attack_precision": round(prec, 4),
        "attack_recall": round(rec, 4),
        "attack_f1": round(f1, 4),
        "attack_roc_auc": round(auc, 4),
        "mia_advantage": round(mia_advantage, 4),
        "confusion_matrix": cm,
        "sample_counts": {
            "members": len(feat_target_members),
            "non_members": len(feat_target_nonmembers),
        },
    }


def run_mia_benchmark(
    target_models_dict: Dict[str, Any],
    mia_splits: Dict[str, Tuple[np.ndarray, np.ndarray]],
    save_path: Optional[Path] = MIA_RESULTS_FILE,
    random_state: int = 42,
) -> List[Dict]:
    X_sh_tr, y_sh_tr = mia_splits["shadow_train"]
    X_sh_te, y_sh_te = mia_splits["shadow_test"]
    X_tg_tr, y_tg_tr = mia_splits["target_train"]
    X_tg_te, y_tg_te = mia_splits["target_test"]

    # 1. Train Shadow Model on shadow training partition
    shadow_model = LogisticRegression(max_iter=1000, random_state=random_state)
    shadow_model.fit(X_sh_tr, y_sh_tr)

    # 2. Train Attack Model on shadow member vs non-member behavior
    attack_classifier = train_attack_model(
        shadow_model, X_sh_tr, y_sh_tr, X_sh_te, y_sh_te, random_state=random_state
    )

    # 3. Evaluate attack classifier on all target models
    results = []
    for model_name, target_model in target_models_dict.items():
        res = evaluate_membership_inference(
            attack_classifier,
            target_model,
            X_tg_tr,
            y_tg_tr,
            X_tg_te,
            y_tg_te,
            model_name=model_name,
        )
        results.append(res)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    from baseline import train_evaluate_baseline_lr
    from data_preparation import get_prepared_data, partition_for_membership_inference
    from differential_privacy import DifferentiallyPrivateLogisticRegression
    from sklearn.neural_network import MLPClassifier

    X_tr, y_tr, X_te, y_te, _, _ = get_prepared_data()
    splits = partition_for_membership_inference(X_tr, y_tr, X_te, y_te)

    # 1. Overfitted MLP (demonstrates membership leakage due to empirical generalization gap)
    overfit_mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, alpha=0.0, random_state=42)
    overfit_mlp.fit(splits["target_train"][0][:3000], splits["target_train"][1][:3000])

    # 2. Regularized Non-Private Baseline LR
    base_m, _ = train_evaluate_baseline_lr(splits["target_train"][0], splits["target_train"][1], X_te, y_te)
    
    # 3. Target Central DP model
    dp_m = DifferentiallyPrivateLogisticRegression(epsilon=1.0)
    dp_m.fit(splits["target_train"][0], splits["target_train"][1])

    eval_models = {
        "High-Capacity Overfitted MLP (Large Gap)": overfit_mlp,
        "Non-Private Regularized Baseline LR": base_m,
        "Central DP-ERM (eps=1.0)": dp_m,
    }

    mia_eval = run_mia_benchmark(eval_models, splits)
    print("MIA Benchmark Results:")
    for r in mia_eval:
        print(f"  {r['evaluated_model']}: Attack Acc={r['attack_accuracy']}, ROC-AUC={r['attack_roc_auc']}, Advantage={r['mia_advantage']}")
