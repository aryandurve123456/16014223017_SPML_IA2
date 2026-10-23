"""
Differentially Private Stochastic Gradient Descent (DP-SGD) Module for IA-2.
Implements formal DP-SGD training using PyTorch and Opacus (Rényi Differential Privacy Accountant),
recording clipping norm, noise multiplier, sampling ratio, and exact epsilon spent per epoch.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from opacus import PrivacyEngine
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DP_SGD_ACCOUNTING_FILE = RESULTS_DIR / "dp_sgd_accounting.json"


class LogisticRegressionPyTorch(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def train_dp_sgd(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    noise_multiplier: float = 1.0,
    max_grad_norm: float = 1.0,
    batch_size: int = 256,
    epochs: int = 5,
    lr: float = 0.05,
    delta: float = 1e-5,
    random_state: int = 42,
) -> Tuple[LogisticRegressionPyTorch, Dict]:
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    n_samples, n_features = X_train.shape
    tensor_x_tr = torch.tensor(X_train, dtype=torch.float32)
    tensor_y_tr = torch.tensor(y_train, dtype=torch.long)
    dataset_tr = TensorDataset(tensor_x_tr, tensor_y_tr)
    loader_tr = DataLoader(dataset_tr, batch_size=batch_size, shuffle=True)

    tensor_x_te = torch.tensor(X_test, dtype=torch.float32)

    model = LogisticRegressionPyTorch(in_features=n_features)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    privacy_engine = PrivacyEngine()
    model, optimizer, loader_tr = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=loader_tr,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )

    epoch_accounting: List[Dict] = []
    t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x_b, y_b in loader_tr:
            optimizer.zero_grad()
            out = model(x_b)
            loss = criterion(out, y_b)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        current_eps = float(privacy_engine.get_epsilon(delta=delta))
        epoch_accounting.append({
            "epoch": epoch,
            "epsilon_spent": round(current_eps, 4),
            "delta": delta,
            "avg_loss": round(total_loss / len(loader_tr), 4),
        })

    train_time = time.perf_counter() - t0

    # Test evaluation
    model.eval()
    with torch.no_grad():
        logits_te = model(tensor_x_te)
        probs_te = torch.softmax(logits_te, dim=1)[:, 1].cpu().numpy()
        preds_te = (probs_te >= 0.5).astype(np.int64)

    final_eps = float(privacy_engine.get_epsilon(delta=delta))

    results = {
        "model_name": f"DP-SGD Logistic Regression (noise={noise_multiplier})",
        "paradigm": "Central DP (Differentially Private SGD via Opacus)",
        "noise_multiplier": noise_multiplier,
        "max_grad_norm": max_grad_norm,
        "batch_size": batch_size,
        "epochs": epochs,
        "sampling_ratio": round(batch_size / n_samples, 6),
        "target_delta": delta,
        "final_epsilon": round(final_eps, 4),
        "train_time_sec": round(train_time, 4),
        "accuracy": round(float(accuracy_score(y_test, preds_te)), 4),
        "precision": round(float(precision_score(y_test, preds_te, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, preds_te, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, preds_te, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probs_te)), 4),
        "confusion_matrix": confusion_matrix(y_test, preds_te).tolist(),
        "epoch_accounting": epoch_accounting,
    }

    return model, results


def run_dp_sgd_experiments(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    noise_multipliers: List[float] = [0.8, 1.2, 2.0],
    save_path: Optional[Path] = DP_SGD_ACCOUNTING_FILE,
) -> List[Dict]:
    experiments = []
    for nm in noise_multipliers:
        _, res = train_dp_sgd(
            X_train, y_train, X_test, y_test,
            noise_multiplier=nm,
            max_grad_norm=1.0,
            batch_size=256,
            epochs=5,
        )
        experiments.append(res)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(experiments, f, indent=2)

    return experiments


if __name__ == "__main__":
    from data_preparation import get_prepared_data

    X_tr, y_tr, X_te, y_te, _, _ = get_prepared_data()
    print("Running Opacus DP-SGD experiments...")
    res_list = run_dp_sgd_experiments(X_tr, y_tr, X_te, y_te, noise_multipliers=[0.8, 1.5])
    for r in res_list:
        print(f"Noise {r['noise_multiplier']}: Final eps={r['final_epsilon']} (delta=1e-5) | Acc={r['accuracy']}, ROC-AUC={r['roc_auc']}, F1={r['f1']}")
