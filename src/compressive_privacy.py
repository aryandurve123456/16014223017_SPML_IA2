"""
Compressive Privacy Demonstration Module for IA-2 (Module 5).
Investigates information-theoretic privacy via privacy-targeted subspace orthogonalization,
contrasting standard PCA with sensitive-attribute suppression across dimensions.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
COMPRESSIVE_RESULTS_FILE = RESULTS_DIR / "compressive_privacy_results.json"


class CompressivePrivacyProjector:
    """
    Implements a Privacy-Targeted Subspace Projection for Compressive Privacy (Kung, 2017).
    Projects features onto the orthogonal complement of the sensitive direction v_S,
    then applies PCA on the sanitized subspace.
    Note: Empirical information-theoretic heuristic; does NOT provide (epsilon, delta)-DP.
    """

    def __init__(self, n_components: int = 8, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.sensitive_direction_: Optional[np.ndarray] = None
        self.pca_: Optional[PCA] = None

    def fit(self, X: np.ndarray, s: np.ndarray) -> "CompressivePrivacyProjector":
        s_binary = np.asarray(s, dtype=np.int64)
        mean_s1 = np.mean(X[s_binary == 1], axis=0)
        mean_s0 = np.mean(X[s_binary == 0], axis=0)
        v = mean_s1 - mean_s0
        norm_v = np.linalg.norm(v)

        if norm_v > 1e-12:
            self.sensitive_direction_ = v / norm_v
        else:
            self.sensitive_direction_ = np.zeros(X.shape[1])

        # Orthogonal projection: X_ortho = X - (X . v) * v^T
        X_proj = X - np.outer(np.dot(X, self.sensitive_direction_), self.sensitive_direction_)

        n_comp = min(self.n_components, X.shape[1] - 1)
        self.pca_ = PCA(n_components=n_comp, random_state=self.random_state)
        self.pca_.fit(X_proj)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.sensitive_direction_ is None or self.pca_ is None:
            raise RuntimeError("Projector must be fitted before transform.")
        X_proj = X - np.outer(np.dot(X, self.sensitive_direction_), self.sensitive_direction_)
        return self.pca_.transform(X_proj)

    def fit_transform(self, X: np.ndarray, s: np.ndarray) -> np.ndarray:
        self.fit(X, s)
        return self.transform(X)


def evaluate_compressive_privacy(
    X_train: np.ndarray,
    y_train: np.ndarray,
    s_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    s_test: np.ndarray,
    dimensions: List[int] = [2, 4, 8, 16, 24],
    save_path: Optional[Path] = COMPRESSIVE_RESULTS_FILE,
    random_state: int = 42,
) -> Dict:
    # 1. Raw baseline
    clf_y_raw = LogisticRegression(max_iter=500, random_state=random_state)
    clf_y_raw.fit(X_train, y_train)
    raw_y_f1 = float(f1_score(y_test, clf_y_raw.predict(X_test), zero_division=0))

    clf_s_raw = LogisticRegression(max_iter=500, random_state=random_state)
    clf_s_raw.fit(X_train, s_train)
    raw_s_leak = float(balanced_accuracy_score(s_test, clf_s_raw.predict(X_test)))

    pca_records = []
    comp_records = []

    for d in dimensions:
        # Standard PCA (Unaware of privacy)
        pca = PCA(n_components=d, random_state=random_state)
        Z_tr_pca = pca.fit_transform(X_train)
        Z_te_pca = pca.transform(X_test)

        clf_y_pca = LogisticRegression(max_iter=500, random_state=random_state)
        clf_y_pca.fit(Z_tr_pca, y_train)
        y_f1_pca = float(f1_score(y_test, clf_y_pca.predict(Z_te_pca), zero_division=0))

        clf_s_pca = LogisticRegression(max_iter=500, random_state=random_state)
        clf_s_pca.fit(Z_tr_pca, s_train)
        s_leak_pca = float(balanced_accuracy_score(s_test, clf_s_pca.predict(Z_te_pca)))

        pca_records.append({
            "dimension": d,
            "utility_y_f1": round(y_f1_pca, 4),
            "adversary_s_leakage": round(s_leak_pca, 4),
        })

        # Compressive Privacy
        comp_proj = CompressivePrivacyProjector(n_components=d, random_state=random_state)
        Z_tr_comp = comp_proj.fit_transform(X_train, s_train)
        Z_te_comp = comp_proj.transform(X_test)

        clf_y_comp = LogisticRegression(max_iter=500, random_state=random_state)
        clf_y_comp.fit(Z_tr_comp, y_train)
        y_f1_comp = float(f1_score(y_test, clf_y_comp.predict(Z_te_comp), zero_division=0))

        clf_s_comp = LogisticRegression(max_iter=500, random_state=random_state)
        clf_s_comp.fit(Z_tr_comp, s_train)
        s_leak_comp = float(balanced_accuracy_score(s_test, clf_s_comp.predict(Z_te_comp)))

        comp_records.append({
            "dimension": d,
            "utility_y_f1": round(y_f1_comp, 4),
            "adversary_s_leakage": round(s_leak_comp, 4),
        })

    results = {
        "raw_baseline": {
            "representation": "Raw Full Features (105-d)",
            "utility_y_f1": round(raw_y_f1, 4),
            "adversary_s_leakage": round(raw_s_leak, 4),
        },
        "standard_pca": pca_records,
        "compressive_privacy": comp_records,
    }

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    from data_preparation import get_prepared_data, load_raw_dataset

    X_tr, y_tr, X_te, y_te, _, _ = get_prepared_data()
    train_raw, test_raw = load_raw_dataset()
    train_dedup = train_raw.drop_duplicates()
    s_tr = (train_dedup["sex"].str.strip() == "Male").astype(int).to_numpy()
    s_te = (test_raw["sex"].str.strip() == "Male").astype(int).to_numpy()

    res = evaluate_compressive_privacy(X_tr, y_tr, s_tr, X_te, y_te, s_te, dimensions=[4, 8, 16])
    print("Raw Baseline:", res["raw_baseline"])
    print("Standard PCA (d=16):", res["standard_pca"][-1])
    print("Compressive Privacy (d=16):", res["compressive_privacy"][-1])
