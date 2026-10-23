"""
Data Preparation and Provenance Module for IA-2.
Handles reproducible acquisition, SHA-256 cryptographic hashing, schema cleaning,
strict train-only preprocessing, and leakage-free dataset partitioning for target and shadow models.
"""

import hashlib
import io
import json
import os
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Base directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

UCI_ADULT_URL = "https://archive.ics.uci.edu/static/public/2/adult.zip"
ADULT_DATA_FILE = DATA_DIR / "adult.data"
ADULT_TEST_FILE = DATA_DIR / "adult.test"
ADULT_NAMES_FILE = DATA_DIR / "adult.names"
PROVENANCE_FILE = RESULTS_DIR / "dataset_provenance.json"
PREPROCESSING_REPORT_FILE = RESULTS_DIR / "preprocessing_report.json"

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]

NUMERICAL_FEATURES = [
    "age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"
]

CATEGORICAL_FEATURES = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country"
]

TARGET_COLUMN = "income"
TARGET_MAPPING = {"<=50K": 0, ">50K": 1}
QUASI_IDENTIFIERS = ["age", "workclass", "education", "marital-status", "sex", "native-country"]


def compute_sha256(filepath: Path) -> str:
    if not filepath.exists():
        return "FILE_NOT_FOUND"
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_adult_dataset(dest_dir: Path = DATA_DIR) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    if ADULT_DATA_FILE.exists() and ADULT_TEST_FILE.exists():
        return

    try:
        req = urllib.request.Request(UCI_ADULT_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            zip_bytes = resp.read()
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
                z.extractall(dest_dir)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download UCI Adult dataset from {UCI_ADULT_URL}.\n"
            f"Error details: {exc}\n"
            "Please download 'adult.data' and 'adult.test' manually from https://archive.ics.uci.edu/dataset/2/adult "
            f"and place them into {dest_dir.resolve()}."
        ) from exc


def load_raw_dataset(data_dir: Path = DATA_DIR) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not (data_dir / "adult.data").exists() or not (data_dir / "adult.test").exists():
        download_adult_dataset(data_dir)

    df_train = pd.read_csv(
        data_dir / "adult.data",
        names=COLUMNS,
        sep=r",\s*",
        engine="python",
        na_values="?",
        skipinitialspace=True,
    )

    df_test = pd.read_csv(
        data_dir / "adult.test",
        names=COLUMNS,
        sep=r",\s*",
        engine="python",
        skiprows=1,
        na_values="?",
        skipinitialspace=True,
    )

    df_test["income"] = df_test["income"].astype(str).str.rstrip(".")
    return df_train, df_test


def generate_provenance_record(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    save_path: Optional[Path] = PROVENANCE_FILE,
) -> Dict:
    df_comb = pd.concat([df_train, df_test], axis=0, ignore_index=True)

    provenance = {
        "dataset_name": "UCI Adult Census Income Dataset",
        "official_url": "https://archive.ics.uci.edu/dataset/2/adult",
        "doi": "https://doi.org/10.24432/C5XW20",
        "citation": "Kohavi, R. (1996). Scaling Up the Accuracy of Naive-Bayes Classifiers: A Decision-Tree Hybrid. KDD-96.",
        "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
        "sha256_checksums": {
            "adult.data": compute_sha256(ADULT_DATA_FILE),
            "adult.test": compute_sha256(ADULT_TEST_FILE),
            "adult.names": compute_sha256(ADULT_NAMES_FILE),
        },
        "file_sizes_bytes": {
            "adult.data": ADULT_DATA_FILE.stat().st_size if ADULT_DATA_FILE.exists() else 0,
            "adult.test": ADULT_TEST_FILE.stat().st_size if ADULT_TEST_FILE.exists() else 0,
        },
        "raw_record_counts": {
            "train_rows": len(df_train),
            "test_rows": len(df_test),
            "total_rows": len(df_comb),
        },
        "attributes": {
            "total_columns": len(df_comb.columns),
            "continuous_count": len(NUMERICAL_FEATURES),
            "categorical_count": len(CATEGORICAL_FEATURES),
            "target": TARGET_COLUMN,
        },
        "target_distribution": {
            "train": {str(k): int(v) for k, v in df_train["income"].value_counts().items()},
            "test": {str(k): int(v) for k, v in df_test["income"].value_counts().items()},
            "combined": {str(k): int(v) for k, v in df_comb["income"].value_counts().items()},
        },
        "missing_values": {
            "train": {k: int(v) for k, v in df_train.isnull().sum().items() if v > 0},
            "test": {k: int(v) for k, v in df_test.isnull().sum().items() if v > 0},
        },
        "duplicates": {
            "train_duplicates": int(df_train.duplicated().sum()),
            "test_duplicates": int(df_test.duplicated().sum()),
        },
    }

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(provenance, f, indent=2)

    return provenance


class DataPreprocessor:
    def __init__(
        self,
        scaling_range: Tuple[float, float] = (-1.0, 1.0),
        handle_duplicates: bool = True,
    ):
        self.scaling_range = scaling_range
        self.handle_duplicates = handle_duplicates
        self.scaler = MinMaxScaler(feature_range=scaling_range)
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.modes: Dict[str, str] = {}
        self.feature_names_out_ = []
        self.is_fitted = False

    def fit(self, df_train: pd.DataFrame) -> "DataPreprocessor":
        df = df_train.copy()
        if self.handle_duplicates:
            df = df.drop_duplicates()

        # Mode calculation strictly on training set
        for col in CATEGORICAL_FEATURES:
            m = df[col].dropna().mode()
            self.modes[col] = m.iloc[0] if not m.empty else "Unknown"

        # Impute
        df_imputed = df.copy()
        for col in CATEGORICAL_FEATURES:
            df_imputed[col] = df_imputed[col].fillna(self.modes[col])

        # Fit numerical scaler and categorical encoder
        self.scaler.fit(df_imputed[NUMERICAL_FEATURES])
        self.encoder.fit(df_imputed[CATEGORICAL_FEATURES])

        self.feature_names_out_ = (
            list(NUMERICAL_FEATURES)
            + list(self.encoder.get_feature_names_out(CATEGORICAL_FEATURES))
        )
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform.")

        df_proc = df.copy()
        for col in CATEGORICAL_FEATURES:
            df_proc[col] = df_proc[col].fillna(self.modes[col])

        X_num = self.scaler.transform(df_proc[NUMERICAL_FEATURES])
        X_num = np.clip(X_num, self.scaling_range[0], self.scaling_range[1])
        X_cat = self.encoder.transform(df_proc[CATEGORICAL_FEATURES])

        X = np.hstack([X_num, X_cat])

        y = None
        if TARGET_COLUMN in df_proc.columns:
            cleaned = df_proc[TARGET_COLUMN].astype(str).str.rstrip(".")
            y = cleaned.map(TARGET_MAPPING).to_numpy(dtype=np.int64)

        return X, y

    def fit_transform(self, df_train: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        self.fit(df_train)
        df_for_tf = df_train.drop_duplicates() if self.handle_duplicates else df_train
        return self.transform(df_for_tf)


def get_prepared_data(
    data_dir: Path = DATA_DIR,
    scaling_range: Tuple[float, float] = (-1.0, 1.0),
    save_reports: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, DataPreprocessor, Dict]:
    df_train_raw, df_test_raw = load_raw_dataset(data_dir)
    provenance = generate_provenance_record(df_train_raw, df_test_raw)

    preprocessor = DataPreprocessor(scaling_range=scaling_range, handle_duplicates=True)
    X_train, y_train = preprocessor.fit_transform(df_train_raw)
    X_test, y_test = preprocessor.transform(df_test_raw)

    report = {
        "raw_train_shape": list(df_train_raw.shape),
        "raw_test_shape": list(df_test_raw.shape),
        "processed_train_shape": list(X_train.shape),
        "processed_test_shape": list(X_test.shape),
        "feature_count_total": int(X_train.shape[1]),
        "numerical_features_count": len(NUMERICAL_FEATURES),
        "categorical_encoded_count": int(X_train.shape[1] - len(NUMERICAL_FEATURES)),
        "scaling_range": list(scaling_range),
        "positive_class_ratio_train": float(np.mean(y_train)),
        "positive_class_ratio_test": float(np.mean(y_test)),
        "train_only_fitting": True,
        "duplicates_removed_from_train": int(df_train_raw.duplicated().sum()),
    }

    if save_reports:
        with open(PREPROCESSING_REPORT_FILE, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    return X_train, y_train, X_test, y_test, preprocessor, report


def partition_for_membership_inference(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    random_state: int = 42,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Constructs four strictly non-overlapping partitions for Membership Inference Attack (MIA):
    1. Target Train (Members for Target Model): 50% of X_train
    2. Target Test (Non-Members for Target Model): 50% of X_test
    3. Shadow Train (Members for Shadow Model): remaining 50% of X_train
    4. Shadow Test (Non-Members for Shadow Model): remaining 50% of X_test
    Guarantees absolute data hygiene with zero overlap across target and shadow universes.
    """
    rng = np.random.default_rng(random_state)
    n_tr = len(X_train)
    tr_indices = rng.permutation(n_tr)
    split_tr = n_tr // 2

    target_train_idx = tr_indices[:split_tr]
    shadow_train_idx = tr_indices[split_tr:]

    n_te = len(X_test)
    te_indices = rng.permutation(n_te)
    split_te = n_te // 2

    target_test_idx = te_indices[:split_te]
    shadow_test_idx = te_indices[split_te:]

    return {
        "target_train": (X_train[target_train_idx], y_train[target_train_idx]),
        "target_test": (X_test[target_test_idx], y_test[target_test_idx]),
        "shadow_train": (X_train[shadow_train_idx], y_train[shadow_train_idx]),
        "shadow_test": (X_test[shadow_test_idx], y_test[shadow_test_idx]),
    }


if __name__ == "__main__":
    X_tr, y_tr, X_te, y_te, prep, rep = get_prepared_data()
    print("Data preparation completed.")
    print(f"X_train: {X_tr.shape}, X_test: {X_te.shape}")
    print(f"Dataset SHA-256 (adult.data): {compute_sha256(ADULT_DATA_FILE)}")
    mia_splits = partition_for_membership_inference(X_tr, y_tr, X_te, y_te)
    print("MIA Partitions generated:")
    for k, v in mia_splits.items():
        print(f"  {k}: X shape {v[0].shape}, y mean {np.mean(v[1]):.4f}")
