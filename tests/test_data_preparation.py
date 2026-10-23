"""
Unit tests for data loading, provenance hashing, leakage prevention, and MIA partitioning.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_preparation import (
    DataPreprocessor,
    compute_sha256,
    get_prepared_data,
    load_raw_dataset,
    partition_for_membership_inference,
)


def test_raw_dataset_and_provenance_hashes():
    df_train, df_test = load_raw_dataset()
    assert len(df_train) == 32561
    assert len(df_test) == 16281

    data_file = PROJECT_ROOT / "data" / "adult.data"
    test_file = PROJECT_ROOT / "data" / "adult.test"

    hash_train = compute_sha256(data_file)
    hash_test = compute_sha256(test_file)

    assert hash_train == "5b00264637dbfec36bdeaab5676b0b309ff9eb788d63554ca0a249491c86603d"
    assert hash_test == "a2a9044bc167a35b2361efbabec64e89d69ce82d9790d2980119aac5fd7e9c05"


def test_preprocessing_integrity_and_bounds():
    X_train, y_train, X_test, y_test, prep, report = get_prepared_data(scaling_range=(-1.0, 1.0))

    assert not np.isnan(X_train).any(), "NaNs found in X_train"
    assert not np.isnan(X_test).any(), "NaNs found in X_test"
    assert not np.isnan(y_train).any()
    assert not np.isnan(y_test).any()

    assert X_train.shape[1] == 105
    assert X_test.shape[1] == 105

    # Numerical features bounds
    X_train_num = X_train[:, :6]
    X_test_num = X_test[:, :6]
    assert np.all(X_train_num >= -1.0 - 1e-7)
    assert np.all(X_train_num <= 1.0 + 1e-7)
    assert np.all(X_test_num >= -1.0 - 1e-7)
    assert np.all(X_test_num <= 1.0 + 1e-7)


def test_leakage_prevention_train_only():
    df_train, df_test = load_raw_dataset()
    prep = DataPreprocessor()
    prep.fit(df_train)

    for col, mode_val in prep.modes.items():
        train_mode = df_train[col].dropna().mode().iloc[0]
        assert mode_val == train_mode, f"Mode mismatch for {col}: {mode_val} vs {train_mode}"


def test_mia_partitions_disjoint():
    X_tr = np.arange(100).reshape(100, 1)
    y_tr = np.zeros(100)
    X_te = np.arange(100, 160).reshape(60, 1)
    y_te = np.zeros(60)

    splits = partition_for_membership_inference(X_tr, y_tr, X_te, y_te)

    # Check zero overlap between target train and shadow train
    tg_tr_set = set(splits["target_train"][0].flatten())
    sh_tr_set = set(splits["shadow_train"][0].flatten())
    assert len(tg_tr_set.intersection(sh_tr_set)) == 0, "Contamination between target train and shadow train"

    # Check zero overlap between target test and shadow test
    tg_te_set = set(splits["target_test"][0].flatten())
    sh_te_set = set(splits["shadow_test"][0].flatten())
    assert len(tg_te_set.intersection(sh_te_set)) == 0, "Contamination between target test and shadow test"
