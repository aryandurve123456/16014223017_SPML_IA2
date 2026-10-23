"""
Privacy-Preserving Data Mining (PPDM) Module for IA-2.
Implements k-Anonymity via Generalization and Suppression, evaluates l-Diversity,
and demonstrates the Homogeneity Attack vulnerability on the UCI Adult dataset.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PPDM_RESULTS_FILE = RESULTS_DIR / "ppdm_results.json"


def generalize_attributes(df: pd.DataFrame) -> pd.DataFrame:
    df_gen = df.copy()

    # Age generalization into decade brackets
    if "age" in df_gen.columns:
        bins = [0, 20, 30, 40, 50, 60, 70, 120]
        labels = ["<20", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
        df_gen["age_gen"] = pd.cut(df_gen["age"], bins=bins, labels=labels, right=False).astype(str)

    # Education generalization into 5 educational tiers
    if "education" in df_gen.columns:
        edu_map = {
            "Preschool": "Less-than-HS",
            "1st-4th": "Less-than-HS",
            "5th-6th": "Less-than-HS",
            "7th-8th": "Less-than-HS",
            "9th": "Less-than-HS",
            "10th": "Less-than-HS",
            "11th": "Less-than-HS",
            "12th": "Less-than-HS",
            "HS-grad": "HS-grad",
            "Some-college": "Some-college",
            "Assoc-voc": "Some-college",
            "Assoc-acdm": "Some-college",
            "Bachelors": "Bachelors",
            "Masters": "Higher-Degree",
            "Prof-school": "Higher-Degree",
            "Doctorate": "Higher-Degree",
        }
        df_gen["education_gen"] = df_gen["education"].map(lambda x: edu_map.get(x, "Other"))

    # Native country generalization into regions
    if "native-country" in df_gen.columns:
        df_gen["country_gen"] = df_gen["native-country"].apply(
            lambda x: "United-States" if x == "United-States" else "Non-US"
        )

    return df_gen


def enforce_k_anonymity(
    df: pd.DataFrame,
    qi_columns: List[str],
    k: int,
) -> Tuple[pd.DataFrame, Dict]:
    if k < 1:
        raise ValueError("k must be >= 1")

    grouped = df.groupby(qi_columns, observed=True)
    eq_class_sizes = grouped.size()
    valid_classes = eq_class_sizes[eq_class_sizes >= k].index

    if len(qi_columns) == 1:
        mask = df[qi_columns[0]].isin(valid_classes)
    else:
        index_df = df.set_index(qi_columns)
        mask = index_df.index.isin(valid_classes)

    df_anonymized = df[mask].copy()

    total_records = len(df)
    retained_records = len(df_anonymized)
    suppressed_records = total_records - retained_records
    suppression_rate = (suppressed_records / total_records) * 100.0 if total_records > 0 else 0.0

    stats = {
        "k_value": k,
        "qi_columns": qi_columns,
        "total_records": total_records,
        "retained_records": retained_records,
        "suppressed_records": suppressed_records,
        "suppression_rate_pct": round(suppression_rate, 2),
        "total_equivalence_classes": len(eq_class_sizes),
        "valid_equivalence_classes": len(valid_classes),
    }

    return df_anonymized, stats


def evaluate_l_diversity(
    df: pd.DataFrame,
    qi_columns: List[str],
    sensitive_column: str,
    l: int = 2,
) -> Dict:
    grouped = df.groupby(qi_columns, observed=True)
    distinct_sensitive_counts = grouped[sensitive_column].nunique()
    
    violating_classes = distinct_sensitive_counts[distinct_sensitive_counts < l]
    total_classes = len(distinct_sensitive_counts)
    violating_count = len(violating_classes)
    compliance_rate = ((total_classes - violating_count) / total_classes * 100.0) if total_classes > 0 else 0.0

    # Homogeneity check: distinct count = 1
    homogeneity_vulnerable_count = int((distinct_sensitive_counts == 1).sum())

    return {
        "sensitive_column": sensitive_column,
        "l_threshold": l,
        "total_equivalence_classes": total_classes,
        "violating_classes": violating_count,
        "compliance_rate_pct": round(compliance_rate, 2),
        "homogeneity_vulnerable_classes": homogeneity_vulnerable_count,
        "homogeneity_risk_pct": round((homogeneity_vulnerable_count / total_classes * 100.0), 2) if total_classes > 0 else 0.0,
    }


def evaluate_ppdm_pipeline(
    df_train_raw: pd.DataFrame,
    df_test_raw: pd.DataFrame,
    k_values: List[int] = [2, 5, 10, 20, 50],
    save_path: Optional[Path] = PPDM_RESULTS_FILE,
    random_state: int = 42,
) -> List[Dict]:
    train_gen = generalize_attributes(df_train_raw)
    test_gen = generalize_attributes(df_test_raw)

    qis = ["age_gen", "education_gen", "marital-status", "sex", "country_gen"]
    results = []
    y_test = (test_gen["income"].astype(str).str.rstrip(".") == ">50K").astype(int)

    for k in k_values:
        anon_train, stats = enforce_k_anonymity(train_gen, qi_columns=qis, k=k)
        l_div = evaluate_l_diversity(anon_train, qi_columns=qis, sensitive_column="income", l=2)

        y_train = (anon_train["income"].astype(str).str.rstrip(".") == ">50K").astype(int)
        
        enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        X_tr_enc = enc.fit_transform(anon_train[qis].fillna("Unknown"))
        X_te_enc = enc.transform(test_gen[qis].fillna("Unknown"))

        clf = LogisticRegression(max_iter=500, random_state=random_state)
        clf.fit(X_tr_enc, y_train)
        preds = clf.predict(X_te_enc)

        acc = round(float(accuracy_score(y_test, preds)), 4)
        f1 = round(float(f1_score(y_test, preds, zero_division=0)), 4)

        results.append({
            "k": k,
            "suppression_rate_pct": stats["suppression_rate_pct"],
            "retained_records": stats["retained_records"],
            "accuracy": acc,
            "f1": f1,
            "homogeneity_risk_pct": l_div["homogeneity_risk_pct"],
            "distinct_l_compliance_pct": l_div["compliance_rate_pct"],
            "homogeneity_vulnerable_classes": l_div["homogeneity_vulnerable_classes"],
        })

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    from data_preparation import load_raw_dataset

    train_df, test_df = load_raw_dataset()
    res = evaluate_ppdm_pipeline(train_df, test_df)
    print("PPDM Results across k:")
    for r in res:
        print(f"  k={r['k']}: Suppr={r['suppression_rate_pct']}%, Acc={r['accuracy']}, F1={r['f1']}, Homogeneity Risk={r['homogeneity_risk_pct']}%")
