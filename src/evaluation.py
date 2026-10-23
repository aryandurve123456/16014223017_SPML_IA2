"""
Comprehensive Evaluation Harness for IA-2.
Executes multi-seed benchmarking across Baseline, DP-ERM, DP-SGD, Local DP,
PPDM, Compressive Privacy, Federated Learning, and Membership Inference Attacks.
Generates master comparison tables and consolidated JSON outputs.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from baseline import train_evaluate_baseline_lr, train_evaluate_baseline_mlp
from compressive_privacy import evaluate_compressive_privacy
from data_preparation import (
    get_prepared_data,
    load_raw_dataset,
    partition_for_membership_inference,
)
from differential_privacy import train_evaluate_dp_model
from dp_sgd import run_dp_sgd_experiments, train_dp_sgd
from federated_learning import (
    partition_data_non_iid,
    train_evaluate_federated_learning,
)
from local_differential_privacy import train_evaluate_ldp_model
from membership_inference import run_mia_benchmark
from privacy_preserving_data_mining import evaluate_ppdm_pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MASTER_METRICS_FILE = RESULTS_DIR / "all_evaluation_metrics.json"
MASTER_TABLE_FILE = RESULTS_DIR / "master_comparison_table.md"


def run_comprehensive_evaluation_pipeline(
    seeds: List[int] = [42, 101, 2024],
    save_json_path: Optional[Path] = MASTER_METRICS_FILE,
    save_table_path: Optional[Path] = MASTER_TABLE_FILE,
) -> Dict:
    print("=" * 80)
    print("  SECURITY AND PRIVACY FOR MACHINE LEARNING: IA-2 BENCHMARK HARNESS")
    print("=" * 80)
    t_start = time.perf_counter()

    X_train, y_train, X_test, y_test, preprocessor, prep_report = get_prepared_data()
    df_train_raw, df_test_raw = load_raw_dataset()

    # 1. Baseline Models
    print("\n[1/8] Benchmarking Non-Private Baseline Models...")
    _, baseline_lr_metrics = train_evaluate_baseline_lr(X_train, y_train, X_test, y_test, random_state=42)
    _, baseline_mlp_metrics = train_evaluate_baseline_mlp(X_train, y_train, X_test, y_test, random_state=42)

    # 2. Central DP-ERM (Output Perturbation) Multi-Seed
    print("[2/8] Benchmarking Central DP-ERM across epsilon levels and seeds...")
    dp_epsilons = [0.1, 0.5, 1.0, 2.0, 5.0]
    dp_erm_results = []
    for eps in dp_epsilons:
        aucs, accs, f1s_cal, precs_cal, recs_cal = [], [], [], [], []
        for seed in seeds:
            _, run_res = train_evaluate_dp_model(
                X_train, y_train, X_test, y_test, epsilon=eps, delta=1e-5, random_state=seed
            )
            aucs.append(run_res["roc_auc"])
            accs.append(run_res["calibrated_threshold"]["accuracy"])
            f1s_cal.append(run_res["calibrated_threshold"]["f1"])
            precs_cal.append(run_res["calibrated_threshold"]["precision"])
            recs_cal.append(run_res["calibrated_threshold"]["recall"])

        dp_erm_results.append({
            "mechanism": "Central DP-ERM (Output Perturbation)",
            "epsilon": eps,
            "delta": 1e-5,
            "roc_auc_mean": round(float(np.mean(aucs)), 4),
            "roc_auc_std": round(float(np.std(aucs)), 4),
            "accuracy_mean": round(float(np.mean(accs)), 4),
            "accuracy_std": round(float(np.std(accs)), 4),
            "f1_mean": round(float(np.mean(f1s_cal)), 4),
            "f1_std": round(float(np.std(f1s_cal)), 4),
            "precision_mean": round(float(np.mean(precs_cal)), 4),
            "recall_mean": round(float(np.mean(recs_cal)), 4),
            "seeds_tested": len(seeds),
        })

    # 3. Central DP-SGD (Opacus Formal RDP Accounting)
    print("[3/8] Benchmarking Central DP-SGD (Opacus) across noise multipliers...")
    dp_sgd_results = run_dp_sgd_experiments(
        X_train, y_train, X_test, y_test, noise_multipliers=[0.8, 1.2, 2.0]
    )

    # 4. Local Differential Privacy (Warner RR + Duchi) Multi-Seed
    print("[4/8] Benchmarking Local Differential Privacy across epsilon levels and seeds...")
    ldp_epsilons = [0.1, 0.5, 1.0, 2.0, 5.0]
    ldp_results = []
    for eps in ldp_epsilons:
        aucs, accs, f1s, precs, recs = [], [], [], [], []
        for seed in seeds:
            run_res = train_evaluate_ldp_model(
                X_train, y_train, X_test, y_test, epsilon=eps, random_state=seed
            )
            aucs.append(run_res["roc_auc"])
            # Use balanced/calibrated threshold
            c = run_res["default_threshold_0.5"]
            accs.append(c["accuracy"])
            f1s.append(c["f1"])
            precs.append(c["precision"])
            recs.append(c["recall"])

        ldp_results.append({
            "mechanism": "Local DP (Warner RR + Duchi)",
            "epsilon": eps,
            "delta": 0.0,
            "roc_auc_mean": round(float(np.mean(aucs)), 4),
            "roc_auc_std": round(float(np.std(aucs)), 4),
            "accuracy_mean": round(float(np.mean(accs)), 4),
            "accuracy_std": round(float(np.std(accs)), 4),
            "f1_mean": round(float(np.mean(f1s)), 4),
            "f1_std": round(float(np.std(f1s)), 4),
            "precision_mean": round(float(np.mean(precs)), 4),
            "recall_mean": round(float(np.mean(recs)), 4),
            "seeds_tested": len(seeds),
        })

    # 5. Privacy-Preserving Data Mining (PPDM k-Anonymity & l-Diversity)
    print("[5/8] Evaluating Privacy-Preserving Data Mining (k-Anonymity & Homogeneity Risk)...")
    ppdm_results = evaluate_ppdm_pipeline(df_train_raw, df_test_raw, k_values=[2, 5, 10, 20, 50])

    # 6. Compressive Privacy (Module 5)
    print("[6/8] Evaluating Compressive Privacy (Utility vs Sensitive Leakage)...")
    train_dedup = df_train_raw.drop_duplicates()
    s_tr = (train_dedup["sex"].str.strip() == "Male").astype(int).to_numpy()
    s_te = (df_test_raw["sex"].str.strip() == "Male").astype(int).to_numpy()
    compressive_results = evaluate_compressive_privacy(
        X_train, y_train, s_tr, X_test, y_test, s_te, dimensions=[2, 4, 8, 16, 24]
    )

    # 7. Federated Learning & Secure Aggregation
    print("[7/8] Simulating Federated Learning (FedAvg 5 Clients) & Pairwise SecAgg Masking...")
    fl_clients = partition_data_non_iid(X_train, y_train, num_clients=5, random_state=42)
    fl_results = train_evaluate_federated_learning(fl_clients, X_test, y_test, rounds=5, random_state=42)
    with open(RESULTS_DIR / "federated_learning_results.json", "w", encoding="utf-8") as f:
        json.dump(fl_results, f, indent=2)

    # 8. Membership Inference Attack (MIA)
    print("[8/8] Executing Controlled Shadow-Model Membership Inference Attack...")
    mia_splits = partition_for_membership_inference(X_train, y_train, X_test, y_test, random_state=42)

    # Fit target models on target training partition
    X_tg_tr, y_tg_tr = mia_splits["target_train"]
    base_target, _ = train_evaluate_baseline_lr(X_tg_tr, y_tg_tr, X_test, y_test, random_state=42)
    dp_target_1 = train_evaluate_dp_model(X_tg_tr, y_tg_tr, X_test, y_test, epsilon=1.0, random_state=42)[0]
    dp_target_05 = train_evaluate_dp_model(X_tg_tr, y_tg_tr, X_test, y_test, epsilon=0.5, random_state=42)[0]
    dp_sgd_target = train_dp_sgd(X_tg_tr, y_tg_tr, X_test, y_test, noise_multiplier=1.2, epochs=3, random_state=42)[0]

    target_models_dict = {
        "Non-Private Baseline LR": base_target,
        "Central DP-ERM (eps=1.0)": dp_target_1,
        "Central DP-ERM (eps=0.5)": dp_target_05,
        "Central DP-SGD (Opacus, noise=1.2)": dp_sgd_target,
    }
    mia_results = run_mia_benchmark(target_models_dict, mia_splits, random_state=42)

    total_time = time.perf_counter() - t_start

    master_output = {
        "benchmarking_metadata": {
            "execution_time_sec": round(total_time, 2),
            "seeds_evaluated": seeds,
            "target_delta": 1e-5,
            "dataset_rows_train": len(X_train),
            "dataset_rows_test": len(X_test),
            "feature_dimension": int(X_train.shape[1]),
        },
        "baseline_lr": baseline_lr_metrics,
        "baseline_mlp": baseline_mlp_metrics,
        "central_dp_erm": dp_erm_results,
        "central_dp_sgd": dp_sgd_results,
        "local_dp": ldp_results,
        "ppdm_k_anonymity": ppdm_results,
        "compressive_privacy": compressive_results,
        "federated_learning": fl_results,
        "membership_inference_attack": mia_results,
    }

    if save_json_path:
        with open(save_json_path, "w", encoding="utf-8") as f:
            json.dump(master_output, f, indent=2)

    # Generate ASCII-safe Master Comparison Markdown Table
    table_lines = [
        "# IA-2 Master Comparison Table: Privacy-Preserving Machine Learning",
        "",
        "| Paradigm / Architecture | Specific Mechanism | Privacy Budget (eps) | Delta | Accuracy (Mean +/- Std) | F1-Score (Mean +/- Std) | ROC-AUC (Mean) | Attack ROC-AUC (MIA Vulnerability) |",
        "|---|---|---|---|---|---|---|---|",
        f"| Baseline Linear | Clean Non-Private LR | N/A | N/A | {baseline_lr_metrics['default_threshold_0.5']['accuracy']:.4f} | {baseline_lr_metrics['default_threshold_0.5']['f1']:.4f} | {baseline_lr_metrics['roc_auc']:.4f} | {mia_results[0]['attack_roc_auc']:.4f} |",
        f"| Baseline Non-Linear | MLPClassifier (64, 32) | N/A | N/A | {baseline_mlp_metrics['accuracy']:.4f} | {baseline_mlp_metrics['f1']:.4f} | {baseline_mlp_metrics['roc_auc']:.4f} | N/A |",
        f"| Federated Learning | FedAvg (5 Clients) + SecAgg | N/A (SecAgg Blinded) | N/A | {fl_results['final_accuracy']:.4f} | {fl_results['final_f1']:.4f} | {fl_results['final_roc_auc']:.4f} | N/A |",
    ]

    for item in dp_erm_results:
        # Match MIA if available
        mia_val = "N/A"
        if item["epsilon"] == 1.0:
            mia_val = f"{mia_results[1]['attack_roc_auc']:.4f}"
        elif item["epsilon"] == 0.5:
            mia_val = f"{mia_results[2]['attack_roc_auc']:.4f}"

        table_lines.append(
            f"| Central DP | DP-ERM Output Perturbation | {item['epsilon']} | {item['delta']} | "
            f"{item['accuracy_mean']:.4f} +/- {item['accuracy_std']:.4f} | "
            f"{item['f1_mean']:.4f} +/- {item['f1_std']:.4f} | "
            f"{item['roc_auc_mean']:.4f} | {mia_val} |"
        )

    for item in dp_sgd_results:
        table_lines.append(
            f"| Central DP | Opacus DP-SGD (noise={item['noise_multiplier']}) | {item['final_epsilon']} | {item['target_delta']} | "
            f"{item['accuracy']:.4f} | {item['f1']:.4f} | {item['roc_auc']:.4f} | {mia_results[3]['attack_roc_auc']:.4f} |"
        )

    for item in ldp_results:
        table_lines.append(
            f"| Local DP | Warner RR + Duchi (Balanced) | {item['epsilon']} | 0.0 | "
            f"{item['accuracy_mean']:.4f} +/- {item['accuracy_std']:.4f} | "
            f"{item['f1_mean']:.4f} +/- {item['f1_std']:.4f} | "
            f"{item['roc_auc_mean']:.4f} | N/A |"
        )

    table_content = "\n".join(table_lines) + "\n"
    if save_table_path:
        with open(save_table_path, "w", encoding="utf-8") as f:
            f.write(table_content)

    print("\n" + "=" * 80)
    print(f"  BENCHMARK COMPLETED IN {total_time:.2f} SECONDS")
    print(f"  Master JSON saved to: {MASTER_METRICS_FILE.name}")
    print(f"  Master Table saved to: {MASTER_TABLE_FILE.name}")
    print("=" * 80)

    return master_output


if __name__ == "__main__":
    run_comprehensive_evaluation_pipeline(seeds=[42, 101])
