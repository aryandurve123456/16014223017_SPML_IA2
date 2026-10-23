"""
Unit tests verifying that all required IA-2 result files, tables, and figures exist and are non-empty.
"""

from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


def test_generated_results_and_figures_exist():
    expected_files = [
        "dataset_provenance.json",
        "preprocessing_report.json",
        "all_evaluation_metrics.json",
        "master_comparison_table.md",
        "dp_sgd_accounting.json",
        "federated_learning_results.json",
        "membership_inference_results.json",
        "compressive_privacy_results.json",
        "ppdm_results.json",
        "fig1_privacy_utility_tradeoff.png",
        "fig2_confusion_matrices.png",
        "fig3_membership_inference_attack.png",
        "fig4_dp_sgd_accounting.png",
        "fig5_federated_learning_convergence.png",
        "fig6_compressive_privacy_tradeoff.png",
        "fig7_k_anonymity_tradeoff.png",
    ]

    for fname in expected_files:
        fpath = RESULTS_DIR / fname
        assert fpath.exists(), f"Missing expected output artifact: {fpath}"
        assert fpath.stat().st_size > 0, f"Output artifact is empty: {fpath}"
