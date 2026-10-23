"""
Master End-to-End Orchestrator for IA-2: Implementation, Evaluation and Technical Outcomes.
Executes the full pipeline:
  Step 1: Dataset acquisition, SHA-256 verification, and provenance reporting
  Step 2: Preprocessing and leakage-free data partitioning
  Step 3: Baseline models (Logistic Regression and Multilayer Perceptron)
  Step 4: Central Differential Privacy (DP-ERM and Opacus DP-SGD with RDP accounting)
  Step 5: Local Differential Privacy (Warner RR, Duchi, Piecewise, and calibrated thresholding)
  Step 6: Privacy-Preserving Data Mining (k-anonymity, l-diversity, homogeneity risk audit)
  Step 7: Compressive Privacy (Privacy-targeted orthogonal projection vs standard PCA)
  Step 8: Federated Learning simulation (FedAvg 5 clients + Pairwise SecAgg masking)
  Step 9: Controlled Shadow-Model Membership Inference Attack (MIA)
  Step 10: Multi-seed benchmarking tables and publication figures generation
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluation import run_comprehensive_evaluation_pipeline
from visualisation import generate_all_visualisations


def main() -> None:
    print("=" * 85)
    print("  SECURITY AND PRIVACY FOR MACHINE LEARNING: IA-2 MASTER PIPELINE")
    print("  Privacy-Preserving Machine Learning Using Differential Privacy and Local DP")
    print("  Evaluating the Privacy-Utility Trade-off in Income Classification")
    print("=" * 85)

    start_time = time.perf_counter()

    # Run comprehensive evaluation harness (steps 1 through 9)
    master_results = run_comprehensive_evaluation_pipeline(seeds=[42, 101, 2024])

    # Generate all publication plots (step 10)
    print("\nGenerating publication-quality figures...")
    figures = generate_all_visualisations(master_results, save_dir=PROJECT_ROOT / "results")
    for fig in figures:
        print(f"  Saved figure: {fig.name} ({fig.stat().st_size:,} bytes)")

    total_time = time.perf_counter() - start_time
    print("\n" + "=" * 85)
    print(f"  MASTER PIPELINE EXECUTION COMPLETED SUCCESSFULLY IN {total_time:.2f} SECONDS")
    print(f"  All outputs generated in: {(PROJECT_ROOT / 'results').resolve()}")
    print("=" * 85)


if __name__ == "__main__":
    main()
