# IA-2 Official Rubric Compliance Matrix

This document maps each requirement from the official **IA-2 Marking Rubric** (Security and Privacy for Machine Learning) to the exact implementation code, generated results, and report sections.

---

| Rubric Stage & Marks | Official Stage Requirement | Evidence Submitted | Project File & Report Location | Verification Status |
|---|---|---|---|---|
| **Stage 1: Problem Refinement and Objectives** (4 Marks) | Review IA-1 findings and refine problem statement, scope, objectives, research questions, or technical requirements. Explain changes made in response to feasibility, preliminary results, or limitations. | Revised problem statement, final objectives, research questions, and explicit justification of changes from IA-1. | `docs/IA2_Comprehensive_Technical_Report.docx` §2, §3, §4; `README.md` §1 | **Fully Completed (Verified)** |
| **Stage 2: Integration of Course Concepts** (4 Marks) | Show how concepts from all five modules contribute to the solution. Implement or demonstrate appropriate techniques where relevant. Justify treatment through conceptual analysis or controlled demonstration. | Final module-mapping table, complete architecture/workflow diagram, concept-integration explanations, and code evidence for all 5 modules. | `docs/IA2_Comprehensive_Technical_Report.docx` §8, §9, §10; `src/` modules; 7 publication figures | **Fully Completed (Verified)** |
| **Stage 3: Dataset Finalisation and Individual Implementation** (4 Marks) | Complete required data preparation. Document source, collection date/method, license/usage terms, original/final dataset sizes, preprocessing, version information, and file hash. Identify student's own contribution. | Complete provenance record, SHA-256 hashes, preprocessing scripts with train-only fitting, and Individual Contribution Statement. | `results/dataset_provenance.json`; `results/preprocessing_report.json`; `src/data_preparation.py`; Report §7, §19 | **Fully Completed (Verified)** |
| **Stage 4: Evaluation, Comparison, and Critical Analysis** (4 Marks) | Evaluate the work using suitable metrics. Compare with baseline, alternative method, or parameter settings (accuracy, precision, recall, F1, privacy budget, MIA advantage, SecAgg error). Discuss limitations and validity threats. | Consolidated master comparison table, multi-seed statistical aggregation (means ± std), Opacus DP-SGD RDP accounting, MIA advantage metrics, and SecAgg error analysis. | `results/master_comparison_table.md`; `results/all_evaluation_metrics.json`; Report §12, §13, §14, §15, §17, §18 | **Fully Completed (Verified)** |
| **Stage 5: Final Technical Documentation and Research/IP Development** (4 Marks) | Prepare a complete technical report covering problem, background, methodology, module integration, provenance, implementation, results, limitations, and conclusions. Include reproducibility instructions. | 22-section comprehensive technical report in DOCX format (1.55 MB) with embedded figures and tables; complete execution guide; 12 passing automated unit tests. | `docs/IA2_Comprehensive_Technical_Report.docx`; `run_all.py`; `tests/`; `README.md` | **Fully Completed (Verified)** |

---

## Breakdown of Key Technical Extensions in IA-2

1. **Extension A: Controlled Membership Inference Attack (MIA)**:
   - File: `src/membership_inference.py`, `results/membership_inference_results.json`, Figure 3.
   - Status: Fully executed with shadow models and 4-way disjoint data partitioning.
2. **Extension B: DP-SGD with Opacus RDP Accounting**:
   - File: `src/dp_sgd.py`, `results/dp_sgd_accounting.json`, Figure 4.
   - Status: Fully executed with PyTorch 2.14 and Opacus 1.6.0 across multiple noise multipliers.
3. **Extension C: Federated Learning & Pairwise SecAgg Simulation**:
   - File: `src/federated_learning.py`, `results/federated_learning_results.json`, Figure 5.
   - Status: Fully executed with 5 non-IID clients and machine-precision mask cancellation ($3.35 \times 10^{-14}$).
4. **Extension D: Threshold Calibration for Local DP**:
   - File: `src/local_differential_privacy.py`, `results/all_evaluation_metrics.json`, Figure 1, Figure 2.
   - Status: Fully executed, rescuing Local DP from zero-recall collapse ($F1 = 0.5614$ at $\epsilon = 5.0$).
