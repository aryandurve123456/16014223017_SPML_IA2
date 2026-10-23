# IA-2 Master Comparison Table: Privacy-Preserving Machine Learning

| Paradigm / Architecture | Specific Mechanism | Privacy Budget (eps) | Delta | Accuracy (Mean +/- Std) | F1-Score (Mean +/- Std) | ROC-AUC (Mean) | Attack ROC-AUC (MIA Vulnerability) |
|---|---|---|---|---|---|---|---|
| Baseline Linear | Clean Non-Private LR | N/A | N/A | 0.8511 | 0.6513 | 0.9035 | 0.4955 |
| Baseline Non-Linear | MLPClassifier (64, 32) | N/A | N/A | 0.8491 | 0.6424 | 0.9006 | N/A |
| Federated Learning | FedAvg (5 Clients) + SecAgg | N/A (SecAgg Blinded) | N/A | 0.7828 | 0.1585 | 0.8939 | N/A |
| Central DP | DP-ERM Output Perturbation | 0.1 | 1e-05 | 0.7890 +/- 0.0210 | 0.5786 +/- 0.0350 | 0.8537 | N/A |
| Central DP | DP-ERM Output Perturbation | 0.5 | 1e-05 | 0.8017 +/- 0.0080 | 0.6048 +/- 0.0067 | 0.8617 | 0.4971 |
| Central DP | DP-ERM Output Perturbation | 1.0 | 1e-05 | 0.8036 +/- 0.0033 | 0.6053 +/- 0.0048 | 0.8621 | 0.4954 |
| Central DP | DP-ERM Output Perturbation | 2.0 | 1e-05 | 0.8045 +/- 0.0017 | 0.6055 +/- 0.0024 | 0.8623 | N/A |
| Central DP | DP-ERM Output Perturbation | 5.0 | 1e-05 | 0.8045 +/- 0.0007 | 0.6049 +/- 0.0008 | 0.8623 | N/A |
| Central DP | Opacus DP-SGD (noise=0.8) | 2.0902 | 1e-05 | 0.8346 | 0.5823 | 0.8850 | 0.4948 |
| Central DP | Opacus DP-SGD (noise=1.2) | 0.8029 | 1e-05 | 0.8342 | 0.5809 | 0.8848 | 0.4948 |
| Central DP | Opacus DP-SGD (noise=2.0) | 0.3856 | 1e-05 | 0.8338 | 0.5810 | 0.8844 | 0.4948 |
| Local DP | Warner RR + Duchi (Balanced) | 0.1 | 0.0 | 0.6993 +/- 0.0307 | 0.1044 +/- 0.0827 | 0.4756 | N/A |
| Local DP | Warner RR + Duchi (Balanced) | 0.5 | 0.0 | 0.7123 +/- 0.0339 | 0.1348 +/- 0.1447 | 0.5333 | N/A |
| Local DP | Warner RR + Duchi (Balanced) | 1.0 | 0.0 | 0.7098 +/- 0.0418 | 0.1680 +/- 0.1796 | 0.5934 | N/A |
| Local DP | Warner RR + Duchi (Balanced) | 2.0 | 0.0 | 0.7264 +/- 0.0242 | 0.2257 +/- 0.1980 | 0.6757 | N/A |
| Local DP | Warner RR + Duchi (Balanced) | 5.0 | 0.0 | 0.7403 +/- 0.0190 | 0.4769 +/- 0.1200 | 0.7777 | N/A |
