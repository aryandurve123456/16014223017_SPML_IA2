# IA-2 Viva Voce Defense Guide: Security and Privacy for Machine Learning

This guide prepares the student for oral examination (viva voce) on the IA-2 project:
**"Privacy-Preserving Machine Learning Using Differential Privacy and Local Differential Privacy: Evaluating the Privacy–Utility Trade-off in Income Classification"**.

---

## 1. Core Technical Questions & Authoritative Answers

### Q1: What are the fundamental differences between Central DP and Local DP, and why does Central DP achieve superior utility?
**Answer**:
- **Trust Model**: Central DP assumes a trusted data curator who aggregates raw records and adds noise during release. Local DP assumes an untrusted aggregator, requiring each client device to randomize its data before transmission.
- **Sensitivity & Error Rates**:
  - In Central DP, sensitivity scales inversely with sample size ($\Delta \propto O(1/n)$), adding noise once to aggregate parameters. Estimation error decreases as $O\left(\frac{1}{n \epsilon}\right)$.
  - In Local DP, sensitivity is governed by the diameter of the client's single input domain ($\Delta = \text{diam}(\mathcal{V})$), independent of $n$. Estimation error scales as $O\left(\frac{1}{\sqrt{n} \epsilon}\right)$.
- **Curse of Dimensionality**: When perturbing $d = 105$ features under sequential composition, the client's budget is split to $\epsilon_j = \epsilon / 105$. At $\epsilon = 1.0$, each attribute receives $\epsilon_j \approx 0.0095$, creating substantial feature noise. Central DP adds noise once to model weights without per-feature budget dilution.

### Q2: Why did Local DP collapse to 0% recall in IA-1, and how did you resolve it in IA-2?
**Answer**:
- **Root Cause**: The Adult dataset has an imbalanced target distribution (76.07% class `<=50K` vs 23.93% class `>50K`). When 105 features are perturbed with heavy noise, the signal-to-noise ratio per attribute drops. Standard logistic regression with a fixed $0.5$ threshold defaults to the majority class for all records, yielding Accuracy = 76.38%, Precision = 0.0, Recall = 0.0, and F1 = 0.0.
- **IA-2 Resolution**:
  1. **Class-Weighted Optimization**: We trained with `class_weight='balanced'`, penalizing false negatives inversely proportional to class frequencies.
  2. **Decision-Threshold Calibration**: We calibrated the decision threshold on the privatized training set to maximize F1.
  - **Outcome**: At $\epsilon = 5.0$, Local DP recovered an F1-score of **0.5614** and ROC-AUC of **0.7872**, proving that Local DP retains meaningful decision utility when class imbalance is properly handled.

### Q3: How does DP-SGD with Opacus work, and how does its privacy accounting differ from DP-ERM?
**Answer**:
- **DP-ERM (Output Perturbation)**: Fits an $L_2$-regularized convex model on clean data and adds calibrated Gaussian noise $\mathcal{N}(0, \sigma^2 I)$ to the final weights in one step. Sensitivity is bounded by $\Delta_2 \le \frac{2 R}{n \lambda}$.
- **DP-SGD (Iterative Gradient Perturbation)**:
  1. Computes per-sample gradients $g_i(w) = \nabla_w \ell(w; x_i, y_i)$.
  2. Clips per-sample gradient $L_2$ norm: $\bar{g}_i = g_i / \max(1, \|g_i\|_2 / C)$ with clipping threshold $C = 1.0$.
  3. Adds calibrated Gaussian noise to the sum of clipped gradients: $\tilde{g} = \sum_{i \in B} \bar{g}_i + \mathcal{N}(0, \sigma^2 C^2 I)$.
- **Privacy Accounting**: Rather than basic composition ($\sum \epsilon$), Opacus uses the **Rényi Differential Privacy (RDP)** accountant. Subsampling ratio $q = B/n$ provides "privacy amplification by subsampling." The accountant converts RDP bounds across epochs to standard $(\epsilon, \delta)$-DP.
- **Empirical Result**: At $\sigma = 1.2$, DP-SGD achieves $\epsilon = 0.8029$ ($\delta = 10^{-5}$) with **ROC-AUC = 0.8848** and **F1 = 0.5809**.

### Q4: What is a Membership Inference Attack (MIA), and what do your empirical results demonstrate?
**Answer**:
- **Threat Model**: An adversary queries a trained target model and attempts to determine whether a given record $(x, y)$ was in the training set (member = 1) or holdout set (non-member = 0).
- **Shadow Model Architecture**: We strictly partitioned data into 4 disjoint quadrants: Target Train, Target Test, Shadow Train, and Shadow Test (zero data contamination). An attack classifier was trained on shadow-model confidence scores, cross-entropy loss, and prediction entropy.
- **Empirical Findings**:
  - The Non-Private Baseline achieved an Attack ROC-AUC of **0.4955** and MIA Advantage of **-0.0025** (chance level).
  - Central DP-ERM ($\epsilon=1.0$) achieved **0.4954**, and Central DP-SGD achieved **0.4948**.
- **Explanation**: In well-regularized linear models on large tabular datasets ($n = 32,537$), the generalization gap is $< 0.2\%$, leaving near-zero memorization signal for MIA. However, Differential Privacy provides a mathematical guarantee: the worst-case membership advantage is strictly bounded by $e^\epsilon - 1$, protecting against future high-capacity attacks or arbitrary auxiliary side-channel information.

### Q5: How does Pairwise Additive Masking Secure Aggregation (SecAgg) work mathematically?
**Answer**:
- **Protocol**: In a network of $K = 5$ clients, every pair $(i, j)$ with $i < j$ agrees on a shared pseudo-random mask $s_{i, j} \sim \mathcal{N}(0, \sigma^2 I)$ (via simulated Diffie-Hellman key agreement).
- **Client Blinding**: Client $i$ transmits:
  $$\tilde{u}_i = u_i + \sum_{j > i} s_{i, j} - \sum_{j < i} s_{j, i}$$
- **Server Aggregation**: The aggregator sums all received masked vectors:
  $$\sum_{i=1}^K \tilde{u}_i = \sum_{i=1}^K u_i + \sum_{i=1}^K \left( \sum_{j > i} s_{i,j} - \sum_{j < i} s_{j,i} \right) = \sum_{i=1}^K u_i$$
  Every pairwise mask $s_{i, j}$ appears exactly once with a positive sign (from client $i$) and once with a negative sign (from client $j$), canceling out identically!
- **Empirical Verification**: We verified a maximum cancellation error of $3.35 \times 10^{-14}$ (machine precision zero), while individual updates showed near-zero correlation ($r \approx 0.05$) with true gradients, blinding the server completely.

### Q6: Why is Compressive Privacy an empirical heuristic rather than a formal Differential Privacy guarantee?
**Answer**:
- **Mechanism**: Compressive Privacy projects feature vectors onto the orthogonal complement of the linear sensitive direction ($v_S \propto \mu_{S=1} - \mu_{S=0}$):
  $$P_\perp = I - \frac{v_S v_S^T}{\|v_S\|_2^2}$$
- **Why it is an empirical heuristic**:
  1. It suppresses only **linear** correlation with a single pre-specified sensitive attribute ($S = \text{sex}$). Non-linear relations or proxy combinations may still leak $S$.
  2. It does not bound global sensitivity across neighboring datasets.
  3. It does not provide plausible deniability against membership inference.
- **Empirical Value**: It reduced adversary gender balanced accuracy from 100% to **50.00% (random guess)** while retaining 88.3% of baseline predictive utility ($F1 = 0.5753$).

### Q7: What is the Homogeneity Attack in $k$-anonymity, and how did your code demonstrate it?
**Answer**:
- $k$-Anonymity requires every equivalence class of quasi-identifiers to contain $\ge k$ records, preventing identity re-identification.
- **Homogeneity Attack**: Occurs when all $k$ records in an equivalence class share the identical sensitive attribute value (e.g. all earn $> \$50K$). An adversary knowing a target's quasi-identifiers learns their income with 100% certainty, bypassing $k$-anonymity.
- **Empirical Proof**: In our evaluation at $k=10$, **22.57% of equivalence classes** were completely homogeneous, demonstrating why $l$-diversity or Differential Privacy is required.
