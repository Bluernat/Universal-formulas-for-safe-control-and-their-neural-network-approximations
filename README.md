# Neural Network-Based Universal Formulas for Control

> **Companion code for the NeurIPS 2025 submission:**
> *"Neural Network-based Universal Formulas for Control"*

This repository contains all code to reproduce the high-dimensional simulation experiments of the paper: dataset generation, backbone training, task-specific fine-tuning, and runtime comparison of five controllers on a 10-dimensional single-integrator system with 1 CLF + 9 CBF constraints.

---

## Overview

A classical challenge in safety-critical control is computing, in real time, a smooth controller that simultaneously satisfies an arbitrary number of affine constraints of the form

$$a_i(x) + b_i(x)^\top u < 0, \quad i = 1, \ldots, N,$$

arising from Control Lyapunov Functions (CLFs), Control Barrier Functions (CBFs), and input bounds. State-of-the-art approaches solve a Quadratic Program (QP) at every state, which is computationally expensive and may produce non-smooth controllers.

This work introduces a **universal formula** — a smooth controller given as the unique minimizer of the strictly convex function

$$J_p(k) = -\sum_{i=1}^{N} \frac{\|B_i\|^2 + \|k\|^2}{2(A_i + B_i^\top k)},$$

and trains a single neural network to approximate it. A key theoretical result (Lemma 4.1 of the paper) shows that the minimizer can be recovered from a *normalized*, bounded input, enabling data collection over a compact set and training a **universal NN** that works for any control problem with input dimension ≤ *m* and number of constraints ≤ *N*, regardless of state dimension.

---

## Repository Structure

```
.
├── Generate_dataset_random_data.py   # Dataset generation for backbone pre-training
├── NN_backbone_train.py              # Backbone ResNet-MLP training
├── nn_finetune.ipynb                 # Fine-tuning to CLF+CBF task (Kaggle/local)
└── runtime.ipynb                     # Runtime comparison of 5 controllers (10D system)
```

---

## Method

### 1. Universal Formula and Scaling (Section 3–4 of the paper)

Given parameters $p = (A_1, \ldots, A_N, B_1, \ldots, B_N)$, define the normalized version

$$q(p) = \left(\frac{A_i}{M}, \frac{B_i}{M}, \frac{1}{M^2}\right), \quad M = \max\{|A_i|, \|B_i\|, 1\}.$$

The minimizer of $J_p$ equals the minimizer of $\tilde{J}_q$, which is defined over the bounded domain $[-1,1]^N \times \mathcal{B}_1(0_m)^N \times [0,1]$. This is what the NN is trained to predict.

### 2. Neural Network Architecture

The model has two parts:

**Backbone** (pre-trained on random data, frozen during fine-tuning):
- Input: $\mathbb{R}^{111}$ (= $N_{\max}(1 + m) + 1$ with $N_{\max} = m = 10$)
- Stem: `Linear(111 → 256)`
- 5 ResBlocks, each: `LayerNorm → SiLU → Linear(256,256) → Dropout(0.15) → LayerNorm → SiLU → Linear(256,256) → Dropout(0.15)` + skip connection
- Head: `LayerNorm(256) → SiLU → Linear(256 → 10)`
- Activation: SiLU throughout

**Fine-tune head** (only part trained during fine-tuning, backbone frozen):
- `LayerNorm(256) → SiLU → Linear(256,128) → Dropout(0.15) → LayerNorm(128) → SiLU → Linear(128,10)`

### 3. Five Controllers Compared

| Controller | Description |
|---|---|
| **NN** | Direct NN prediction of $k^*(q)$ |
| **HardNet-Aff** | NN output projected to satisfy all constraints (Min & Azizan 2025) |
| **CLF-CBF QP** | Minimum-norm QP solved online at every state |
| **$u^*$ w/ QP warmstart** | Gradient-flow integration of $\tilde{J}_q$, warmstarted from QP |
| **$u^*$ w/ NN warmstart** | Gradient-flow integration of $\tilde{J}_q$, warmstarted from NN |

---

## Experiment: 10D Single Integrator

The simulation benchmark uses a single integrator $\dot{x} = u$ in $\mathbb{R}^{10}$ with:

- **CLF:** $V(x) = \tfrac{1}{2}\|x\|^2$, $W(x) = 0.5\|x\|^2$ → constraint $x^\top u + 0.5\|x\|^2 \leq 0$
- **9 CBFs:** reciprocal barriers $h_i(x) = 8\left(1 - R_i^2/\|x - c_i\|^2\right)$, $\alpha(s) = s$

Obstacle centers are placed along the straight line from each initial condition to the origin (at 45% of the way), forcing trajectories to navigate around them. All radii are $R = 0.8$.

<details>
<summary>Obstacle centers in ℝ¹⁰</summary>

```
c1 = [ 2.20,  0.00,  0.55, -0.55,  1.10,  0.00,  0.55, -1.10,  0.275, -0.275]
c2 = [-2.40,  0.60, -0.60,  1.20,  0.00, -0.60, -1.80,  0.60, -0.90,  0.60 ]
c3 = [ 0.00,  2.00,  1.00, -0.50, -1.00,  0.25,  0.00,  1.50,  0.50, -0.50 ]
c4 = [ 1.74, -1.74,  0.00,  0.58,  0.58, -1.16,  1.16, -0.58, -0.29,  0.29 ]
c5 = [-1.04, -2.08,  0.78, -0.78,  0.26,  0.78, -0.52, -1.04,  0.00,  1.04 ]
c6 = [ 1.24,  1.86, -1.24,  0.00, -0.62,  1.24,  0.93,  0.93, -0.62, -0.62 ]
c7 = [-1.68,  1.12,  0.28,  1.12, -0.28, -0.56, -1.12,  0.00,  1.12,  0.56 ]
c8 = [ 0.54, -2.16, -0.54,  0.54,  1.08,  0.00,  1.62, -1.62, -0.54,  0.00 ]
c9 = [ 1.375, 0.00, -0.55,  0.275, 0.00,  0.00,  0.9625, 0.1375, -0.4125, -0.1375]
```
</details>

---

## Files in Detail

### `Generate_dataset_random_data.py`

Generates the pre-training dataset of $(q, k^*(q))$ pairs for random constraint parameters.

**Pipeline:**
1. Sample $N = 10$ scalars $A_i \in [-1, 1]$ and vectors $B_i \in \mathcal{B}_1(0_{10})$, plus $r \in (0, 1]$.
2. Warmstart a feasibility check via a minimum-norm QP (CVXPY).
3. Minimize $J_p$ via SLSQP with gradient $\nabla J_p$ supplied analytically.
4. Accept a sample only if optimization succeeds, the point is strictly feasible ($A_i + B_i^\top k < -10^{-5}$ for all $i$), and $\|\nabla J_p(k^*)\| < 10^{-6}$.

**Output:** A CSV with columns `a0,...,a9`, `b0_0,...,b9_9`, `r`, `kf_0,...,kf_9` (50,000 samples by default).

**Key parameters:**
```python
N_CONSTRAINTS = 10
M_DIM         = 10
howmanysamples = 50000
```

---

### `NN_backbone_train.py`

Trains the backbone ResNet-MLP on the generated CSV dataset.

**Training details:**
- Optimizer: AdamW, lr = 3×10⁻³, weight decay = 10⁻⁴
- Scheduler: OneCycleLR (cosine annealing, 10% warmup)
- Loss: Huber loss (δ = 1.0)
- Max epochs: 300, early stopping patience = 20
- Batch size: 512
- Train/Val/Test split: 80/10/10

**Features are standardized** with `StandardScaler`; scalers are saved as `.pkl` files for later inference.

**Outputs:**
- `best_kf_model_N10_m10.pt` — best checkpoint (lowest validation loss)
- `scaler_X_N10_m10.pkl`, `scaler_y_N10_m10.pkl` — input/output scalers

---

### `nn_finetune.ipynb`

Fine-tunes the pre-trained backbone to the specific task of single-integrator CLF + CBF control.

**Fine-tuning dataset (40,000 points):**
- For each sample, draw $n_\text{cbf} \in \{1,\ldots,9\}$ obstacles and state dimension $n_\text{dim} \geq n_\text{cbf}$.
- State $x$ is sampled uniformly in $[\|x\| \in [0.3, 5.0]]$ with a random direction; coordinates beyond $n_\text{dim}$ are zero-padded.
- CLF coefficient $C_\text{CLF} \sim \text{Uniform}(0.05, 2.0)$.
- Each obstacle: center at random direction with distance $\in [1.0, 5.0]$, radius $R \in [0.3, 1.5]$; states inside an obstacle are discarded.
- Label $k^*(q)$: computed by integrating the gradient flow $\dot{k} = -\nabla \tilde{J}_q(k)$ with RK45 until $\|\nabla \tilde{J}_q\| < 10^{-6}$ (tolerance: `rtol = atol = 1e-6`, $T_\text{max} = 50$ s).
- Parameters are normalized with Lemma 4.1 before being passed to the NN; the fixed backbone scalers are reused.

**Fine-tuning training:**
- Only the `ft_head` is trained; backbone weights are frozen.
- Optimizer: AdamW, lr = 5×10⁻⁵, weight decay = 10⁻⁴
- Scheduler: CosineAnnealingLR
- Max epochs: 150, early stopping patience = 20
- Batch size: 256, Huber loss (δ = 1.0)

**Output:** `best_kf_model_N10_m10_ft.pt` — fine-tuned head checkpoint.

---

### `runtime.ipynb`

Benchmarks all five controllers on the 10D single-integrator system.

**Simulation setup:**
- $\Delta t = 0.005$ s, $T_\text{max} = 20$ s, stop radius $= 0.2$
- 9 initial conditions spread around the obstacle configuration
- Remark 4.2 of the paper: unused constraint slots are padded with dummy satisfied constraints, so the same NN trained for $N_\text{max} = 10$ handles any $N \leq 10$

**Outputs:** trajectory plots per controller, timing box plots, and a summary table analogous to Table 1 of the paper.

---

## Installation

```bash
pip install torch numpy scipy pandas scikit-learn cvxpy joblib matplotlib
```

Python 3.9+ recommended. A CUDA-capable GPU will speed up backbone training but is not required; fine-tuning and inference run comfortably on CPU.

---

## Usage

**Step 1 — Generate pre-training data:**
```bash
python Generate_dataset_random_data.py
# Produces: generated_data_N10_m10.csv  (≈50 000 samples, may take several hours)
```

**Step 2 — Train the backbone:**
```bash
python NN_backbone_train.py
# Produces: best_kf_model_N10_m10.pt, scaler_X_N10_m10.pkl, scaler_y_N10_m10.pkl
```

**Step 3 — Fine-tune to the CLF+CBF task:**

Open `nn_finetune.ipynb` and run all cells. Place the three files from Step 2 in the working directory (or point `INPUT_DIR` accordingly). Produces `best_kf_model_N10_m10_ft.pt`.

**Step 4 — Runtime comparison:**

Open `runtime.ipynb` and run all cells with all four files from Steps 2–3 in the working directory. Reproduces the trajectory plots and timing table.

---

## Citation

If you use this code, please cite the paper:

```bibtex
@inproceedings{anonymous2025universalformulas,
  title     = {Neural Network-based Universal Formulas for Control},
  author    = {Anonymous},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2025},
  note      = {Under review}
}
```

---

## References

- A. D. Ames et al., "Control barrier function based quadratic programs for safety critical systems," *IEEE TAC*, 2017.
- E. D. Sontag, "A universal construction of Artstein's theorem on nonlinear stabilization," *Systems & Control Letters*, 1989.
- Y. Min and N. Azizan, "Hard-constrained neural networks with universal approximation guarantees," *arXiv:2410.10807*, 2025.
- P. Ong and J. Cortés, "Universal formula for smooth safe stabilization," *IEEE CDC*, 2019.
