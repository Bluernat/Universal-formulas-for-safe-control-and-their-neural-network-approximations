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

Generates the pre-training dataset of $(q, k^*(q))$ pairs from randomly sampled constraint parameters. This is the data pipeline that makes the universal NN possible: because of the scaling property in Lemma 4.1, every possible control problem with $N \leq 10$ constraints and input dimension $m \leq 10$ maps into the same bounded input domain, so a single random dataset is sufficient.

**Key functions:**

- **`sample_in_unit_ball(dim)`** — draws a point uniformly at random from the $d$-dimensional unit ball using the standard normal-then-normalize trick with a radius sampled as $r \sim \text{Uniform}(0,1)^{1/d}$. Used to sample the $B_i$ vectors so they land uniformly inside $\mathcal{B}_1(0_m)$.

- **`min_norm_controller(A_list, B_list)`** — solves the minimum-norm QP
  $$\min_{u} \tfrac{1}{2}\|u\|^2 \quad \text{s.t.} \quad A_i + B_i^\top u \leq -0.05, \; \forall i,$$
  via CVXPY. This serves as a **feasibility warm-start**: it produces an initial guess $k_0$ that is already strictly inside the feasible polytope $\mathcal{K}_p$, which is necessary for SLSQP to converge reliably.

- **`J_p(k, A_list, B_list, r)`** — evaluates the cost function $\tilde{J}_q(k)$ (scaled form from Lemma 4.1). Returns a large penalty ($10^{12}$) if any constraint is violated, so the optimizer stays inside $\mathcal{K}_p$.

- **`grad_J_p(k, A_list, B_list, r)`** — computes the analytical gradient $\nabla \tilde{J}_q(k)$. Supplying the exact gradient to SLSQP is critical for both speed and the tight convergence check used to accept samples.

- **`robust_minimize_Jp(A_list, B_list, r, k0)`** — runs `scipy.optimize.minimize` with method `SLSQP`, passing `J_p` and `grad_J_p` together with explicit inequality constraints ($A_i + B_i^\top k \leq -10^{-4}$). After convergence, it performs a **three-way acceptance check**: `res.success`, strict feasibility ($< -10^{-5}$), and gradient norm $< 10^{-6}$. Only samples that pass all three are kept — this ensures the labels are genuine minimizers and not saddle points or boundary artifacts.

- **`generate_dataset(n, lower, upper)`** — the main data loop. Calls the functions above in sequence for each candidate sample, printing progress every 100 accepted points. Because many random draws fail the feasibility or accuracy checks, the actual number of attempts is typically 2–5× the number of accepted samples.

**Output:** A CSV with columns `a0,...,a9`, `b0_0,...,b9_9`, `r`, `kf_0,...,kf_9` (50,000 samples by default).

**Key parameters:**
```python
N_CONSTRAINTS  = 10
M_DIM          = 10
howmanysamples = 50000
```

---

### `NN_backbone_train.py`

Trains the backbone ResNet-MLP from scratch on the CSV produced by the data generation script. The script auto-detects $N$ and $m$ from the filename and column names, so it generalizes cleanly to other problem sizes without modification.

**Principal classes:**

- **`ResBlock(dim, dropout)`** — a pre-activation residual block with the structure:
  ```
  x  →  LayerNorm → SiLU → Linear(dim, dim) → Dropout
      →  LayerNorm → SiLU → Linear(dim, dim) → Dropout
      →  + x   (skip connection)
  ```
  Pre-activation normalization (LayerNorm before the nonlinearity) stabilizes training in deep networks and avoids the representation collapse that can occur with post-activation designs. The skip connection lets gradients flow directly to early layers.

- **`KfNet(in_dim, out_dim, hidden, n_blocks, dropout)`** — the full backbone model:
  ```
  Input (111) → Linear stem (256) → 5 × ResBlock(256) → LayerNorm → SiLU → Linear(256 → 10)
  ```
  With 5 blocks, the model has approximately 660K parameters. The SiLU activation is used throughout because it is smooth ($C^\infty$), which means the composed function is also smooth — an important property for the theoretical guarantees in Remark 4.4.

**Training pipeline:**
- `StandardScaler` is fit on the training split only and applied to both inputs and outputs; scalers are serialized with `joblib` so they can be reused at inference time without refitting.
- Data is split 80/10/10 (train/val/test) with a fixed random seed for reproducibility.
- The **OneCycleLR** scheduler ramps the learning rate up over the first 10% of training and then anneals it with a cosine schedule, which tends to find flatter minima and improves generalization.
- **Huber loss** ($\delta = 1.0$) is used instead of MSE to reduce sensitivity to the occasional large-error outlier that can arise from near-boundary samples.
- Early stopping monitors validation loss with patience = 20 epochs; the best checkpoint is saved and reloaded before test evaluation.
- At the end, per-dimension $R^2$ scores are printed on the held-out test set.

**Outputs:**
- `best_kf_model_N10_m10.pt` — best checkpoint (lowest validation loss)
- `scaler_X_N10_m10.pkl`, `scaler_y_N10_m10.pkl` — input/output scalers

**Key hyperparameters:**
```python
HIDDEN       = 256
N_BLOCKS     = 5
DROPOUT      = 0.15
BATCH_SIZE   = 512
MAX_EPOCHS   = 300
PATIENCE     = 20
LR           = 3e-3
WEIGHT_DECAY = 1e-4
```

---

### `nn_finetune.ipynb`

Adapts the pre-trained backbone to the specific geometry of the single-integrator CLF + reciprocal-barrier CBF task. The key design choice is **freezing the backbone** and training only a new two-layer head appended after the backbone's penultimate features — this preserves the general representation learned from random data while quickly specializing to the task distribution with far less data and compute.

**Architecture change from backbone to fine-tuned model:**

The backbone's original head (`LayerNorm → SiLU → Linear(256→10)`) is replaced by a deeper `ft_head`:
```
backbone features (256)
  → LayerNorm(256) → SiLU → Linear(256, 128) → Dropout(0.15)
  → LayerNorm(128) → SiLU → Linear(128, 10)
```
Only `ft_head` parameters are passed to the optimizer; backbone weights receive no gradient updates.

**Fine-tuning data generation (40,000 points):**

Each sample represents a random CLF + CBF scenario in varying dimension:

- Draw $n_\text{cbf} \in \{1,\ldots,9\}$ (number of obstacles) and $n_\text{dim} \in [n_\text{cbf}, 10]$ (effective state dimension). Coordinates beyond $n_\text{dim}$ are zero-padded so the full 10D NN input is always populated.
- State $x$: sampled with $\|x\| \in [0.3, 5.0]$ uniformly in radius and random direction in $\mathbb{R}^{n_\text{dim}}$.
- CLF: $V(x) = \tfrac{1}{2}\|x\|^2$, $W(x) = C_\text{CLF}\|x\|^2$ with $C_\text{CLF} \sim \text{Uniform}(0.05, 2.0)$. Constraint: $x^\top u + C_\text{CLF}\|x\|^2 \leq 0$.
- Each CBF: reciprocal barrier $h_i(x) = 8(1 - R_i^2/\|x-c_i\|^2)$ with $\alpha(s) = s$. Obstacle center at distance $\in [1.0, 5.0]$ in a random direction, radius $R \in [0.3, 1.5]$. Samples where $x$ is inside any obstacle ($h_i \leq 0$) are discarded.
- The label $k^*(q)$ is computed by numerically integrating the gradient flow $\dot{k} = -\nabla\tilde{J}_q(k)$ with SciPy's `solve_ivp` (RK45, `rtol = atol = 1e-6`, $T_\text{max} = 50$ s), stopping when $\|\nabla\tilde{J}_q\| < 10^{-6}$.
- The fixed backbone scalers (`scaler_X`, `scaler_y`) are reused to normalize inputs and outputs — this is essential so the frozen backbone receives the same input distribution it was trained on.

**Key functions in the notebook:**

- **`grad_Jtilde(k, A_t, B_t, r)`** — analytical gradient of $\tilde{J}_q$, used both as the ODE right-hand side during label computation and as the convergence criterion.
- **`make_feasible(k, A_t, B_t)`** — projects an infeasible point back into $\mathcal{K}_p$ by iteratively pushing along violated constraint normals. Used to repair warm-start guesses that land outside the feasible set.
- **`generate_finetune_data(n_samples, seed)`** — parallel data generation loop (single worker pool) with acceptance/rejection logic, printing periodic progress.

**Fine-tuning training:**
- Optimizer: AdamW, lr = 5×10⁻⁵, weight decay = 10⁻⁴ (much lower lr than backbone to avoid disrupting the frozen features)
- Scheduler: CosineAnnealingLR over 150 epochs
- Early stopping patience = 20; best `ft_head` state dict saved separately
- Batch size: 256, Huber loss (δ = 1.0)

**Output:** `best_kf_model_N10_m10_ft.pt` — fine-tuned head checkpoint.

---

### `runtime.ipynb`

The simulation and benchmarking notebook. It loads the trained model, defines five controller implementations, runs closed-loop trajectories from 9 initial conditions, and produces timing statistics and trajectory plots analogous to Table 1 and Figure 1 of the paper — extended to the 10D setting.

**System and constraints:**

For the single integrator $\dot{x} = u$ in $\mathbb{R}^{10}$, `gather_constraints(x)` assembles the full $(A_i, B_i)$ list at every state:
- CLF constraint: $a_0 = C_\text{CLF}\|x\|^2$, $b_0 = x$
- CBF constraints (one per obstacle): $a_i = -h_i(x)$, $b_i = -\nabla h_i(x)$, where $\nabla h_i(x) = \frac{2R_i^2}{\|x-c_i\|^4}(x - c_i)$ scaled by `CBF_GAIN`.

`scale_params(A_list, B_list)` applies the Lemma 4.1 normalization and computes $r = 1/M^2$ before passing to the NN. Unused constraint slots are zero-padded to fill the full $N_\text{max} = 10$ slots (Remark 4.2).

**Five controller functions:**

- **`ctrl_nn(x)`** — queries `nn_forward()` directly. Fastest: one forward pass through the model, no additional computation.
- **`ctrl_hardnet(x)`** — calls `nn_forward()` then applies `hardnet_project(k_nn, A_list, B_list)`, which solves $P(k) = k - A^\top(AA^\top + \varepsilon I)^{-1}\text{ReLU}(Ak + a)$ to project the NN output onto the feasible polytope (Min & Azizan 2025). A small $\varepsilon$ regularization handles ill-conditioning near $x = 0$ where $b_\text{clf} = x \to 0$.
- **`ctrl_qp(x)`** — solves the minimum-norm QP $\min\|u\|^2$ s.t. $a_i + b_i^\top u \leq 0$ via CVXPY/OSQP with warm starting. The CVXPY `Problem` and `Parameter` objects are constructed once and reused across time steps for efficiency.
- **`ctrl_ustar_qp(x)`** — computes $u^*$ by integrating $\dot{k} = -\nabla\tilde{J}_q(k)$ (RK45) from a QP warm-start, stopping when $\|\nabla\tilde{J}_q\|$ falls below `GRAD_TOL`. The warm-start substantially reduces the number of ODE steps needed.
- **`ctrl_ustar_nn(x)`** — same gradient-flow integration as above, but warm-started from the NN prediction instead of the QP. This is the key practical contribution: the NN provides a high-quality starting point that is already close to $u^*$, making the refinement faster than using the QP.

**`simulate(x0, control_fn)`** — runs a fixed-step Euler simulation ($\Delta t = 0.005$ s) up to $T_\text{max} = 20$ s, recording the full state trajectory, per-obstacle CBF values $h_i(x(t))$, and CLF value $V(x(t))$. Integration stops early when $\|x\| < 0.2$.

**`plot_all(trajs, all_h, all_V, label, prefix)`** — produces a $2 \times 2$ grid of 2D projections of the 10D trajectories (pairs of dimensions), plus time-series plots of $V(t)$ and $h_i(t)$ per obstacle, useful for visually verifying stability and constraint satisfaction.

**Outputs:** trajectory plots per controller (saved as PNG), timing arrays per controller (saved as `.npy`), box plots of execution time distributions, and a summary table with mean ± std timing and constraint violation statistics.

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
  author    = {},
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
