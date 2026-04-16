# Neural Network-based Universal Formulas for Control

Companion code for the NeurIPS 2025 submission: *"Neural Network-based Universal Formulas for Control"*

This repository contains the full pipeline to reproduce the high-dimensional simulation experiments: dataset generation, backbone training, task-specific fine-tuning, and runtime comparison of five controllers on a 10-dimensional single-integrator system with 1 CLF + 9 CBF constraints.

---

## Background

A core challenge in safety-critical control is computing, in real time, a smooth controller that simultaneously satisfies an arbitrary number of affine constraints

$$a_i(x) + b_i(x)^\top u < 0, \quad i = 1, \ldots, N,$$

arising from Control Lyapunov Functions (CLFs), Control Barrier Functions (CBFs), and input bounds. State-of-the-art approaches solve a Quadratic Program (QP) at every state, which is computationally expensive and can produce non-smooth controllers.

This work introduces a **universal formula**: a smooth controller given as the unique minimizer of the strictly convex function

$$J_p(k) = -\sum_{i=1}^{N} \frac{\|B_i\|^2 + \|k\|^2}{2(A_i + B_i^\top k)},$$

and trains a neural network to approximate it. A key result (Lemma 4.1) shows that the minimizer can be recovered from a *normalized*, bounded input, enabling training on a compact set and producing a **universal NN** that generalizes to any control problem with input dimension $\leq m$ and number of constraints $\leq N$, regardless of state dimension.

---

## Repository Structure

```
.
├── Generate_dataset_random_data.py   # Dataset generation for backbone pre-training
├── NN_backbone_train.py              # Backbone ResNet-MLP training
├── nn_finetune.ipynb                 # Fine-tuning to CLF+CBF task
└── runtime.ipynb                     # Runtime comparison of 5 controllers (10D system)
```

---

## Method

### Universal Formula and Scaling

Given parameters $p = (A_1, \ldots, A_N, B_1, \ldots, B_N)$, the normalized form is

$$q(p) = \left(\frac{A_i}{M},\ \frac{B_i}{M},\ \frac{1}{M^2}\right), \quad M = \max\{|A_i|, \|B_i\|, 1\}.$$

The minimizer of $J_p$ equals the minimizer of $\tilde{J}_q$, which lives over the bounded domain $[-1,1]^N \times \mathcal{B}_1(0_m)^N \times [0,1]$. This is what the NN learns to predict.

### Architecture

The model has two parts:

**Backbone** (pre-trained on random data, frozen during fine-tuning):
- Input: $\mathbb{R}^{111}$ (i.e., $N_{\max}(1 + m) + 1$ with $N_{\max} = m = 10$)
- Stem: `Linear(111 → 256)`
- 5 ResBlocks: `LayerNorm → SiLU → Linear(256,256) → Dropout(0.15) → LayerNorm → SiLU → Linear(256,256) → Dropout(0.15)` + skip
- Head: `LayerNorm(256) → SiLU → Linear(256 → 10)`

**Fine-tune head** (trained on task-specific data; backbone frozen):
- `LayerNorm(256) → SiLU → Linear(256,128) → Dropout(0.15) → LayerNorm(128) → SiLU → Linear(128,10)`

### Controllers Compared

| Controller | Description |
|---|---|
| **NN** | Direct NN prediction of $k^*(q)$ |
| **HardNet-Aff** | NN output projected to satisfy all constraints (Min & Azizan 2025) |
| **CLF-CBF QP** | Minimum-norm QP solved online at every state |
| **$u^*$ w/ QP warmstart** | Gradient-flow integration of $\tilde{J}_q$, initialized from QP solution |
| **$u^*$ w/ NN warmstart** | Gradient-flow integration of $\tilde{J}_q$, initialized from NN prediction |

---

## Experiment: 10D Single Integrator

The benchmark uses a single integrator $\dot{x} = u$ in $\mathbb{R}^{10}$ with:

- **CLF:** $V(x) = \tfrac{1}{2}\|x\|^2$, decay condition $x^\top u + 0.5\|x\|^2 \leq 0$
- **9 CBFs:** reciprocal barriers $h_i(x) = 8\left(1 - R_i^2/\|x - c_i\|^2\right)$ with $\alpha(s) = s$

Obstacle centers are placed along the straight line from each initial condition to the origin (at 45% of the way), forcing trajectories to navigate around them. All radii are $R = 0.8$.

<details>
<summary>Obstacle centers in $\mathbb{R}^{10}$</summary>

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

Generates the pre-training dataset of $(q, k^*(q))$ pairs from randomly sampled constraint parameters. Because of the scaling property in Lemma 4.1, every control problem with $N \leq 10$ constraints and input dimension $m \leq 10$ maps into the same bounded input domain, so a single random dataset is sufficient for the universal backbone.

Key functions:

- **`sample_in_unit_ball(dim)`** -- draws uniformly from the $d$-dimensional unit ball using the normalize-then-scale trick. Used to sample the $B_i$ vectors.
- **`min_norm_controller(A_list, B_list)`** -- solves the minimum-norm QP via CVXPY to produce a feasibility warm-start for SLSQP.
- **`J_p(k, A_list, B_list, r)`** -- evaluates $\tilde{J}_q(k)$; returns a large penalty if any constraint is violated.
- **`grad_J_p(k, A_list, B_list, r)`** -- analytical gradient of $\tilde{J}_q$, passed directly to SLSQP for speed and accuracy.
- **`robust_minimize_Jp(...)`** -- runs SLSQP and accepts a sample only if it passes three checks: `res.success`, strict feasibility ($< -10^{-5}$), and gradient norm $< 10^{-6}$.
- **`generate_dataset(n, lower, upper)`** -- the main data loop. Expects roughly 2--5x the number of raw attempts as accepted samples due to rejection.

Output: a CSV with columns `a0,...,a9`, `b0_0,...,b9_9`, `r`, `kf_0,...,kf_9` (50,000 samples by default).

Key parameters:
```python
N_CONSTRAINTS  = 10
M_DIM          = 10
howmanysamples = 50000
```

---

### `NN_backbone_train.py`

Trains the backbone ResNet-MLP from scratch on the CSV from the previous step. Auto-detects $N$ and $m$ from the filename and column names.

Key classes:

- **`ResBlock(dim, dropout)`** -- pre-activation residual block with LayerNorm before each nonlinearity. The skip connection lets gradients flow to early layers and avoids the representation collapse common with post-activation designs.
- **`KfNet(in_dim, out_dim, hidden, n_blocks, dropout)`** -- the full backbone (approx. 660K parameters).

Training details:
- `StandardScaler` fit on training split only; serialized with `joblib` for reuse at inference time.
- 80/10/10 train/val/test split with a fixed seed.
- **OneCycleLR** scheduler (warmup + cosine annealing).
- **Huber loss** ($\delta = 1.0$) to reduce sensitivity to near-boundary outliers.
- Early stopping with patience = 20; best checkpoint reloaded before test evaluation.

Outputs:
- `best_kf_model_N10_m10.pt`
- `scaler_X_N10_m10.pkl`, `scaler_y_N10_m10.pkl`

Key hyperparameters:
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

Adapts the pre-trained backbone to the CLF + reciprocal-barrier CBF task. The backbone is frozen; only a new two-layer head is trained. This preserves the general representation from random data while quickly specializing to the task distribution with far less data and compute.

Fine-tuning data (40,000 points): each sample draws a random number of CBFs and effective state dimension, samples a state and obstacle geometry, and computes the label $k^*(q)$ by integrating the gradient flow $\dot{k} = -\nabla\tilde{J}_q(k)$ with RK45 until $\|\nabla\tilde{J}_q\| < 10^{-6}$. Samples where $x$ is inside any obstacle are discarded. The backbone scalers from Step 2 are reused so the frozen network receives the same input distribution it was trained on.

Key functions:

- **`grad_Jtilde(k, A_t, B_t, r)`** -- analytical gradient of $\tilde{J}_q$, used both as the ODE right-hand side and as the convergence check.
- **`make_feasible(k, A_t, B_t)`** -- projects an infeasible warm-start back into $\mathcal{K}_p$ by pushing along violated constraint normals.
- **`generate_finetune_data(n_samples, seed)`** -- parallel data generation with acceptance/rejection logic.

Training: AdamW (lr = 5e-5, wd = 1e-4), CosineAnnealingLR over 150 epochs, Huber loss, batch size 256, early stopping with patience 20.

Output: `best_kf_model_N10_m10_ft.pt`

---

### `runtime.ipynb`

Loads the trained model, runs closed-loop trajectories from 9 initial conditions under all five controllers, and produces timing statistics and trajectory plots (Table 1 and Figure 1 of the paper, extended to the 10D setting).

`gather_constraints(x)` assembles $(A_i, B_i)$ at every state from the CLF and CBF formulas. `scale_params(A_list, B_list)` applies the Lemma 4.1 normalization. Unused constraint slots are zero-padded to fill the full $N_\text{max} = 10$ slots (Remark 4.2).

Controller implementations:

- **`ctrl_nn(x)`** -- one forward pass, no additional computation.
- **`ctrl_hardnet(x)`** -- NN prediction followed by projection onto the feasible polytope via $P(k) = k - A^\top(AA^\top + \varepsilon I)^{-1}\text{ReLU}(Ak + a)$ (Min & Azizan 2025).
- **`ctrl_qp(x)`** -- minimum-norm QP via CVXPY/OSQP with warm starting; Problem and Parameter objects constructed once and reused.
- **`ctrl_ustar_qp(x)`** -- gradient-flow integration from a QP warm-start.
- **`ctrl_ustar_nn(x)`** -- gradient-flow integration from the NN prediction. The NN provides a starting point close to $u^*$, making refinement faster than using the QP.

`simulate(x0, control_fn)` runs Euler integration ($\Delta t = 0.005$ s, $T_\text{max} = 20$ s) and stops early at $\|x\| < 0.2$. Outputs include full state trajectories, CBF values $h_i(x(t))$, and CLF values $V(x(t))$.

---

## Installation

```bash
pip install torch numpy scipy pandas scikit-learn cvxpy joblib matplotlib
```

Python 3.9+ recommended. A CUDA GPU speeds up backbone training but is not required; fine-tuning and inference run comfortably on CPU.

---

## Usage

**Step 1 -- Generate pre-training data:**
```bash
python Generate_dataset_random_data.py
# Produces: generated_data_N10_m10.csv  (~50k samples; may take several hours)
```

**Step 2 -- Train the backbone:**
```bash
python NN_backbone_train.py
# Produces: best_kf_model_N10_m10.pt, scaler_X_N10_m10.pkl, scaler_y_N10_m10.pkl
```

**Step 3 -- Fine-tune to the CLF+CBF task:**

Open `nn_finetune.ipynb` and run all cells with the Step 2 outputs in the working directory (or set `INPUT_DIR` accordingly). Produces `best_kf_model_N10_m10_ft.pt`.

**Step 4 -- Runtime comparison:**

Open `runtime.ipynb` and run all cells with all outputs from Steps 2 and 3 present. Reproduces the trajectory plots and timing table.

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{,
  title     = {Neural Network-based Universal Formulas for Control},
  author    = {},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2025},
}
```

---

## References

- A. D. Ames et al., "Control barrier function based quadratic programs for safety critical systems," *IEEE TAC*, 2017.
- E. D. Sontag, "A universal construction of Artstein's theorem on nonlinear stabilization," *Systems & Control Letters*, 1989.
- Y. Min and N. Azizan, "Hard-constrained neural networks with universal approximation guarantees," *arXiv:2410.10807*, 2025.
- P. Ong and J. Cortés, "Universal formula for smooth safe stabilization," *IEEE CDC*, 2019.
