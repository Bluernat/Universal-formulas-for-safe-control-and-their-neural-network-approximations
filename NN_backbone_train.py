"""
Neural Network for predicting kf_0..kf_{m-1} from N*(1+m)+1 input features.
Architecture: Deep residual MLP with LayerNorm + Dropout.
Training: AdamW + OneCycleLR + early stopping.

Set DATA_PATH to whichever CSV was produced by generate_dataset.py.
N and m are inferred automatically from the file.
"""

# CONFIG 

DATA_PATH    = "generated_data_N10_m10.csv"   

BATCH_SIZE   = 512
MAX_EPOCHS   = 300
PATIENCE     = 20
LR           = 3e-3
WEIGHT_DECAY = 1e-4
VAL_FRAC     = 0.10
TEST_FRAC    = 0.10
SEED         = 42

HIDDEN       = 256
N_BLOCKS     = 5
DROPOUT      = 0.15


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import joblib
import re
import os

torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data and auto-detect N, m
print("Loading data...")
df = pd.read_csv(DATA_PATH)

TARGET_COLS  = [c for c in df.columns if c.startswith("kf_")]
feature_cols = [c for c in df.columns if c not in TARGET_COLS]

# Infer m from number of kf_ columns, N from filename or feature structure
M = len(TARGET_COLS)

match = re.search(r'N(\d+)_m(\d+)', DATA_PATH)
if match:
    N = int(match.group(1))
    assert int(match.group(2)) == M, "m in filename doesn't match kf_ columns"
else:
    # Fall back: infer N from number of a_i columns
    N = sum(1 for c in feature_cols if re.fullmatch(r'a\d+', c))

IN_DIM  = len(feature_cols)   # should equal N*(1+M)+1
OUT_DIM = M

print(f"  Detected: N={N}, M={M} | features={IN_DIM}, targets={OUT_DIM}")
print(f"  Samples: {len(df):,}")
assert IN_DIM == N * (1 + M) + 1, (
    f"Feature count mismatch: got {IN_DIM}, expected {N*(1+M)+1}"
)

X_raw = df[feature_cols].values.astype(np.float32)
y_raw = df[TARGET_COLS].values.astype(np.float32)

# Scale 
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X_raw).astype(np.float32)
y = scaler_y.fit_transform(y_raw).astype(np.float32)

joblib.dump(scaler_X, f"scaler_X_N{N}_m{M}.pkl")
joblib.dump(scaler_y, f"scaler_y_N{N}_m{M}.pkl")

# Split
dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
n       = len(dataset)
n_test  = int(n * TEST_FRAC)
n_val   = int(n * VAL_FRAC)
n_train = n - n_val - n_test
train_ds, val_ds, test_ds = random_split(
    dataset, [n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(SEED)
)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  pin_memory=True)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

# Model
class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim), nn.SiLU(),
            nn.Linear(dim, dim), nn.Dropout(dropout),
            nn.LayerNorm(dim), nn.SiLU(),
            nn.Linear(dim, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.block(x)


class KfNet(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=256, n_blocks=5, dropout=0.15):
        super().__init__()
        self.stem   = nn.Linear(in_dim, hidden)
        self.blocks = nn.Sequential(*[ResBlock(hidden, dropout) for _ in range(n_blocks)])
        self.head   = nn.Sequential(nn.LayerNorm(hidden), nn.SiLU(), nn.Linear(hidden, out_dim))

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


model = KfNet(IN_DIM, OUT_DIM, HIDDEN, N_BLOCKS, DROPOUT).to(DEVICE)
print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Train {n_train:,} | Val {n_val:,} | Test {n_test:,} | Device: {DEVICE}\n")

# Training
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=LR, epochs=MAX_EPOCHS,
    steps_per_epoch=len(train_dl), pct_start=0.1, anneal_strategy='cos',
)
criterion = nn.HuberLoss(delta=1.0)

best_val_loss  = float('inf')
patience_ctr   = 0
train_losses, val_losses = [], []
model_path = f"best_kf_model_N{N}_m{M}.pt"

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    run = 0.0
    for xb, yb in train_dl:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); scheduler.step()
        run += loss.item() * len(xb)
    tl = run / n_train

    model.eval()
    run = 0.0
    with torch.no_grad():
        for xb, yb in val_dl:
            run += criterion(model(xb.to(DEVICE)), yb.to(DEVICE)).item() * len(xb)
    vl = run / n_val

    train_losses.append(tl); val_losses.append(vl)
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:>4} | train={tl:.5f} | val={vl:.5f} | lr={scheduler.get_last_lr()[0]:.2e}")

    if vl < best_val_loss - 1e-6:
        best_val_loss, patience_ctr = vl, 0
        torch.save(model.state_dict(), model_path)
    else:
        patience_ctr += 1
        if patience_ctr >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

# Test evaluation 
model.load_state_dict(torch.load(model_path))
model.eval()

preds, tgts = [], []
with torch.no_grad():
    for xb, yb in test_dl:
        preds.append(model(xb.to(DEVICE)).cpu().numpy())
        tgts.append(yb.numpy())

preds_orig = scaler_y.inverse_transform(np.vstack(preds))
tgts_orig  = scaler_y.inverse_transform(np.vstack(tgts))

print("\n Test R² per target")
for j, col in enumerate(TARGET_COLS):
    print(f"  {col}: R² = {r2_score(tgts_orig[:, j], preds_orig[:, j]):.4f}")
print(f"  Overall R²: {r2_score(tgts_orig, preds_orig, multioutput='uniform_average'):.4f}")