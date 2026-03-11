#!/usr/bin/env python3
"""
Train a tiny MLP to classify valid / invalid hand positions.

Input:  hand_data.npz  (produced by collect_data.py)
Output: hand_validator.npz  (weights + biases + normalisation stats)

Architecture:  42 → 32 → 16 → 1  (sigmoid output)
Trained with:  binary cross-entropy, Adam, optional early stopping
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ── reproducibility ─────────────────────────────────────────────────────────
RNG_SEED   = 42
np.random.seed(RNG_SEED)

# ── hyper-parameters ─────────────────────────────────────────────────────────
HIDDEN1    = 32
HIDDEN2    = 16
LR         = 1e-3
EPOCHS     = 300
BATCH_SIZE = 32
VAL_SPLIT  = 0.2
PATIENCE   = 25          # early-stopping patience (epochs without improvement)
DATA_FILE  = "hand_data.npz"
MODEL_FILE = "hand_validator.npz"


# ── activations ──────────────────────────────────────────────────────────────
def relu(x):        return np.maximum(0, x)
def relu_d(x):      return (x > 0).astype(np.float32)
def sigmoid(x):     return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
def bce(y, p):      # binary cross-entropy
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


# ── weight init ──────────────────────────────────────────────────────────────
def he(fan_in, fan_out):
    return np.random.randn(fan_in, fan_out).astype(np.float32) * np.sqrt(2.0 / fan_in)


# ── forward pass ─────────────────────────────────────────────────────────────
def forward(X, params):
    W1, b1, W2, b2, W3, b3 = (params[k] for k in ("W1","b1","W2","b2","W3","b3"))
    z1 = X @ W1 + b1;   a1 = relu(z1)
    z2 = a1 @ W2 + b2;  a2 = relu(z2)
    z3 = a2 @ W3 + b3;  a3 = sigmoid(z3)
    return a3.squeeze(-1), (z1, a1, z2, a2, z3, a3)


# ── backward pass (manual, no autograd) ──────────────────────────────────────
def backward(X, y, params, cache):
    W1, b1, W2, b2, W3, b3 = (params[k] for k in ("W1","b1","W2","b2","W3","b3"))
    z1, a1, z2, a2, z3, a3 = cache
    n = X.shape[0]

    dz3 = (a3.squeeze(-1) - y).reshape(-1, 1) / n
    dW3 = a2.T @ dz3;  db3 = dz3.sum(0)

    da2 = dz3 @ W3.T
    dz2 = da2 * relu_d(z2)
    dW2 = a1.T @ dz2;  db2 = dz2.sum(0)

    da1 = dz2 @ W2.T
    dz1 = da1 * relu_d(z1)
    dW1 = X.T @ dz1;   db1 = dz1.sum(0)

    return dict(W1=dW1, b1=db1, W2=dW2, b2=db2, W3=dW3, b3=db3)


# ── Adam optimiser state ──────────────────────────────────────────────────────
def adam_init(params):
    return {k: np.zeros_like(v) for k, v in params.items()}, \
           {k: np.zeros_like(v) for k, v in params.items()}

def adam_step(params, grads, m, v, t,
              lr=LR, beta1=0.9, beta2=0.999, eps=1e-8):
    t += 1
    for k in params:
        m[k] = beta1 * m[k] + (1 - beta1) * grads[k]
        v[k] = beta2 * v[k] + (1 - beta2) * grads[k]**2
        m_hat = m[k] / (1 - beta1**t)
        v_hat = v[k] / (1 - beta2**t)
        params[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)
    return params, m, v, t


# ── metrics ───────────────────────────────────────────────────────────────────
def accuracy(y, p, thresh=0.5):
    return np.mean((p >= thresh) == y.astype(bool))

def confusion(y, p, thresh=0.5):
    pred = (p >= thresh)
    tp = np.sum(pred & y.astype(bool))
    tn = np.sum(~pred & ~y.astype(bool))
    fp = np.sum(pred & ~y.astype(bool))
    fn = np.sum(~pred & y.astype(bool))
    return tp, tn, fp, fn


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    # ── load data ────────────────────────────────────────────────────────────
    if not Path(DATA_FILE).exists():
        raise FileNotFoundError(f"{DATA_FILE} not found — run collect_data.py first.")

    data = np.load(DATA_FILE)
    X, y = data["X"].astype(np.float32), data["y"].astype(np.float32)
    print(f"Loaded {X.shape[0]} samples  (valid={int(y.sum())}, invalid={int((1-y).sum())})")

    # ── normalise ────────────────────────────────────────────────────────────
    mu  = X.mean(0)
    std = X.std(0) + 1e-8
    X   = (X - mu) / std

    # ── train / val split ────────────────────────────────────────────────────
    idx = np.random.permutation(len(X))
    val_n = int(len(X) * VAL_SPLIT)
    val_idx, tr_idx = idx[:val_n], idx[val_n:]
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    print(f"Train: {len(X_tr)}  Val: {len(X_val)}")

    # ── init weights ─────────────────────────────────────────────────────────
    params = dict(
        W1=he(42, HIDDEN1), b1=np.zeros((1, HIDDEN1), dtype=np.float32),
        W2=he(HIDDEN1, HIDDEN2), b2=np.zeros((1, HIDDEN2), dtype=np.float32),
        W3=he(HIDDEN2, 1),      b3=np.zeros((1, 1),       dtype=np.float32),
    )
    m, v = adam_init(params)
    t = 0

    # ── training loop ─────────────────────────────────────────────────────────
    tr_losses, val_losses, tr_accs, val_accs = [], [], [], []
    best_val_loss = np.inf
    best_params   = None
    patience_ctr  = 0

    n_batches = int(np.ceil(len(X_tr) / BATCH_SIZE))

    for epoch in range(1, EPOCHS + 1):
        # shuffle
        perm = np.random.permutation(len(X_tr))
        X_s, y_s = X_tr[perm], y_tr[perm]

        epoch_loss = 0.0
        for b in range(n_batches):
            Xb = X_s[b*BATCH_SIZE:(b+1)*BATCH_SIZE]
            yb = y_s[b*BATCH_SIZE:(b+1)*BATCH_SIZE]
            p_b, cache = forward(Xb, params)
            epoch_loss += bce(yb, p_b) * len(Xb)
            grads = backward(Xb, yb, params, cache)
            params, m, v, t = adam_step(params, grads, m, v, t)

        tr_loss = epoch_loss / len(X_tr)
        p_tr,  _ = forward(X_tr,  params)
        p_val, _ = forward(X_val, params)
        val_loss  = bce(y_val, p_val)
        tr_acc    = accuracy(y_tr,  p_tr)
        val_acc   = accuracy(y_val, p_val)

        tr_losses.append(tr_loss);  val_losses.append(val_loss)
        tr_accs.append(tr_acc);     val_accs.append(val_acc)

        # early stopping
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_params   = {k: v_.copy() for k, v_ in params.items()}
            patience_ctr  = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"Early stop at epoch {epoch}")
                break

        if epoch % 25 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | "
                  f"loss {tr_loss:.4f}/{val_loss:.4f} | "
                  f"acc {tr_acc:.3f}/{val_acc:.3f}")

    # ── restore best weights & final eval ────────────────────────────────────
    params = best_params
    p_val, _ = forward(X_val, params)
    tp, tn, fp, fn = confusion(y_val, p_val)
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    print("\n── Final Validation Metrics ──────────────────────")
    print(f"  Accuracy  : {accuracy(y_val, p_val):.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1        : {f1:.4f}")
    print(f"  Confusion : TP={tp} TN={tn} FP={fp} FN={fn}")

    # ── save model ───────────────────────────────────────────────────────────
    np.savez(MODEL_FILE,
             mu=mu, std=std,
             **{k: v_ for k, v_ in params.items()})
    print(f"\nModel saved → {MODEL_FILE}")

    # ── plots ────────────────────────────────────────────────────────────────
    epochs_ran = len(tr_losses)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(range(1, epochs_ran+1), tr_losses,  label="Train loss")
    axes[0].plot(range(1, epochs_ran+1), val_losses, label="Val loss")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch")
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(range(1, epochs_ran+1), tr_accs,  label="Train acc")
    axes[1].plot(range(1, epochs_ran+1), val_accs, label="Val acc")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1); axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=130)
    print("Training curves saved → training_curves.png")
    plt.show()


if __name__ == "__main__":
    main()