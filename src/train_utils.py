# src/train_utils.py
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


def make_loaders(X_train, y_train, X_val, y_val, batch_size: int = 512):
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


@torch.no_grad()
def predict_proba(model, X, device: str):
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    logits = model(X_t)
    return torch.sigmoid(logits).detach().cpu().numpy()


def train_model(
    model,
    train_loader,
    val_loader,
    device: str,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 30,
    patience: int = 6,
):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best_val = float("inf")
    best_state = None
    bad = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            val_losses.append(loss.item())

        tr = float(np.mean(train_losses))
        va = float(np.mean(val_losses))
        print(f"[DANet] epoch={epoch} train_loss={tr:.4f} val_loss={va:.4f}")

        if va < best_val - 1e-5:
            best_val = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print("[DANet] Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model