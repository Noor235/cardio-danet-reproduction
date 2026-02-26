# src/train_danet.py
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import joblib

from src.data import load_cardio, split_and_scale
from src.metrics import compute_metrics
from src.danet_model import DANetClassifier
from src.train_utils import make_loaders, train_model, predict_proba


def run_one_seed(seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    df = load_cardio("data/cardio_train.csv")
    X_train, X_test, y_train, y_test, scaler = split_and_scale(df, seed=seed)

    # val split for early stopping
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=0.15,
        random_state=seed,
        stratify=y_train
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    model = DANetClassifier(
        in_dim=X_train.shape[1],
        hidden_dim=128,
        num_layers=4,
        groups=16,
        dropout=0.15,
    )

    train_loader, val_loader = make_loaders(X_tr, y_tr, X_val, y_val, batch_size=512)

    model = train_model(
        model,
        train_loader,
        val_loader,
        device=device,
        lr=1e-3,
        weight_decay=1e-4,
        epochs=30,
        patience=6,
    )

    proba = predict_proba(model, X_test, device=device)
    pred = (proba >= 0.5).astype(int)

    m = compute_metrics(y_test, pred, proba)
    print(f"[DANet] seed={seed} acc={m.accuracy:.4f} f1={m.f1:.4f} auc={m.auc:.4f}")

    # SAVE EVERYTHING NEEDED TO MATCH PREDICTIONS
    torch.save(model.state_dict(), "outputs/danet_model.pt")
    joblib.dump(scaler, "outputs/danet_scaler.pkl")
    with open("outputs/danet_seed.txt", "w") as f:
        f.write(str(seed))

    return {
        "model": "DANet",
        "seed": seed,
        "accuracy": m.accuracy,
        "f1": m.f1,
        "auc": m.auc,
        "device": device,
    }


if __name__ == "__main__":
    row = run_one_seed(seed=0)
    pd.DataFrame([row]).to_csv("outputs/danet_one_seed.csv", index=False)
    print("Saved outputs/danet_one_seed.csv")