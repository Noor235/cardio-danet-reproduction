# src/danet_permutation_importance.py
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from src.data import load_cardio, split_and_scale
from src.danet_model import DANetClassifier
from src.train_utils import predict_proba


def main(seed: int = 0, repeats: int = 5):
    df = load_cardio("data/cardio_train.csv")
    X_train, X_test, y_train, y_test, _ = split_and_scale(df, seed=seed)
    feature_names = df.drop(columns=["cardio"]).columns.tolist()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DANetClassifier(in_dim=X_train.shape[1], hidden_dim=128, num_layers=4, groups=16, dropout=0.15)
    model.load_state_dict(torch.load("outputs/danet_model.pt", map_location=device))
    model.to(device).eval()

    # baseline AUC
    base_proba = predict_proba(model, X_test, device=device)
    base_auc = roc_auc_score(y_test, base_proba)
    print("Baseline AUC:", round(float(base_auc), 4))

    rng = np.random.RandomState(seed)
    importances = []

    for j, name in enumerate(feature_names):
        drops = []
        for r in range(repeats):
            X_perm = X_test.copy()
            rng.shuffle(X_perm[:, j])  # permute one feature
            proba = predict_proba(model, X_perm, device=device)
            auc = roc_auc_score(y_test, proba)
            drops.append(base_auc - auc)

        importances.append({
            "feature": name,
            "auc_drop_mean": float(np.mean(drops)),
            "auc_drop_std": float(np.std(drops)),
        })
        print(f"{name:12s} AUC drop mean={np.mean(drops):.5f} std={np.std(drops):.5f}")

    out = pd.DataFrame(importances).sort_values("auc_drop_mean", ascending=False)
    out.to_csv("outputs/danet_permutation_importance.csv", index=False)
    print("Saved outputs/danet_permutation_importance.csv")


if __name__ == "__main__":
    main()