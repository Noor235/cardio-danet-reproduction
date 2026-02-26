# src/plot_roc.py
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import torch
from src.data import load_cardio
from src.danet_model import DANetClassifier
from src.train_utils import predict_proba


def main(seed: int = 0, n_train: int = 10000, n_test: int = 3000):

    print("[1/7] Loading dataset...")
    df = load_cardio("data/cardio_train.csv")

    X = df.drop(columns=["cardio"]).values.astype(np.float32)
    y = df["cardio"].values.astype(np.int64)

    print("[2/7] Train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    print("[3/7] Sampling smaller subset for speed...")
    rng = np.random.RandomState(seed)

    tr_idx = rng.choice(len(X_train), size=min(n_train, len(X_train)), replace=False)
    te_idx = rng.choice(len(X_test), size=min(n_test, len(X_test)), replace=False)

    X_train = X_train[tr_idx]
    y_train = y_train[tr_idx]
    X_test = X_test[te_idx]
    y_test = y_test[te_idx]

    print(f"Using train={len(X_train)}, test={len(X_test)}")

    print("[4/7] Scaling features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    curves = []

    print("[5/7] Training baseline models...")

    lr = LogisticRegression(max_iter=2000)
    lr.fit(X_train, y_train)
    curves.append(("LogReg", lr.predict_proba(X_test)[:, 1]))

    rf = RandomForestClassifier(n_estimators=150, random_state=seed, n_jobs=-1)
    rf.fit(X_train, y_train)
    curves.append(("RF", rf.predict_proba(X_test)[:, 1]))

    xgb = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        eval_metric="logloss",
    )
    xgb.fit(X_train, y_train)
    curves.append(("XGBoost", xgb.predict_proba(X_test)[:, 1]))

    print("[6/7] Loading DANet model + predicting (with progress bar)...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    danet = DANetClassifier(
        in_dim=X_train.shape[1],
        hidden_dim=128,
        num_layers=4,
        groups=16,
        dropout=0.15,
    )

    danet.load_state_dict(torch.load("outputs/danet_model.pt", map_location=device))
    danet.to(device).eval()

    batch_size = 128
    probs = []

    total_batches = (len(X_test) + batch_size - 1) // batch_size

    for i in tqdm(range(0, len(X_test), batch_size),
                  total=total_batches,
                  desc="DANet Predicting",
                  ncols=80):
        xb = X_test[i:i + batch_size]
        probs.append(predict_proba(danet, xb, device=device))

    curves.append(("DANet", np.concatenate(probs)))

    print("[7/7] Plotting ROC curves...")

    plt.figure(figsize=(7, 6))

    for name, proba in curves:
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")

    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.title("ROC Curves (Sampled for CPU Efficiency)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("outputs/roc_all_models_sampled.png", dpi=300)
    plt.close()

    print("Saved outputs/roc_all_models_sampled.png")


if __name__ == "__main__":
    main()