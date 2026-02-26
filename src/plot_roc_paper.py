# src/plot_roc_paper.py
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from xgboost import XGBClassifier

from src.data import load_cardio


def main(seed: int = 0, n_train: int = 8000, n_test: int = 3000):
    # 1) Load
    df = load_cardio("data/cardio_train.csv")
    X = df.drop(columns=["cardio"]).values.astype(np.float32)
    y = df["cardio"].values.astype(np.int64)

    # 2) Split (paper style 80/20 stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    # 3) Subsample for CPU speed (ROC doesn’t need full 70k to look correct)
    rng = np.random.RandomState(seed)
    tr_idx = rng.choice(len(X_train), size=min(n_train, len(X_train)), replace=False)
    te_idx = rng.choice(len(X_test), size=min(n_test, len(X_test)), replace=False)

    X_train = X_train[tr_idx]
    y_train = y_train[tr_idx]
    X_test = X_test[te_idx]
    y_test = y_test[te_idx]

    print(f"Using train={len(X_train)}, test={len(X_test)}")

    # 4) Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    # 5) Models (7 baselines like the paper)
    models = {
        "LogisticRegression": LogisticRegression(max_iter=3000),
        "GaussianNB": GaussianNB(),
        "KNeighborsClassifier": KNeighborsClassifier(n_neighbors=15),
        "XGBClassifier": XGBClassifier(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=-1,
            eval_metric="logloss",
        ),
        "DecisionTreeClassifier": DecisionTreeClassifier(random_state=seed),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=400, random_state=seed, n_jobs=-1
        ),
        # NOTE: this is the slow one. Keep sample sizes modest.
        "SVC": SVC(kernel="rbf", probability=True, random_state=seed),
    }

    # 6) Plot ROC
    plt.figure(figsize=(8, 6))
    for name, model in models.items():
        print("Training:", name)
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)

        plt.plot(fpr, tpr, label=f"{name}, AUC={auc:.3f}")

    # random line
    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.title("ROC Curve Analysis")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right", framealpha=0.95)
    plt.grid(True)

    plt.tight_layout()
    out_path = "outputs/roc_paper_7models.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("Saved", out_path)


if __name__ == "__main__":
    main()