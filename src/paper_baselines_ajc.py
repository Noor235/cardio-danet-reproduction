# src/paper_baselines_ajc.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    roc_auc_score, brier_score_loss
)

from xgboost import XGBClassifier


def find_cardio_csv(project_root: Path) -> Path:
    hits = list(project_root.rglob("cardio_train.csv"))
    if not hits:
        raise FileNotFoundError(f"Could not find cardio_train.csv under: {project_root}")
    return hits[0]


def load_cardio(csv_path: Path):
    # Kaggle cardio sometimes uses ';'
    df = pd.read_csv(csv_path, sep=";")
    if df.shape[1] == 1:
        df = pd.read_csv(csv_path)

    if "id" in df.columns:
        df = df.drop(columns=["id"])

    if "cardio" not in df.columns:
        raise ValueError("Target column 'cardio' not found.")

    y = df["cardio"].astype(int).to_numpy()
    X = df.drop(columns=["cardio"])

    return X, y


def eval_one_split(X, y, seed: int):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=3000),
        "RandomForest": RandomForestClassifier(n_estimators=400, random_state=seed, n_jobs=-1),
        "DecisionTree": DecisionTreeClassifier(random_state=seed),
        "KNeighbors": KNeighborsClassifier(n_neighbors=15),
        "SVM": SVC(kernel="rbf", probability=True, random_state=seed),
        "GaussianNB": GaussianNB(),
        "XGBoost": XGBClassifier(
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
    }

    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)

        rows.append({
            "seed": seed,
            "model": name,
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "accuracy": accuracy_score(y_test, pred),
            "auc": roc_auc_score(y_test, proba),
            "brier": brier_score_loss(y_test, proba),
        })

    return pd.DataFrame(rows)


def main(seeds=(0, 1, 2, 3, 4)):
    project_root = Path(__file__).resolve().parents[1]
    csv_path = find_cardio_csv(project_root)
    print("Using dataset:", csv_path)

    X, y = load_cardio(csv_path)

    all_rows = []
    for seed in seeds:
        print("Running seed:", seed)
        all_rows.append(eval_one_split(X, y, seed))

    results = pd.concat(all_rows, ignore_index=True)

    out_dir = project_root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    results_path = out_dir / "ajc_baselines_results.csv"
    results.to_csv(results_path, index=False)

    summary = (
        results.groupby("model")[["precision", "recall", "f1", "accuracy", "auc", "brier"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary_path = out_dir / "ajc_baselines_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\nSaved:")
    print(" -", results_path)
    print(" -", summary_path)
    print("\nTop by mean AUC:")
    print(summary.sort_values(("auc", "mean"), ascending=False)[["model", ("auc", "mean"), ("accuracy", "mean")]].head(10))


if __name__ == "__main__":
    main()