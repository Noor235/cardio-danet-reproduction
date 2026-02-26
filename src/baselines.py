# src/baselines.py
from __future__ import annotations
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.data import load_cardio, split_and_scale
from src.metrics import compute_metrics


def run_baselines(
    data_path: str = "data/cardio_train.csv",
    seeds: list[int] = [0, 1, 2, 3, 4],
):
    df = load_cardio(data_path)

    rows = []
    for seed in seeds:
        X_train, X_test, y_train, y_test, _ = split_and_scale(df, seed=seed)

        models = {
            "LogReg": LogisticRegression(max_iter=3000),
            "RF": RandomForestClassifier(
                n_estimators=400, random_state=seed, n_jobs=-1
            ),
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

        for name, model in models.items():
            model.fit(X_train, y_train)
            proba = model.predict_proba(X_test)[:, 1]
            pred = (proba >= 0.5).astype(int)

            m = compute_metrics(y_test, pred, proba)
            print(f"[{name}] seed={seed} acc={m.accuracy:.4f} f1={m.f1:.4f} auc={m.auc:.4f}")

            rows.append({
                "model": name,
                "seed": seed,
                "accuracy": m.accuracy,
                "f1": m.f1,
                "auc": m.auc,
            })

    out = pd.DataFrame(rows)
    out.to_csv("outputs/baselines_results.csv", index=False)

    summary = out.groupby("model")[["accuracy", "f1", "auc"]].agg(["mean", "std"]).reset_index()
    summary.to_csv("outputs/baselines_summary.csv", index=False)

    print("\nSaved:")
    print(" - outputs/baselines_results.csv")
    print(" - outputs/baselines_summary.csv")


if __name__ == "__main__":
    run_baselines()