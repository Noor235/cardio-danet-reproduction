# src/shap_xgb.py
from __future__ import annotations
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from src.data import load_cardio, split_and_scale


def main(seed: int = 0, n_background: int = 1000, n_explain: int = 2000):
    df = load_cardio("data/cardio_train.csv")
    X_train, X_test, y_train, y_test, _ = split_and_scale(df, seed=seed)

    # Train XGBoost (same as baseline)
    model = XGBClassifier(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    feature_names = df.drop(columns=["cardio"]).columns.tolist()

    # Use a subset to keep SHAP fast
    bg_idx = np.random.choice(len(X_train), size=min(n_background, len(X_train)), replace=False)
    ex_idx = np.random.choice(len(X_test), size=min(n_explain, len(X_test)), replace=False)

    X_bg = X_train[bg_idx]
    X_ex = X_test[ex_idx]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_ex)

    # Summary plots
    plt.figure()
    shap.summary_plot(shap_values, X_ex, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig("outputs/shap_xgb_beeswarm.png", dpi=200)
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, X_ex, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig("outputs/shap_xgb_bar.png", dpi=200)
    plt.close()

    # Save mean(|shap|) ranking
    mean_abs = np.abs(shap_values).mean(axis=0)
    rank = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs}).sort_values(
        "mean_abs_shap", ascending=False
    )
    rank.to_csv("outputs/shap_xgb_feature_ranking.csv", index=False)

    # Local explanation for 1 random sample
    i = 0
    sample = X_ex[i:i+1]
    expected = explainer.expected_value
    sv = explainer.shap_values(sample)[0]

    plt.figure()
    shap.waterfall_plot(shap.Explanation(values=sv, base_values=expected, data=sample[0], feature_names=feature_names), show=False)
    plt.tight_layout()
    plt.savefig("outputs/shap_xgb_waterfall.png", dpi=200)
    plt.close()

    print("Saved:")
    print(" - outputs/shap_xgb_beeswarm.png")
    print(" - outputs/shap_xgb_bar.png")
    print(" - outputs/shap_xgb_feature_ranking.csv")
    print(" - outputs/shap_xgb_waterfall.png")


if __name__ == "__main__":
    main()