# src/shap_danet.py
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import shap
import matplotlib.pyplot as plt

from src.data import load_cardio, split_and_scale
from src.danet_model import DANetClassifier
from src.train_utils import predict_proba


class WrappedDANet2(torch.nn.Module):
    """
    Return 2-class logits (B, 2) so SHAP never sees a 1D tensor.
    class0_logit = -logit, class1_logit = +logit
    """
    def __init__(self, base: DANetClassifier):
        super().__init__()
        self.base = base

    def forward(self, x):
        logit = self.base(x)  # (B,)
        logits2 = torch.stack([-logit, logit], dim=1)  # (B, 2)
        return logits2


def main(seed: int = 0, n_background: int = 256, n_explain: int = 512):
    df = load_cardio("data/cardio_train.csv")
    X_train, X_test, y_train, y_test, _ = split_and_scale(df, seed=seed)
    feature_names = df.drop(columns=["cardio"]).columns.tolist()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load trained DANet
    base_model = DANetClassifier(in_dim=X_train.shape[1], hidden_dim=128, num_layers=4, groups=16, dropout=0.15)
    base_model.load_state_dict(torch.load("outputs/danet_model.pt", map_location=device))
    base_model.to(device).eval()

    model = WrappedDANet2(base_model).to(device).eval()

    # Subsets for speed
    bg_idx = np.random.choice(len(X_train), size=min(n_background, len(X_train)), replace=False)
    ex_idx = np.random.choice(len(X_test), size=min(n_explain, len(X_test)), replace=False)

    X_bg = torch.tensor(X_train[bg_idx], dtype=torch.float32).to(device)
    X_ex = torch.tensor(X_test[ex_idx], dtype=torch.float32).to(device)

    # DEBUG: confirm shape is (B,2)
    with torch.no_grad():
        out = model(X_ex[:8])
    print("Wrapped model output shape:", tuple(out.shape))  # must be (8,2)

    explainer = shap.GradientExplainer(model, X_bg)

    shap_vals = explainer.shap_values(X_ex)
    # shap_vals may be list (one per output class)
    if isinstance(shap_vals, list):
        # take class-1 explanation (cardio=1)
        shap_vals = shap_vals[1]

    shap_vals_np = np.array(shap_vals)           # (n, d)
    X_ex_np = X_ex.detach().cpu().numpy()

    # Global plots
    shap.summary_plot(shap_vals_np, X_ex_np, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig("outputs/shap_danet_beeswarm.png", dpi=200)
    plt.close()

    shap.summary_plot(shap_vals_np, X_ex_np, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig("outputs/shap_danet_bar.png", dpi=200)
    plt.close()

    # Ranking
    mean_abs = np.abs(shap_vals_np).mean(axis=0)
    rank = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs}).sort_values(
        "mean_abs_shap", ascending=False
    )
    rank.to_csv("outputs/shap_danet_feature_ranking.csv", index=False)

    # Local waterfall (first sample)
    i = 0
    # baseline logit for class 1: use mean of class-1 logit on background
    with torch.no_grad():
        base_val = model(X_bg)[:, 1].mean().detach().cpu().item()

    shap.waterfall_plot(
        shap.Explanation(
            values=shap_vals_np[i],
            base_values=base_val,
            data=X_ex_np[i],
            feature_names=feature_names,
        ),
        show=False
    )
    plt.tight_layout()
    plt.savefig("outputs/shap_danet_waterfall.png", dpi=200)
    plt.close()

    # Prediction for the same sample using the real model
    proba = predict_proba(base_model, X_ex_np[i:i+1], device=device)[0]
    print("Example sample prediction proba(cardio=1):", float(proba))

    print("Saved:")
    print(" - outputs/shap_danet_beeswarm.png")
    print(" - outputs/shap_danet_bar.png")
    print(" - outputs/shap_danet_feature_ranking.csv")
    print(" - outputs/shap_danet_waterfall.png")


if __name__ == "__main__":
    main()