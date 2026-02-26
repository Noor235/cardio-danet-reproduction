# src/predict_example.py
from __future__ import annotations

import numpy as np
import torch
import joblib

from src.data import load_cardio, split_and_scale
from src.danet_model import DANetClassifier
from src.train_utils import predict_proba


# Always show high-risk case
INITIAL_THRESHOLD = 0.8
MIN_THRESHOLD = 0.6
STEP = 0.05


def main():
    df = load_cardio("data/cardio_train.csv")

    # Load training seed
    with open("outputs/danet_seed.txt", "r") as f:
        seed = int(f.read().strip())

    X_train, X_test, y_train, y_test, _ = split_and_scale(df, seed=seed)
    scaler = joblib.load("outputs/danet_scaler.pkl")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    print("Split seed:", seed)

    model = DANetClassifier(
        in_dim=X_train.shape[1],
        hidden_dim=128,
        num_layers=4,
        groups=16,
        dropout=0.15,
    )
    model.load_state_dict(torch.load("outputs/danet_model.pt", map_location=device))
    model.to(device)
    model.eval()

    # Predict whole test set
    proba_all = predict_proba(model, X_test, device=device).flatten()
    pred_all = (proba_all >= 0.5).astype(int)
    y_true = y_test.astype(int)

    # Try to find high-risk patient
    threshold = INITIAL_THRESHOLD
    idx = None

    while threshold >= MIN_THRESHOLD:
        candidates = np.where(proba_all >= threshold)[0]
        if len(candidates) > 0:
            idx = int(np.random.choice(candidates))
            break
        threshold -= STEP

    if idx is None:
        raise RuntimeError("No high-risk cases found even at lower threshold.")

    # Prepare display
    sample_scaled = X_test[idx:idx+1]
    sample_original = scaler.inverse_transform(sample_scaled)

    proba = float(proba_all[idx])
    pred_label = int(pred_all[idx])
    true_label = int(y_true[idx])

    feature_names = df.drop(columns=["cardio"]).columns.tolist()

    print("\n===== HIGH-RISK PATIENT SAMPLE =====")
    print("Test index:", idx)
    print("Used threshold:", round(threshold, 2))
    print("\nFeatures (original scale):")

    for k, v in zip(feature_names, sample_original.flatten()):
        print(f"{k:12s}: {float(v):.3f}")

    print("\nPrediction probability (cardio=1):", round(proba, 4))
    print("Predicted label:", pred_label)
    print("True label:", true_label)
    print("Result:", "✅ Correct" if pred_label == true_label else "❌ Wrong")


if __name__ == "__main__":
    main()