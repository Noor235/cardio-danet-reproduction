# src/data.py
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CARDIO_COLUMNS = [
    "age", "gender", "height", "weight", "ap_hi", "ap_lo",
    "cholesterol", "gluc", "smoke", "alco", "active", "cardio"
]

def load_cardio(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    missing = [c for c in CARDIO_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}\nFound: {list(df.columns)}")

    # age is in days -> years
    df["age"] = df["age"] / 365.25

    # gender often 1/2 -> convert to 0/1 (0=female, 1=male)
    if set(df["gender"].unique()).issubset({1, 2}):
        df["gender"] = (df["gender"] == 2).astype(int)

    df["cardio"] = df["cardio"].astype(int)
    return df

def split_and_scale(
    df: pd.DataFrame,
    target_col: str = "cardio",
    test_size: float = 0.2,
    seed: int = 42,
):
    X = df.drop(columns=[target_col]).values.astype(np.float32)
    y = df[target_col].values.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    return X_train, X_test, y_train, y_test, scaler