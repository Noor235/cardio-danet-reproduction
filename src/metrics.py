# src/metrics.py
from __future__ import annotations
from dataclasses import dataclass
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

@dataclass
class Metrics:
    accuracy: float
    f1: float
    auc: float

def compute_metrics(y_true, y_pred_label, y_pred_proba) -> Metrics:
    acc = accuracy_score(y_true, y_pred_label)
    f1 = f1_score(y_true, y_pred_label)
    auc = roc_auc_score(y_true, y_pred_proba)
    return Metrics(accuracy=acc, f1=f1, auc=auc)