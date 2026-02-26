import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier

DATA_PATH = "data/cardio_train.csv"
SEED = 0

FEATURES = [
    "age","gender","height","weight","ap_hi","ap_lo",
    "cholesterol","gluc","smoke","alco","active"
]
TARGET = "cardio"

# Load dataset
df = pd.read_csv(DATA_PATH, sep=";")
if "id" in df.columns:
    df = df.drop(columns=["id"])

X = df[FEATURES].values
y = df[TARGET].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train XGBoost
model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    random_state=SEED,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

# Default threshold = 0.5
y_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= 0.5).astype(int)

cm = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = cm.ravel()

sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
fn_rate = fn / (fn + tp)

print("Confusion Matrix:")
print(cm)
print(f"Sensitivity: {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"False Negative Rate: {fn_rate:.4f}")