# src/check_dataset.py
from src.data import load_cardio, split_and_scale

def main():
    df = load_cardio("data/cardio_train.csv")

    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("\nHead:")
    print(df.head())

    X_train, X_test, y_train, y_test, _ = split_and_scale(df, seed=0)
    print("\nSplit shapes:")
    print("Train:", X_train.shape, "Test:", X_test.shape)
    print("Positive rate train:", y_train.mean(), "test:", y_test.mean())

if __name__ == "__main__":
    main()