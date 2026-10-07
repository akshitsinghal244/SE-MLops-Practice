import os

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from common import params


def main():
    p = params()["data"]
    X, y = load_digits(return_X_y=True, as_frame=True)
    df = X.copy()
    df["target"] = y
    train, test = train_test_split(
        df, test_size=p["test_size"], random_state=p["seed"], stratify=df["target"]
    )
    os.makedirs("data/processed", exist_ok=True)
    train.to_csv("data/processed/train.csv", index=False)
    test.to_csv("data/processed/test.csv", index=False)
    print(f"train={len(train)} test={len(test)}")


if __name__ == "__main__":
    main()
