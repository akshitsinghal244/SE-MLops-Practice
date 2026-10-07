import json
import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from common import base_tags, log_confusion_matrix, params, register_name, setup_mlflow


def main():
    cfg = params()
    p, seed = cfg["ml"], cfg["data"]["seed"]
    train = pd.read_csv("data/processed/train.csv")
    test = pd.read_csv("data/processed/test.csv")
    Xtr, ytr = train.drop(columns="target"), train["target"]
    Xte, yte = test.drop(columns="target"), test["target"]

    setup_mlflow()
    with mlflow.start_run(run_name="random_forest"):
        mlflow.set_tags(base_tags("ml"))
        mlflow.log_params(p)
        mlflow.log_artifact("params.yaml")

        model = RandomForestClassifier(
            n_estimators=p["n_estimators"], max_depth=p["max_depth"], random_state=seed
        ).fit(Xtr, ytr)
        pred = model.predict(Xte)
        acc = accuracy_score(yte, pred)
        f1 = f1_score(yte, pred, average="macro")

        mlflow.log_metrics({"accuracy": acc, "f1_macro": f1})
        log_confusion_matrix(yte, pred, "random_forest")
        mlflow.sklearn.log_model(
            model, artifact_path="model", registered_model_name=register_name("digits-rf")
        )

    os.makedirs("models", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)
    joblib.dump(model, "models/ml_model.joblib")
    with open("metrics/ml.json", "w") as f:
        json.dump({"accuracy": acc, "f1_macro": f1}, f, indent=2)
    print(f"RandomForest accuracy={acc:.4f}")


if __name__ == "__main__":
    main()
