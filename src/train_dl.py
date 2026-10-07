import json
import os

import mlflow
import mlflow.pytorch
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

from common import base_tags, log_confusion_matrix, params, register_name, setup_mlflow


def build_model(in_dim, hidden, n_classes=10):
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden, hidden // 2),
        nn.ReLU(),
        nn.Linear(hidden // 2, n_classes),
    )


def to_tensors(df):
    X = torch.tensor(df.drop(columns="target").values / 16.0, dtype=torch.float32)
    y = torch.tensor(df["target"].values, dtype=torch.long)
    return X, y


def main():
    cfg = params()
    p, seed = cfg["dl"], cfg["data"]["seed"]
    torch.manual_seed(seed)
    Xtr, ytr = to_tensors(pd.read_csv("data/processed/train.csv"))
    Xte, yte = to_tensors(pd.read_csv("data/processed/test.csv"))
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=p["batch_size"], shuffle=True)

    model = build_model(Xtr.shape[1], p["hidden"])
    opt = torch.optim.Adam(model.parameters(), lr=p["lr"])
    loss_fn = nn.CrossEntropyLoss()

    setup_mlflow()
    with mlflow.start_run(run_name="pytorch_mlp"):
        mlflow.set_tags(base_tags("dl"))
        mlflow.log_params(p)
        mlflow.log_artifact("params.yaml")

        for epoch in range(p["epochs"]):
            model.train()
            total = 0.0
            for xb, yb in loader:
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                opt.step()
                total += loss.item() * len(xb)
            model.eval()
            with torch.no_grad():
                logits = model(Xte)
                val_loss = loss_fn(logits, yte).item()
                val_acc = (logits.argmax(1) == yte).float().mean().item()
            mlflow.log_metrics(
                {"train_loss": total / len(Xtr), "val_loss": val_loss, "val_accuracy": val_acc},
                step=epoch,
            )

        pred = logits.argmax(1).numpy()
        acc = accuracy_score(yte.numpy(), pred)
        f1 = f1_score(yte.numpy(), pred, average="macro")
        mlflow.log_metrics({"accuracy": acc, "f1_macro": f1})
        log_confusion_matrix(yte.numpy(), pred, "pytorch_mlp")
        mlflow.pytorch.log_model(
            model, artifact_path="model", registered_model_name=register_name("digits-mlp")
        )

    os.makedirs("models", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)
    torch.save(model.state_dict(), "models/dl_model.pt")
    with open("metrics/dl.json", "w") as f:
        json.dump({"accuracy": acc, "f1_macro": f1}, f, indent=2)
    print(f"PyTorch MLP accuracy={acc:.4f}")


if __name__ == "__main__":
    main()
