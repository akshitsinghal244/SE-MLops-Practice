import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import yaml  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay  # noqa: E402

EXPERIMENT = "digits-classification"


def params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def setup_mlflow():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment(EXPERIMENT)


def base_tags(model_type):
    return {
        "model_type": model_type,
        "git_sha": os.getenv("GITHUB_SHA", "local"),
        "trigger": os.getenv("GITHUB_EVENT_NAME", "manual"),
    }


def register_name(name):
    return name if os.getenv("REGISTER_MODEL") == "1" else None


def log_confusion_matrix(y_true, y_pred, name):
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, colorbar=False)
    ax.set_title(name)
    mlflow.log_figure(fig, f"{name}_confusion_matrix.png")
    plt.close(fig)
