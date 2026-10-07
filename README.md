# MLOps pipeline: ML + DL with MLflow, DVC, Git, GitHub Actions

Digits classification: RandomForest (ML) and PyTorch MLP (DL), tracked in MLflow, pipeline in DVC.

## Flow
```
git push -> CI  (flake8, pytest, dvc repro + quality gate)
main/tag -> CD  (calls CT -> docker build -> GHCR -> smoke test)
cron/manual -> CT (retrain, register models in MLflow registry)
```

## Run locally
```bash
git init && dvc init
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# terminal 1: MLflow web GUI -> http://127.0.0.1:5000
mlflow server --host 127.0.0.1 --port 5000 \
  --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts

# terminal 2
dvc repro            # prepare -> train_ml -> train_dl -> evaluate
dvc metrics show
dvc params diff
```
Change `params.yaml`, run `dvc repro` again, compare runs in the MLflow UI (select runs -> Compare).

## Push to GitHub
```bash
git add . && git commit -m "init mlops pipeline"
git branch -M main
git remote add origin https://github.com/<user>/<repo>.git
git push -u origin main
```
Repo Settings -> Actions -> Workflow permissions: allow read and write (for GHCR).

## View MLflow GUI of a CI/CT run
Actions -> run -> Artifacts -> download `mlflow-results-*`, unzip in repo root, then start the
`mlflow server` command above and open http://127.0.0.1:5000.

## Optional DVC remote
```bash
dvc remote add -d storage <gdrive://... | s3://... | ../dvc-storage>
dvc push
```
