import os

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Digits classifier")
model = joblib.load(os.getenv("MODEL_PATH", "models/ml_model.joblib"))


class Request(BaseModel):
    pixels: list[float]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: Request):
    if len(req.pixels) != 64:
        raise HTTPException(422, "pixels must have 64 values (8x8 image, 0-16)")
    return {"digit": int(model.predict([req.pixels])[0])}
