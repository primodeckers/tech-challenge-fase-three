from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "model.joblib"

app = FastAPI(title="Triagem de laudos")
model = joblib.load(MODEL_PATH)


class PredictIn(BaseModel):
    text: str = Field(min_length=1)


class PredictOut(BaseModel):
    label: str
    proba: dict[str, float]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictOut)
def predict(body: PredictIn):
    proba = model.predict_proba([body.text])[0]
    scores = {str(cls): float(p) for cls, p in zip(model.classes_, proba)}
    label = max(scores, key=scores.get)
    return PredictOut(label=label, proba=scores)
