import os
import time
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, Request
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from starlette.responses import Response

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "model.joblib"
ONNX_PATH = ROOT / "artifacts" / "model.onnx"
CLASSES_PATH = ROOT / "artifacts" / "classes.joblib"
USAR_ONNX = os.getenv("MODEL_RUNTIME", "onnx") != "sklearn"

app = FastAPI(title="Triagem de laudos")

if USAR_ONNX:
    sessao = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    INPUT_NAME = sessao.get_inputs()[0].name
    CLASSES = [str(c) for c in joblib.load(CLASSES_PATH)]
    modelo_sklearn = None
else:
    sessao = None
    INPUT_NAME = ""
    modelo_sklearn = joblib.load(MODEL_PATH)
    CLASSES = [str(c) for c in modelo_sklearn.classes_]

REQUESTS = Counter(
    "http_requests_total", "Total de requisicoes", ["endpoint", "method", "status"]
)
LATENCIA = Histogram(
    "http_request_duration_seconds", "Duracao da requisicao em segundos", ["endpoint"]
)
ERROS = Counter("http_errors_total", "Total de respostas 4xx/5xx", ["endpoint", "status"])


@app.middleware("http")
async def metricas(request: Request, call_next):
    inicio = time.perf_counter()
    response = await call_next(request)
    endpoint = request.url.path
    LATENCIA.labels(endpoint=endpoint).observe(time.perf_counter() - inicio)
    REQUESTS.labels(endpoint=endpoint, method=request.method, status=response.status_code).inc()
    if response.status_code >= 400:
        ERROS.labels(endpoint=endpoint, status=response.status_code).inc()
    return response


class PredictIn(BaseModel):
    text: str = Field(min_length=1)


class PredictOut(BaseModel):
    label: str
    proba: dict[str, float]


def classificar(texto: str) -> tuple[str, dict[str, float]]:
    if sessao is not None:
        rotulo, proba = sessao.run(None, {INPUT_NAME: np.array([[texto]])})
        scores = {cls: float(p) for cls, p in zip(CLASSES, proba[0])}
        return str(rotulo[0]), scores
    proba = modelo_sklearn.predict_proba([texto])[0]
    scores = {str(cls): float(p) for cls, p in zip(modelo_sklearn.classes_, proba)}
    return max(scores, key=scores.get), scores


@app.get("/health")
def health():
    return {"status": "ok", "runtime": "onnx" if USAR_ONNX else "sklearn"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictOut)
def predict(body: PredictIn):
    label, scores = classificar(body.text)
    return PredictOut(label=label, proba=scores)
