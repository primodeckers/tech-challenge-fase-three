import time
from pathlib import Path

import joblib
from fastapi import FastAPI, Request
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from starlette.responses import Response

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "model.joblib"

app = FastAPI(title="Triagem de laudos")
model = joblib.load(MODEL_PATH)

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


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictOut)
def predict(body: PredictIn):
    proba = model.predict_proba([body.text])[0]
    scores = {str(cls): float(p) for cls, p in zip(model.classes_, proba)}
    label = max(scores, key=scores.get)
    return PredictOut(label=label, proba=scores)
