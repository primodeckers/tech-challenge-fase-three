"""Compara latencia sklearn vs ONNX no mesmo conjunto de textos."""

from __future__ import annotations

import statistics
import time
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "laudos.csv"
MODEL_PATH = ROOT / "artifacts" / "model.joblib"
ONNX_PATH = ROOT / "artifacts" / "model.onnx"
N = 200
WARMUP = 10
SEED = 42


def _p95(valores: list[float]) -> float:
    valores = sorted(valores)
    return valores[int(0.95 * (len(valores) - 1))]


def _medir(fn, textos: list[str]) -> list[float]:
    for t in textos[:WARMUP]:
        fn(t)
    tempos = []
    for t in textos:
        t0 = time.perf_counter()
        fn(t)
        tempos.append((time.perf_counter() - t0) * 1000)
    return tempos


def main() -> None:
    textos = (
        pd.read_csv(DATA_PATH)
        .sample(N, random_state=SEED)["texto"]
        .astype(str)
        .tolist()
    )
    pipe = joblib.load(MODEL_PATH)
    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name

    def sklearn_fn(texto: str):
        pipe.predict_proba([texto])

    def onnx_fn(texto: str):
        sess.run(None, {inp: np.array([[texto]])})

    t_sk = _medir(sklearn_fn, textos)
    t_on = _medir(onnx_fn, textos)

    print(f"n={N} warmup={WARMUP} (inferencia local, sem HTTP)")
    print(
        "sklearn "
        f"media={statistics.mean(t_sk):.3f} "
        f"p50={statistics.median(t_sk):.3f} "
        f"p95={_p95(t_sk):.3f}"
    )
    print(
        "onnx    "
        f"media={statistics.mean(t_on):.3f} "
        f"p50={statistics.median(t_on):.3f} "
        f"p95={_p95(t_on):.3f}"
    )


if __name__ == "__main__":
    main()
