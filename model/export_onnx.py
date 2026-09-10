"""Exporta o pipeline sklearn (TF-IDF + logreg) para ONNX."""

from __future__ import annotations

from pathlib import Path

import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "model.joblib"
ONNX_PATH = ROOT / "artifacts" / "model.onnx"
CLASSES_PATH = ROOT / "artifacts" / "classes.joblib"


def exportar() -> None:
    pipe = joblib.load(MODEL_PATH)
    onnx_model = convert_sklearn(
        pipe,
        initial_types=[("texto", StringTensorType([None, 1]))],
        options={id(pipe.named_steps["clf"]): {"zipmap": False}},
        target_opset=15,
    )
    ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)
    ONNX_PATH.write_bytes(onnx_model.SerializeToString())
    joblib.dump(list(pipe.classes_), CLASSES_PATH)
    print(f"onnx: {ONNX_PATH} ({ONNX_PATH.stat().st_size} bytes)")
    print(f"classes: {CLASSES_PATH} {list(pipe.classes_)}")


if __name__ == "__main__":
    exportar()
