from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

LABELS = {"neoplasms", "digestive", "nervous", "cardiovascular", "general"}


def test_predict_classe_do_corpus():
    resp = client.post(
        "/predict",
        json={
            "text": (
                "Neuropeptide Y and neuron-specific enolase levels in benign "
                "and malignant pheochromocytomas. Serum NSE may help distinguish "
                "malignant from benign pheochromocytoma."
            )
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] in LABELS
    assert set(body["proba"]) == LABELS


def test_predict_texto_vazio():
    resp = client.post("/predict", json={"text": ""})
    assert resp.status_code == 422
