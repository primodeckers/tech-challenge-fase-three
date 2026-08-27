from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

LABELS = {"normal", "atencao", "urgente"}


def test_predict_urgente():
    resp = client.post(
        "/predict",
        json={
            "text": (
                "Paciente com dor toracica intensa e sudorese. "
                "ECG com supradesnivel. Suspeita de sindrome coronariana aguda."
            )
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] == "urgente"
    assert set(body["proba"]) == LABELS


def test_predict_texto_vazio():
    resp = client.post("/predict", json={"text": ""})
    assert resp.status_code == 422
