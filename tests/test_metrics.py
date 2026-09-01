from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_metrics_formato_prometheus():
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert b"http_requests_total" in resp.content
    assert b"http_request_duration_seconds" in resp.content
