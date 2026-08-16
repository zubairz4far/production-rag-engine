from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_liveness_includes_request_id() -> None:
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.headers["x-request-id"]


def test_request_id_is_propagated() -> None:
    response = client.get("/health/live", headers={"x-request-id": "trace-123"})
    assert response.status_code == 200
    assert response.headers["x-request-id"] == "trace-123"


def test_metrics_endpoint_exposes_http_metrics() -> None:
    client.get("/health/live")
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "rag_http_requests_total" in response.text
    assert "rag_http_request_duration_seconds" in response.text
