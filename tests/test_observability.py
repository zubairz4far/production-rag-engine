from fastapi.testclient import TestClient

from app.api import routes
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


def test_security_headers_are_present() -> None:
    response = client.get("/health/live")

    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]


def test_unhandled_exception_is_sanitized(monkeypatch) -> None:
    class BrokenService:
        def query(self, question: str, top_k: int):
            raise RuntimeError("private backend detail")

    monkeypatch.setattr(routes, "get_rag_service", lambda: BrokenService())
    failing_client = TestClient(app, raise_server_exceptions=False)
    response = failing_client.post(
        "/v1/query",
        headers={"x-request-id": "failure-trace"},
        json={"question": "trigger a controlled failure", "top_k": 3},
    )

    assert response.status_code == 500
    assert response.json() == {
        "detail": "Internal server error.",
        "request_id": "failure-trace",
    }
    assert response.headers["x-request-id"] == "failure-trace"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "private backend detail" not in response.text
