from starlette.requests import Request

from app.core.rate_limit import client_identity


def make_request(headers: dict[str, str]) -> Request:
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/v1/query",
        "headers": [(key.lower().encode(), value.encode()) for key, value in headers.items()],
        "client": ("203.0.113.10", 43210),
        "server": ("testserver", 80),
        "scheme": "http",
        "query_string": b"",
    }
    return Request(scope)


def test_socket_peer_is_default_even_when_forwarded_header_is_spoofed() -> None:
    request = make_request({"x-forwarded-for": "198.51.100.99"})

    assert client_identity(request, None) == "203.0.113.10"


def test_explicit_trusted_header_is_used() -> None:
    request = make_request({"cf-connecting-ip": "198.51.100.42"})

    assert client_identity(request, "cf-connecting-ip") == "198.51.100.42"


def test_missing_trusted_header_falls_back_to_socket_peer() -> None:
    request = make_request({})

    assert client_identity(request, "cf-connecting-ip") == "203.0.113.10"
