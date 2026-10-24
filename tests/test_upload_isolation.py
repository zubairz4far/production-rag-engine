"""Upload isolation and size enforcement: no shared filename writes."""

from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException, UploadFile

from app.api import routes
from app.services.document_loader import PageText


class RecordingService:
    def __init__(self) -> None:
        self.documents = []
        self.vector_store = self

    def index_chunks(self, document_id, source, chunks):
        self.documents.append((document_id, source, chunks))
        return len(chunks)


def configure(monkeypatch, service):
    monkeypatch.setattr(
        routes,
        "get_settings",
        lambda: SimpleNamespace(
            demo_mode=False, max_upload_mb=1, chunk_size_words=20, chunk_overlap_words=4
        ),
    )
    monkeypatch.setattr(routes, "RAGService", RecordingService)
    monkeypatch.setattr(routes, "get_rag_service", lambda: service)
    monkeypatch.setattr(routes, "chunk_pages", lambda pages, *_: pages)


def test_same_filename_does_not_overwrite_other_uploads(monkeypatch):
    service = RecordingService()
    configure(monkeypatch, service)
    temporary_paths: list[Path] = []

    def capture_document(path: Path):
        temporary_paths.append(path)
        return [PageText(page=None, text=path.read_text(encoding="utf-8"))]

    monkeypatch.setattr(routes, "load_document", capture_document)

    first = routes.upload_document(UploadFile(file=BytesIO(b"first"), filename="shared.txt"))
    second = routes.upload_document(UploadFile(file=BytesIO(b"second"), filename="shared.txt"))

    assert first.document_id == hashlib.sha256(b"first").hexdigest()[:24]
    assert second.document_id == hashlib.sha256(b"second").hexdigest()[:24]
    assert first.document_id != second.document_id
    assert [x.source for x in (first, second)] == ["shared.txt", "shared.txt"]
    assert len(service.documents) == 2
    assert len(set(temporary_paths)) == 2
    assert all(not path.exists() for path in temporary_paths)


def test_upload_size_limit_blocks_before_indexing_and_cleans_up(monkeypatch):
    service = RecordingService()
    configure(monkeypatch, service)
    monkeypatch.setattr(routes, "load_document", lambda _: pytest.fail("must not parse oversize"))

    with pytest.raises(HTTPException) as exc:
        routes.upload_document(
            UploadFile(file=BytesIO(b"z" * (1024 * 1024 + 1)), filename="huge.txt")
        )

    assert exc.value.status_code == 413
    assert service.documents == []


def test_parser_failure_does_not_leave_untrusted_file_on_disk(monkeypatch):
    service = RecordingService()
    configure(monkeypatch, service)
    seen = []

    def fail_parse(path: Path):
        seen.append(path)
        raise ValueError("invalid source")

    monkeypatch.setattr(routes, "load_document", fail_parse)
    with pytest.raises(HTTPException) as exc:
        routes.upload_document(UploadFile(file=BytesIO(b"not a pdf"), filename="bad.pdf"))
    assert exc.value.status_code == 422
    assert "invalid source" not in exc.value.detail

    assert len(seen) == 1
    assert not seen[0].exists()
    assert service.documents == []


@pytest.mark.parametrize("error", [ValueError("sensitive parser path"), RuntimeError("bad PDF")])
def test_malformed_documents_return_sanitized_422_and_are_not_indexed(monkeypatch, error):
    service = RecordingService()
    configure(monkeypatch, service)
    monkeypatch.setattr(routes, "load_document", lambda _: (_ for _ in ()).throw(error))
    with pytest.raises(HTTPException) as exc:
        routes.upload_document(UploadFile(file=BytesIO(b"not a valid document"), filename="bad.pdf"))
    assert exc.value.status_code == 422
    assert str(error) not in exc.value.detail
    assert service.documents == []


def test_empty_extraction_is_rejected_without_indexing(monkeypatch):
    service = RecordingService()
    configure(monkeypatch, service)
    monkeypatch.setattr(routes, "load_document", lambda _: [PageText(page=1, text="   ")])
    with pytest.raises(HTTPException) as exc:
        routes.upload_document(UploadFile(file=BytesIO(b"blank"), filename="blank.pdf"))
    assert exc.value.status_code == 422
    assert "no extractable text" in exc.value.detail
    assert service.documents == []
