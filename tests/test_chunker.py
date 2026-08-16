import pytest

from app.services.chunker import chunk_pages
from app.services.document_loader import PageText


def test_chunking_preserves_overlap():
    page = PageText(page=1, text=" ".join(str(i) for i in range(25)))
    chunks = chunk_pages([page], chunk_size_words=10, overlap_words=2)

    assert len(chunks) == 3
    assert chunks[0].text.split()[-2:] == chunks[1].text.split()[:2]
    assert chunks[1].text.split()[-2:] == chunks[2].text.split()[:2]


def test_invalid_overlap_is_rejected():
    with pytest.raises(ValueError):
        chunk_pages([PageText(page=1, text="hello world")], 10, 10)
