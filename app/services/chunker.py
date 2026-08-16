from dataclasses import dataclass
from app.services.document_loader import PageText


@dataclass(slots=True)
class Chunk:
    text: str
    page: int | None
    chunk_index: int


def chunk_pages(pages: list[PageText], chunk_size_words: int, overlap_words: int) -> list[Chunk]:
    if chunk_size_words <= 0:
        raise ValueError("chunk_size_words must be positive")
    if overlap_words < 0 or overlap_words >= chunk_size_words:
        raise ValueError("overlap_words must be >= 0 and smaller than chunk_size_words")
    step = chunk_size_words - overlap_words
    chunks: list[Chunk] = []
    idx = 0
    for page in pages:
        words = page.text.split()
        for start in range(0, len(words), step):
            window = words[start:start + chunk_size_words]
            if not window:
                continue
            chunks.append(Chunk(text=" ".join(window), page=page.page, chunk_index=idx))
            idx += 1
            if start + chunk_size_words >= len(words):
                break
    return chunks
