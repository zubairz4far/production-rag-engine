from dataclasses import dataclass
from pathlib import Path
import pymupdf


@dataclass(slots=True)
class PageText:
    page: int | None
    text: str


SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".markdown"}


def load_document(path: Path) -> list[PageText]:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported file type: {suffix}")
    if suffix == ".pdf":
        pages = []
        with pymupdf.open(path) as doc:
            for page_index, page in enumerate(doc):
                text = page.get_text("text").strip()
                if text:
                    pages.append(PageText(page=page_index + 1, text=text))
        return pages
    return [PageText(page=None, text=path.read_text(encoding="utf-8", errors="replace"))]
