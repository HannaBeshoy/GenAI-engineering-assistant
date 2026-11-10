"""PDF to page texts, with the checks that turn bad uploads into clear messages."""
from __future__ import annotations

import io
import re
from dataclasses import dataclass

from .errors import PdfError

MAX_PAGES = 120
MAX_BYTES = 25 * 1024 * 1024


@dataclass
class Page:
    number: int          # 1-based, as printed in the PDF viewer
    text: str


def read_pdf(data: bytes) -> list:
    if not data:
        raise PdfError("The file is empty.")
    if len(data) > MAX_BYTES:
        raise PdfError(f"The file is larger than {MAX_BYTES // 2**20} MB.")
    if not data.lstrip()[:5].startswith(b"%PDF"):
        raise PdfError("This is not a PDF file.")
    try:
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError
    except ImportError:  # pragma: no cover
        raise PdfError("pypdf is not installed: run pip install -r requirements.txt") from None
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise PdfError("The PDF is password-protected.")
        if len(reader.pages) > MAX_PAGES:
            raise PdfError(f"The PDF has {len(reader.pages)} pages; the limit is {MAX_PAGES}. Upload only the needed chapter.")
        pages = [Page(i + 1, _clean(_text(p))) for i, p in enumerate(reader.pages)]
    except PdfError:
        raise
    except (PdfReadError, ValueError, KeyError, OSError) as e:
        raise PdfError(f"The PDF could not be read ({type(e).__name__}).") from e
    if sum(len(p.text) for p in pages) < 200:
        raise PdfError("The PDF has (almost) no text layer; it is probably a scanned image.")
    return pages


def _text(page) -> str:
    """Layout mode keeps table rows on one line, which models read far better than one cell per line."""
    try:
        return page.extract_text(extraction_mode="layout") or ""
    except Exception:                       # noqa: BLE001 - fall back to the simple extractor
        return page.extract_text() or ""


def _clean(text: str) -> str:
    lines = []
    for line in text.splitlines():
        line = re.sub(r" {3,}", " | ", line.strip())     # wide gaps are table column breaks
        line = re.sub(r" {2}", " ", line)
        if line:
            lines.append(line)
    return "\n".join(lines)
