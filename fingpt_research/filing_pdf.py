"""Offline extraction of page-numbered layout text from a single immutable PDF."""
import hashlib
import io
import unicodedata
from collections import Counter
from contextlib import closing
from importlib.metadata import version
from pathlib import Path


def _characters(text):
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return Counter(c for c in normalized if c.isalnum())


def _image_needs_review(page, image, char_count):
    # Ignore small margin logos/decorations. On text-rich pages, ignore moderate
    # decorative charts/headshots; flag large raster blocks (>=35% page area) or
    # non-margin images on sparse pages (typical of scanned tables/notices).
    if image.level:
        return True
    left, bottom, right, top = page.get_bbox()
    width, height = right - left, top - bottom
    page_area = abs(width * height)
    if page_area <= 0:
        return True
    x0, y0, x1, y1 = image.get_bounds()
    area_ratio = abs((x1 - x0) * (y1 - y0)) / page_area
    in_margin = (y0 >= top - height * 0.12 or y1 <= bottom + height * 0.12
                 or x0 >= right - width * 0.12 or x1 <= left + width * 0.12)
    if in_margin and area_ratio <= 0.02:
        return False
    if area_ratio >= 0.35:
        return True
    return char_count < 120 and area_ratio > 0.02


def _coverage_reason(text, geometry):
    from pypdfium2.raw import FPDF_PAGEOBJ_IMAGE

    chars = _characters(text)
    char_count = sum(chars.values())
    if not char_count:
        return "No readable selectable text"
    if any(unicodedata.category(c) in {"Co", "Cs"} or c == "\ufffd" for c in text):
        return "Unmapped glyphs in selectable text"
    if any(_image_needs_review(geometry, obj, char_count)
           for obj in geometry.get_objects(filter=[FPDF_PAGEOBJ_IMAGE])):
        return "Raster content requires visual review"
    with closing(geometry.get_textpage()) as independent:
        if chars != _characters(independent.get_text_bounded()):
            return "PDF decoders disagree on the page text"
    return None


def read_pdf(path):
    """Return layout texts and provenance. All physical pages retain their indexes."""
    try:
        import pypdfium2
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError
    except ImportError as exc:
        raise ImportError("install fingpt_trader[filings] to read source PDFs") from exc

    data = Path(path).read_bytes()
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise ValueError("the source PDF requires a password")
        native = pypdfium2.PdfDocument(data)
    except (PdfReadError, pypdfium2.PdfiumError) as exc:
        raise ValueError(f"could not open source PDF: {exc}") from exc
    texts, pages = [], []
    with closing(native):
        if not len(native) or len(reader.pages) != len(native):
            raise ValueError("could not establish the physical page count")
        for i in range(len(native)):
            text = ""
            try:
                text = reader.pages[i].extract_text(
                    extraction_mode="layout", layout_mode_space_vertically=False,
                    layout_mode_strip_rotated=False,
                )
                with closing(native[i]) as geometry:
                    reason = _coverage_reason(text, geometry)
            except (PdfReadError, pypdfium2.PdfiumError, ValueError, TypeError,
                    IndexError, KeyError, ArithmeticError, NotImplementedError) as exc:
                reason = f"Page decoding failed ({type(exc).__name__})"
            info = {"page": i + 1, "status": "readable"}
            if reason:
                info.update(status="partial" if _characters(text) else "unreadable", reason=reason)
            texts.append(text)
            pages.append(info)

    complete = all(p["status"] == "readable" for p in pages)
    return texts, {
        "pdf_sha256": hashlib.sha256(data).hexdigest(),
        "parser": {name: version(name) for name in ("pypdf", "pypdfium2")},
        "coverage": {"status": "complete" if complete else "partial",
                     "page_count": len(pages), "pages": pages},
    }
