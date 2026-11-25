"""Run an offline extractor against one immutable filing and expose its evidence."""
import hashlib
import json
from time import perf_counter

from .filing_events import discover, extract_events
from .filing_pdf import read_pdf
from .filing_statements import extract_financials


def prepare_filing(path, symbol, source):
    if not symbol.strip() or not source.strip():
        raise ValueError("supply a listing symbol and source reference")
    start = perf_counter()
    pages, metadata = read_pdf(path)
    candidates = discover(pages)
    financials = extract_financials(pages)
    payload = json.dumps(candidates, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {**metadata, "symbol": symbol.strip().upper(), "source": source,
            "candidates": candidates, "financials": financials,
            "candidate_sha256": hashlib.sha256(payload.encode()).hexdigest(),
            "discovery_seconds": perf_counter() - start}


def _render_pdf_page_png(pdf_path, page_num):
    """Render one 1-based PDF page to PNG bytes in memory via pypdfium2."""
    import io
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        page = doc[page_num - 1]
        bitmap = page.render(scale=2.0)
        pil_img = bitmap.to_pil()
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return buf.getvalue()
    finally:
        doc.close()


def extract_filing(prepared, model=None, *, fallback_only=True, pdf_path=None):
    start = perf_counter()
    before_calls = dict(model.diagnostics) if model else {}
    before_tokens = model.tokens if model else 0
    events = extract_events(prepared["candidates"], prepared["symbol"], model=model, fallback_only=fallback_only)
    financials = dict(prepared.get("financials") or {})
    if (
        model is not None
        and pdf_path is not None
        and hasattr(model, "extract_financials_from_image")
        and prepared.get("coverage", {}).get("status") != "complete"
    ):
        flagged = [
            p["page"] for p in prepared.get("coverage", {}).get("pages", [])
            if p.get("status") in {"partial", "unreadable"}
        ]
        for page_num in flagged:
            if model.diagnostics["calls"] >= model.max_calls:
                break
            if {"revenue", "profit_after_tax", "eps"}.issubset(financials):
                break
            png_bytes = _render_pdf_page_png(pdf_path, page_num)
            recovered = model.extract_financials_from_image(png_bytes, prepared["symbol"], page_num)
            for key, item in recovered.items():
                if key not in financials:
                    financials[key] = item
    method = {"name": "rules"}
    if model:
        method = {
            "name": f"rules+{model.name}" if fallback_only else model.name,
            "provider": model.provider,
            "model": model.version,
            "fallback_only": fallback_only,
            "tokens": model.tokens - before_tokens,
            "diagnostics": {key: count - before_calls.get(key, 0)
                            for key, count in model.diagnostics.items()},
        }
    return {**prepared, "financials": financials, "events": events, "method": method,
            "extraction_seconds": perf_counter() - start}


def extract_filing_financials(path, symbol, source):
    """Extract cited income-statement, EPS, and balance-sheet metrics from a filing PDF."""
    if not symbol.strip() or not source.strip():
        raise ValueError("supply a listing symbol and source reference")
    pages, metadata = read_pdf(path)
    return {
        **metadata,
        "symbol": symbol.strip().upper(),
        "source": source,
        "financials": extract_financials(pages),
    }
