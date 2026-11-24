"""Run an offline extractor against one immutable filing and expose its evidence."""
import hashlib
import json
from time import perf_counter

from .filing_events import discover, extract_events
from .filing_pdf import read_pdf


def prepare_filing(path, symbol, source):
    if not symbol.strip() or not source.strip():
        raise ValueError("supply a listing symbol and source reference")
    start = perf_counter()
    pages, metadata = read_pdf(path)
    candidates = discover(pages)
    payload = json.dumps(candidates, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {**metadata, "symbol": symbol.strip().upper(), "source": source,
            "candidates": candidates, "candidate_sha256": hashlib.sha256(payload.encode()).hexdigest(),
            "discovery_seconds": perf_counter() - start}


def extract_filing(prepared, model=None):
    start = perf_counter()
    before_calls = dict(model.diagnostics) if model else {}
    before_tokens = model.tokens if model else 0
    events = extract_events(prepared["candidates"], prepared["symbol"], model=model)
    method = {"name": "rules"}
    if model:
        method = {"name": model.name, "provider": model.provider, "weights_sha256": model.version,
                  "structured": model.structured, "tokens": model.tokens - before_tokens,
                  "diagnostics": {key: count - before_calls.get(key, 0)
                                  for key, count in model.diagnostics.items()}}
    return {**prepared, "events": events, "method": method, "extraction_seconds": perf_counter() - start}
