# Full-Filing PDF Extraction and NSE Market Data Ingestion

FinGPT Trader includes an offline ingestion pipeline for Kenyan/NSE issuer filings and daily equity price lists so you can build real, point-in-time local datasets and corporate-action catalogs without a broker or commercial feed.

Install the optional PDF dependencies:

```sh
python3 -m pip install -e '.[filings]'
```

---

## 1. Extract Dividend Events from a Complete Issuer PDF

```sh
python3 -m fingpt_research filing-extract \
  --pdf scbk-fy24.pdf \
  --symbol SCBK \
  --source "https://av.sc.com/ke/content/docs/ke-press-release-standard-chartered-fy-2024-results.pdf" \
  > scbk-fy24-report.json
```

### How it works
1. **Dual-decoder coverage check (`filing_pdf.py`):** Every physical page is decoded with `pypdf` (layout mode) and cross-checked against `pypdfium2` geometry and text extraction. Pages with no text layer, unmapped glyphs, decoder disagreements, or large scanned raster blocks (>=35% of page area, or non-margin images on sparse pages) are marked `partial` or `unreadable`. Moderate charts/logos on text-rich press releases stay `readable`. A `partial` document returning zero events is never confused with a `complete` zero-event filing.
2. **Deterministic rules first (`filing_events.py`):** Candidate dividend sentences are discovered across columns, numbered AGM resolutions, polling verdict tables, and cross-page continuations. Every extracted field retains exact page numbers and verbatim PDF line citations. Par values (e.g., `"ordinary share of KShs 5.00"`), prior-year comparators, and annual totals are filtered out.
3. **Optional hosted Gemini fallback (`--vertex-project`):** When `--vertex-project YOUR_PROJECT` is passed, deterministic rules still run first. Hosted Gemini (`gemini-3.5-flash-lite` via `vertex.py`) is called **only** for candidates missing a core field (`distribution_type`, `period`, `status`, `amount`, or `currency`), and any model claim without a verbatim quote from the candidate text is rejected.

### Verified live example (Standard Chartered Bank Kenya FY2024)
Running `filing-extract` offline on Standard Chartered Kenya's four-page FY2024 results PDF returns:
- `coverage.status == "complete"` (all 4 pages `readable`)
- **Exactly 2 events** with zero uncertainties:
  - `SCBK-2024-final`: `amount = 37`, `currency = KES`, `status = proposed` (page 2, ignores KES 5.00 par value and KES 45.00 annual total)
  - `SCBK-2024-interim`: `amount = 8`, `currency = KES`, `status = paid` (page 2, `"was declared and paid"`)

---

## 2. Promote Extracted Events to a Local Catalog

Model or rule extractions from `filing-extract` are research reports — they never mutate portfolio accounting automatically. After reviewing a report, promote it into a local, append-only corporate-actions catalog:

```sh
python3 -m fingpt_research catalog-add \
  --report scbk-fy24-report.json \
  --notice-date 2025-03-19 \
  --catalog ./local-catalog

python3 -m fingpt_research actions \
  --as-of 2025-03-20 \
  --symbol SCBK \
  --catalog ./local-catalog
```

Promotion is idempotent by `(symbol, notice_date, event_id)`. Point-in-time queries apply the same next-Nairobi-midnight disclosure clock as the bundled nine-issuer catalog.

---

## 3. Import NSE Daily Equity Price List PDFs

Build or append to a point-in-time `nse-daily` dataset from one PDF or a directory of selectable-text NSE Daily Equity Price List PDFs:

```sh
python3 -m fingpt_research nse-import \
  --pdf ./nse-daily-lists \
  --symbol SCOM \
  --out ./datasets/scom

python3 -m fingpt_research replay --dataset ./datasets/scom
python3 -m fingpt_research paper --dataset ./datasets/scom --policy buy_hold
```

### Important notes on real NSE PDFs
- **VWAP valuation marks:** Standard NSE daily lists publish High, Low, VWAP, Previous, and Volume without an opening price. `nse-import` preserves the blank open and sets `price_type="vwap"`, matching `fingpt_research/fixtures/nse-public-sample`.
- **Scanned / image-only PDFs:** Some distributor-hosted NSE price lists (for example, I&M Bank's hosted `02-SEP-26.pdf`) are flattened raster images with no embedded text layer. `nse-import` detects image-only PDFs via `filing_pdf.py` and fails fast rather than silently producing empty datasets; run OCR first or use a selectable-text PDF.
- **Built-in ticker & ISIN mapping:** Recognizes major NSE tickers (`SCOM`, `EQTY`, `KCB`, `SCBK`, `EABL`, `BAT`, `KPLC`, `KEGN`, `IMH`, `ABSA`, `COOP`, `NCBA`) out of the box, and accepts `--isin` and `--issuer` overrides for any other NSE-listed security.
