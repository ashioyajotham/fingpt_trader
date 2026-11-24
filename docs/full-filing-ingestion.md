# Full-Filing PDF Extraction and NSE Market Data Ingestion

**AsOf** includes an offline ingestion pipeline for Kenyan/NSE issuer filings and daily equity price lists so you can build real, point-in-time local datasets and corporate-action catalogs without a broker or commercial feed.

Install the optional PDF dependencies:

```sh
python3 -m pip install -e '.[filings]'
```

---

## 1. Extract Dividend Events from a Complete Issuer PDF

```sh
python3 -m asof filing-extract \
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
Running `filing-extract` (or `financials-extract`) offline on Standard Chartered Kenya's four-page FY2024 results PDF returns:
- `coverage.status == "complete"` (all 4 pages `readable`)
- **Dividends (Page 2):** Exactly 2 events with zero uncertainties:
  - `SCBK-2024-final`: `amount = 37`, `currency = KES`, `status = proposed` (ignores KES 5.00 par value and KES 45.00 annual total)
  - `SCBK-2024-interim`: `amount = 8`, `currency = KES`, `status = paid` (`"was declared and paid"`)
- **Financials (Page 3, stacked `KShs million` header with unscaled EPS):**
  - `revenue` (Total operating income): `KES 50,677,000,000` (prior `2023`: `41,728,000,000`)
  - `profit_before_tax`: `KES 28,208,000,000` (`2023`: `19,668,000,000`)
  - `profit_after_tax`: `KES 20,060,000,000` (`2023`: `13,835,000,000`)
  - `eps`: `52.7 KES/share` (`2023`: `36.2 KES/share`, kept unscaled)
  - `loans_and_advances`: `KES 151,647,000,000`, `customer_deposits`: `KES 295,690,000,000`

---

## 2. Promote Extracted Events to a Local Catalog

Model or rule extractions from `filing-extract` are research reports — they never mutate portfolio accounting automatically. After reviewing a report, promote it into a local, append-only corporate-actions catalog:

```sh
python3 -m asof catalog-add \
  --report scbk-fy24-report.json \
  --notice-date 2025-03-19 \
  --catalog ./local-catalog

python3 -m asof actions \
  --as-of 2025-03-20 \
  --symbol SCBK \
  --catalog ./local-catalog
```

Promotion is idempotent by `(symbol, notice_date, event_id)`. Point-in-time queries apply the same next-Nairobi-midnight disclosure clock as the bundled nine-issuer catalog.

---

## 3. Import NSE Daily Equity Price Lists (PDF, CSV, or OCR)

Build or append to a point-in-time `nse-daily` dataset from one file or a directory of selectable-text NSE Daily Equity Price List PDFs or CSV exports:

```sh
python3 -m asof nse-import \
  --input ./nse-daily-lists \
  --symbol SCOM \
  --out ./datasets/scom

# Optional OCR fallback for scanned/image-only distributor PDFs:
python3 -m asof nse-import \
  --input ./02-SEP-26.pdf \
  --symbol SCOM \
  --out ./datasets/scom \
  --ocr-cmd "ocrmypdf --force-ocr {in} {out}"

python3 -m asof replay --dataset ./datasets/scom
python3 -m asof paper --dataset ./datasets/scom --policy buy_hold
```

### Important notes on real NSE price lists
- **CSV & PDF support:** Accepts both multi-ticker NSE daily price list CSVs (`Date,Symbol,ISIN,High,Low,VWAP,Prev,Volume`) and single-ticker OHLC CSVs (`Date,Open,High,Low,Close,Volume`), preserving `price_type="close"` when an opening price is present and `price_type="vwap"` when absent (matching `asof/fixtures/nse-public-sample`).
- **Scanned / image-only PDFs:** Some distributor-hosted NSE price lists (for example, I&M Bank's hosted `02-SEP-26.pdf`) are flattened raster images with no embedded text layer. `nse-import` detects image-only PDFs via `filing_pdf.py` and fails fast unless `--ocr-cmd` is supplied.
- **Built-in ticker & ISIN mapping:** Recognizes major NSE tickers (`SCOM`, `EQTY`, `KCB`, `SCBK`, `EABL`, `BAT`, `KPLC`, `KEGN`, `IMH`, `ABSA`, `COOP`, `NCBA`) out of the box, and accepts `--isin` and `--issuer` overrides for any other NSE-listed security.
