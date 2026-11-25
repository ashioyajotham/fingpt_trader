# Working with Real NSE Kenya Market Data & Issuer Filings

`AsOf` intentionally ships only compact fixtures in git so tests stay fast and third-party licensed market data or copyrighted PDFs are never redistributed. This guide shows how to load real multi-year Nairobi Securities Exchange (NSE) prices and official issuer filings into your own local workspace.

---

## 1. Bulk-Import Multi-Ticker NSE Daily Prices (`--universe`)

`asof nse-import --universe` natively reads multi-ticker NSE daily price CSVs, including:
- **Official NSE Data Services exports** (`https://www.nse.co.ke/dataservices/`)
- **Academic open-access NSE All-Stocks datasets** (e.g., Mendeley Data `NSE_data_all_stocks_*.csv`, CC BY 4.0) containing columns `Date, Code, Name, 12m Low, 12m High, Day Low, Day High, Day Price, Previous, Change, Change%, Volume, Adjusted Price`

`AsOf` always uses raw `Day Price` / `Close` / `VWAP` and ignores pre-adjusted columns (`Adjusted Price`) so corporate actions remain explicit and auditable.

```sh
# Import an entire multi-year directory of NSE All-Stocks CSVs:
python3 -m asof nse-import \
  --input ./local-data/nse-csvs \
  --universe ./local-data/universe \
  --symbols SCOM,EQTY,KCB,SCBK,EABL,BAT,COOP,ABSA,NCBA,KEGN,KPLC,IMH

# Replay any imported listing immediately:
python3 -m asof replay --dataset ./local-data/universe/SCOM
```

## 1b. Keep Your Universe Current Daily (`asof nse-sync`)

After importing historical CSVs (or to bootstrap a fresh universe from scratch), append newly closed NSE daily sessions after 15:00 EAT using the built-in append-only End-of-Day adapter:

```sh
# Public EOD sync (default: StockAnalysis NASE raw OHLCV, zero API key, ignores adjusted close):
python3 -m asof nse-sync \
  --universe ./local-data/universe \
  --provider stockanalysis \
  --range 6M

# Commercial EODHD sync (uses EODHD_API_KEY from environment or --api-key):
python3 -m asof nse-sync \
  --universe ./local-data/universe \
  --provider eodhd \
  --since 2026-07-01

# Remote daily CSV URL sync (e.g. NSE Data Services or broker export URL):
python3 -m asof nse-sync \
  --universe ./local-data/universe \
  --provider csv-url \
  --url "https://your-source.example/nse-daily.csv"
```

`asof nse-sync` is strictly **append-only** by default (never silently overwriting stored sessions), automatically warns if any overlapping close differs by >1% across providers, and refuses to ingest partial/unclosed intraday sessions before 15:00 EAT.

---

## 2. Extract & Persist Point-in-Time Dividends and Financials from Real Issuer PDFs

Download official results press releases or AGM notices directly from issuer investor-relations portals (Safaricom, Equity Group, KCB, Standard Chartered Kenya, EABL, BAT Kenya, KPLC, KenGen, I&M Group, Absa, Co-op Bank, NCBA) and process them offline:

```sh
# 1. Extract both dividend events and scale-aware financial statements:
python3 -m asof filing-extract \
  --pdf ./local-data/pdfs/scbk-fy24.pdf \
  --symbol SCBK \
  --source "https://av.sc.com/ke/content/docs/ke-press-release-standard-chartered-fy-2024-results.pdf" \
  > /tmp/scbk-fy24.json

# 2. Promote reviewed dividends AND financials into a shared local catalog:
python3 -m asof catalog-add \
  --report /tmp/scbk-fy24.json \
  --notice-date 2025-03-19 \
  --catalog ./local-data/catalog

python3 -m asof financials-add \
  --report /tmp/scbk-fy24.json \
  --notice-date 2025-03-19 \
  --catalog ./local-data/catalog

# 3. Query strictly point-in-time (--as-of) under the Nairobi midnight disclosure clock:
python3 -m asof actions --as-of 2025-03-20 --symbol SCBK --catalog ./local-data/catalog
python3 -m asof financials --as-of 2025-03-20 --symbol SCBK --catalog ./local-data/catalog

# 4. Evaluate cross-sectional factors and run a zero-lookahead factor portfolio backtest:
python3 -m asof factors \
  --universe ./local-data/universe \
  --catalog ./local-data/catalog \
  --min-adv 500000

python3 -m asof factor-backtest \
  --universe ./local-data/universe \
  --catalog ./local-data/catalog \
  --factor-blend "dividend_yield:0.5,momentum_20d:-0.5" \
  --top-n 5 --max-sector-weight 0.40 --rebalance monthly \
  --min-adv 500000 --fee-bps 180 --slippage-bps 15 --wht-pct 5

# 5. Launch the local workstation pointed at your real NSE universe and catalog:
python3 -m asof serve \
  --universe ./local-data/universe \
  --catalog ./local-data/catalog
```

### Real-World Kenyan PDF Notes
- **Dual-decoder coverage:** Always inspect `coverage.status`. Some distributor-hosted press ads or price lists are flattened raster scans (`partial`/`unreadable`); pass `--ocr-cmd` on `nse-import` or inspect flagged pages manually.
- **Scale headers & bank formats:** `filing_statements.py` automatically detects stacked (`KShs\n million`), apostrophe (`Kshs 'M`, `KShs. '000`), and inline scale headers across two-column press ads, handles CBK bank rows (`Profit/ (loss) before tax and exceptional items`, `EARNINGS PER SHARE - BASIC & DILUTED`), and keeps `eps` unscaled (`KES/share`).
