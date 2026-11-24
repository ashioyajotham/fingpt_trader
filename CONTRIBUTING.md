# Contributing to AsOf

## Development Setup

Python 3.11 or later runs the core research CLI using the standard library. Optional extras add PDF parsing (`filings`), local GGUF inference (`model`), and test/lint tooling (`dev`).

```sh
git clone https://github.com/ashioyajotham/asof.git
cd asof
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e '.[dev,filings]'
```

## Architecture Principles

1. **Research and execution stay separated:** The CLI runs offline and paper-only. Model outputs and extracted filings never submit live orders or mutate portfolio accounting directly.
2. **Point-in-time discipline:** Every market observation, FX quote, borrow schedule, and issuer disclosure carries explicit timezone-aware availability semantics. Never introduce look-ahead bias.
3. **Deterministic baselines first:** Prefer stdlib-only deterministic rules and explicit provenance. Optional hosted Gemini (`vertex.py`) or local GGUF inference is routed only as a fallback when rules cannot resolve a core field, and every model claim must be backed by a verbatim source quote.
4. **Asset-agnostic core, explicit venue adapters:** Market replay, paper accounting, and risk calculations live in `asof/` without venue lock-in; NSE equities and Binance spot sit behind clean adapters.

## Project Layout

```text
asof/
├── cli.py              # CLI entry point (replay, actions, filing-extract, catalog-add, nse-import, paper, compare, portfolio)
├── data.py             # Immutable bars, Binance adapter, dataset loader
├── nse.py              # NSE daily listing loader & corporate action validation (dividends, splits, bonuses)
├── nse_pricelist.py    # NSE Daily Equity Price List PDF/CSV parser & OCR-ready dataset importer
├── filing_pdf.py       # Dual-decoder (pypdf + pypdfium2) PDF text & coverage checker
├── filing_events.py    # Full-filing dividend discovery, rules extraction & uncertainty-routed model fallback
├── filings.py          # Single-filing extraction orchestration
├── dividends.py        # Point-in-time dividend catalog queries & filing promotion
├── dividend_eval.py    # Frozen passage benchmark scoring
├── evidence.py         # Passage-grounded extraction & optional LocalFinGPT backend
├── vertex.py           # Optional hosted Vertex AI Gemini fallback backend
├── paper.py            # Single-listing Decimal paper ledger (dividends, splits/bonuses, short borrow)
├── borrow.py           # ACT/365 short borrow schedules & forced buy-ins
├── portfolio.py        # Multi-currency portfolio aggregation, periodic rebalancing & benchmark attribution
├── experiments.py      # Walk-forward baseline comparisons & timestamped model features
├── forward.py          # Append-only streaming paper book
├── risk.py             # Drawdown, exposure, Sharpe/Sortino, concentration, attribution & liquidity metrics
├── web.py              # Local read-only stdlib web workstation (asof serve)
└── fixtures/           # Bundled offline test fixtures (binance, nse, nse-public-sample, dividend-notices)
tests/                  # Fast offline pytest suite
docs/                   # Design notes, validation logs & ingestion guides
tools/                  # Standalone helper scripts (blind review reconciliation, public Binance check)
```

## Testing & Verification

All pull requests must keep the offline test suite green with zero network calls:

```sh
python3 -m pytest -q tests
python3 -I -m asof replay --dataset nse --as-of 2025-01-08T12:00:00Z
```

## Branching & Pull Requests

- Branch from `main` using `feat/<topic>` or `fix/<topic>`.
- Include unit or regression tests for every behavior change.
- Keep documentation in `README.md` and `docs/` synchronized with CLI flags and schemas.
- Do not commit raw multi-megabyte model dumps or copyrighted full issuer PDFs to git.

## License

By contributing, you agree that your contributions will be licensed under the MIT License. See [LICENSE.md](LICENSE.md).
