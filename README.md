# FinGPT Trader

FinGPT Trader replays market data and issuer documents as they became available, preserves source citations, and simulates paper portfolios. Individual books use their listing currency. A portfolio study can combine separately funded books in one base currency. Python 3.11 or later runs the research commands with the standard library.

## Try the replay

```sh
python3 main.py
python3 main.py replay --dataset nse --as-of 2025-01-08T15:00:00+03:00
python3 main.py evidence --dataset nse --as-of 2025-01-09T15:00:00+03:00
python3 main.py paper --dataset nse --policy buy_hold
python3 main.py compare --dataset nse
python3 main.py evaluate --dataset nse
```

The default replay contains 12 captured Binance BTC/USDT daily bars. The NSE example contains 13 fictional KES sessions, an issuer statement, two versions of one dividend notice, a court filing, a decision and a news report. Try the two January cutoffs above. At the January 8 close the original dividend is KES 1.00. The correction to KES 1.20 arrives after that close. The example calendar, fees, prices, liquidity and labels are teaching inputs. The bundled cases exercise the software and provide no estimate of investment performance.

## Import local data

Copy a [sample dataset](fingpt_research/fixtures/nse) and pass its directory to `--dataset`. Keep privately licensed source files locally. One manifest identifies the stable listing, issuer, currency, venue, lot size, adapter, source, capture time and calendar version. Use raw prices with separately recorded dividends. A single-listing report uses its listing currency. The portfolio command translates the funded books using dated FX observations.

| File | Contents |
| --- | --- |
| `sessions.csv` | Versioned session dates and explicit timezone-aware opens and closes. Supply the applicable exchange calendar, including holidays and exceptional sessions. |
| `prices.csv` | One row per session with raw OHLCV, availability time and trading status. Suspended sessions carry an explicit valuation mark. |
| `documents.json` | Immutable document IDs, source references, issuer, attributed document kind, publication and ingestion times, and an array of page texts. Optional fields include the financial period, predecessor document ID and curated dividend terms. |
| `labels.json` | Optional field references for evaluation, including expected text, page and supporting span. Maintain these independently for a real study. |
| `klines.json` | For the Binance spot adapter, use a frozen UTC daily REST kline response with millisecond timestamps and its capture time in the manifest. |

Replay reconstructs public availability. Ingestion time separately records when the snapshot was collected. A date-only publication becomes eligible at the next listed session open. For bulk-imported history, the output describes what public evidence supports, rather than the arrival history of an actual trading feed. Extracted pages must preserve source page order. The rules extractor abstains on ambiguous matches. A filing preserves the claimant's attribution and a judgment preserves the court's attribution.

## Paper portfolios and comparisons

Paper is the only execution mode of the research CLI. The default paper policy holds cash. A producer emits a timestamped research signal with its version, horizon and evidence. Portfolio sizing converts it to a long-only target and approves a quantity. A pending quantity can fill on a later tradable raw close, capped by participation, available cash, lot size and exposure. Fee and slippage assumptions apply to each fill. The closing-price approximation is suited to daily-bar research. A dividend becomes a receivable using the holding before its ex-date and becomes cash on the first session on or after payment. Only curated corporate action terms can post to the ledger.

```sh
python3 main.py paper --dataset nse --policy price --fee-bps 15 --slippage-bps 8 --participation 0.005
python3 main.py paper --dataset /data/initial-snapshot --policy price --follow
```

In follow mode, a local publisher writes the directory of each completed snapshot to standard input, one per line. Standard output emits a JSON portfolio report for each extension. Repeated snapshots are idempotent. Publish each snapshot atomically, append corrections as new documents, and retain the snapshot sequence for recovery. A changed historical bar or backdated source requires a fresh historical replay.

The comparison reads the fixed test start, one-session horizon and fold length from the manifest. Earlier prices warm up the factor. The book carries through consecutive test folds. Cash, buy-and-hold including dividends, and one-session price momentum use the same costs and liquidity rules. A second scenario doubles fee and slippage and halves participation. Reports include drawdown, turnover, exposure, fees and net returns. The historical comparison uses long-only policies. For a borrow or cross-currency study, use the portfolio command below.

### Cross-currency and borrow studies

```sh
python3 main.py replay --dataset nse-public-sample
python3 main.py portfolio --spec examples/kes-portfolio.json
```

The public sample is a single Safaricom observation checked against the NSE price list dated September 2, 2026. Its opening price is unknown. Its VWAP can value a holding but cannot execute a paper order. The captured document gives the dividend amount and book closure. Posting a payment requires separately verified entitlement terms.

The [example study](examples/kes-portfolio.json) pre-funds two books in KES and USDT. FX is quoted as base currency per unit of listing currency. Each translation uses a rate available at the valuation time. Missing rates, stale rates and stale held-position marks stop the study. Weights must total one. Books have separate cash and position limits. This first version assumes frictionless initial FX conversion and excludes collateral transfers, settlement lags and taxes.

A short sale requires explicit dated loan inventory. A schedule sets maximum outstanding units, an annual rate, expiry and a buy-in premium. Loan fees accrue ACT/365 on the preceding mark, including weekends. A reduction in the allowed outstanding quantity or an expiry triggers a buy-in subject to tradable volume. The last known loan rate continues on overdue units. Short holders owe any eligible dividend. For a single book pass `paper --policy short_hold --borrow loan.json`, with `instrument` and a `records` array using the example's borrow fields. The included FX and borrowing terms are synthetic scenario assumptions.

## FinGPT experiments

Supply a locally merged FinGPT GGUF for model experiments. Install `.[model]` in a virtual environment. The model file digest identifies the weights. The evaluator compares FinGPT and the deterministic parser against the same reference fields with field accuracy, citation precision, abstentions, elapsed time and model tokens.

```sh
python3 main.py evaluate --dataset nse --gguf /models/fingpt.gguf --structured
python3 main.py evaluate --corpus fingpt_research/fixtures/issuer-excerpts --gguf /models/fingpt.gguf --structured
python3 main.py features --dataset nse --gguf /models/fingpt.gguf --structured --model-available-at 2024-12-01T00:00:00Z --latency-seconds 60 > features.json
python3 main.py compare --dataset nse --features features.json
```

Use the actual model availability date and a defensible inference delay in an experiment. The example arguments illustrate the format. The additional policy applies the same price factor with a sentiment filter on its first eligible observation. Treat generated sentiment as a hypothesis. Assess its incremental value on a separately frozen, reviewed sample before drawing conclusions.

[Local model evaluation and provider design](docs/research-validation.md)

## Development

```sh
python3 -m pip install -e '.[dev]'
python3 -m pytest -q tests
```

New read-only adapters can be supplied to `load_dataset(..., adapters={...})`. Custom signal producers can be passed to `PaperEngine` and receive the listing, current bar, previous tradable bar and available features. The original optimizer, strategy modules, Binance SDK integration and optional FinGPT code remain in the source tree. The legacy orchestrator is importable through `main.TradingSystem`; research signals cannot submit exchange orders through it.

The legacy crypto news service reads publisher RSS feeds configured in `config/services.yaml`, with an optional NewsAPI fallback.

## License

MIT License - See [LICENSE](LICENSE.md) for details.

## Acknowledgements

This project builds on the foundation of several important open-source projects and research:

- **FinGPT Research**: Core ideas on financial sentiment analysis and market prediction using large language models were inspired by research from the [FinGPT paper](https://arxiv.org/abs/2306.06031).

- **llama.cpp**: Our inferencing pipeline leverages the [llama.cpp](https://github.com/ggml-org/llama.cpp) framework for running language models with minimal computational resources.

- **Binance API**: Trading functionality is built on the [Binance exchange API](https://binance-docs.github.io/apidocs/spot/en/#public-rest-api) and Python client libraries.

- **Python AsyncIO**: The asynchronous architecture is powered by [Python's AsyncIO framework](https://docs.python.org/3/library/asyncio.html).

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes. Ensure to follow the code style and include tests for new features. See [CONTRIBUTING.md](CONTRIBUTING.md) for more details.
