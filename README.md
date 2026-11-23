# FinGPT Trader

FinGPT Trader replays market data and issuer documents as they became available, preserves source citations, and simulates long-only paper portfolios. Each run covers one listing in its quote currency. Python 3.11 or later runs the research commands with the standard library.

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

Copy a [sample dataset](fingpt_research/fixtures/nse) and pass its directory to `--dataset`. Keep privately licensed source files locally. One manifest identifies the stable listing, issuer, currency, venue, lot size, adapter, source, capture time and calendar version. Use raw prices with separately recorded dividends. Quote currencies remain distinct across portfolios.

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

The comparison reads the fixed test start, one-session horizon and fold length from the manifest. Earlier prices warm up the factor. The book carries through consecutive test folds. Cash, buy-and-hold including dividends, and one-session price momentum use the same costs and liquidity rules. A second scenario doubles fee and slippage and halves participation. Reports include drawdown, turnover, exposure, fees and net returns. Single-listing results omit FX translation, settlement lags, taxes and stock borrow. A larger licensed universe and borrow records would be needed for portfolio-wide or short-selling studies.

## FinGPT experiments

Supply a locally merged FinGPT GGUF for model experiments. Install `.[model]` in a virtual environment. The model file digest identifies the weights. The evaluator compares FinGPT and the deterministic parser against the same reference fields with field accuracy, citation precision, abstentions, elapsed time and model tokens.

```sh
python3 main.py evaluate --dataset nse --gguf /models/fingpt.gguf
python3 main.py features --dataset nse --gguf /models/fingpt.gguf --model-available-at 2024-12-01T00:00:00Z --latency-seconds 60 > features.json
python3 main.py compare --dataset nse --features features.json
```

Use the actual model availability date and a defensible inference delay in an experiment. The example arguments illustrate the format. The additional policy applies the same price factor with a sentiment filter on its first eligible observation. Treat generated sentiment as a hypothesis. Assess its incremental value on a separately frozen, reviewed sample before drawing conclusions.

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
