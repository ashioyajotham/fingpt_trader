# FinGPT Trader

FinGPT Trader is a Python 3.11+ toolkit for point-in-time financial research. Run the default command from a source checkout to replay twelve captured daily Binance BTC/USDT candles. It uses the Python standard library and reads the fixture from disk.

```sh
python3 main.py
python3 main.py replay --as-of 2025-01-05T00:00:00Z
python3 -m unittest tests.test_system_integration
```

Each dataset manifest identifies its listing, venue, currency, raw price basis, calendar version, capture time and source. Replay uses the documented historical availability assumption. The optional legacy model, exchange clients, strategies and optimizer remain available in their original Python modules. Model confidence requires held-out calibration.

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
