# Market and model validation, November 23, 2026

## A. Public exchange connectivity

The read-only Binance probe retrieved twelve historical BTC/USDT daily candles and received a current public WebSocket trade through the repository's trade parser. The REST response matched the pinned fixture. Its SHA-256 was `8668ce75fc2341affc8e7cb214376c870ad5fd9c7144420b1ceb2f97c83c41f2`. The stream returned trade 6900340914 at 10:58:16.938 UTC. Reproduce with `python tools/verify_public_binance.py`. The probe exercises public data only.

## B. NSE source coverage

The public replay covers one Safaricom listing and one documented session. The source identifies the security by ISIN, gives high, low, VWAP, previous price and volume, and summarizes its final dividend. The importer preserves the absent open, uses the public list as a valuation source and records the time of verification as the conservative availability bound. The row was checked visually against page 2. A production history still needs a defined universe and date range, source rights, session coverage and independently reconciled corporate actions. [NSE daily list hosted by I&M](https://www.imbankgroup.com/ke/wp-content/uploads/sites/2/2026/09/02-SEP-26.pdf)

## C. FinGPT extraction

We ran the Q4_K_M quantization of FinGPT-MT Llama-3 8B locally with llama-cpp-python 0.3.38. The pinned revision was `bd23528d188976308fe113f53179954514997c21` and the downloaded SHA-256 was `4bebb8b47fc9d113f01ab5113152074a806600017ea12040e547acce0a35922a`. The corpus contains six synthetic documents and two short issuer excerpts. The reference labels were transcribed before inference. An independent human review is outstanding. [Model card and adapter provenance](https://huggingface.co/second-state/FinGPT-MT-Llama-3-8B-LoRA-GGUF/tree/bd23528d188976308fe113f53179954514997c21)

| Corpus | Extractor | Returned | Correct / expected | Correct with reference span |
| --- | --- | ---: | ---: | ---: |
| Synthetic | Rules | 20 | 20 / 20 | 20 |
| Synthetic | FinGPT, raw completion | 0 | 0 / 20 | 0 |
| Synthetic | FinGPT, schema-constrained | 18 | 18 / 20 | 16 |
| Issuer excerpts | Rules | 5 | 5 / 6 | 5 |
| Issuer excerpts | FinGPT, raw completion | 0 | 0 / 6 | 0 |
| Issuer excerpts | FinGPT, schema-constrained | 4 | 3 / 6 | 3 |

Every raw completion failed strict JSON parsing. The follow-up experiment constrained output syntax and retained the same passages, reference values and citation checks. Both synthetic dividend status fields were missing in the constrained model output. On real excerpts, returning the source currency spelling instead of the normalized currency code cost one field. These are end-to-end extraction results on selected excerpts. They do not establish full-filing accuracy. The rules path remains the default. [KCB release](https://kcbgroup.com/download-report?document=kcb-group-press-release--fy-2025-performance--final.pdf) · [Equity Group AGM release](https://equitygroupholdings.com/nairobi-24th-june-2026-equity-group-holdings-plc-shareholders-have-approved-all-proposed-resolutions-tabled-at-the-22nd-annual-general-meeting-agm-including-dividend-payout-board-appointments-a/)

### Provider design

The shared `EvidenceModel` task code owns the field names, page grounding, abstention and sentiment interpretation. Each backend supplies passage windows, a generation method, model identity and usage diagnostics. The shipped backend runs local GGUF weights. For a hosted comparison, freeze a common set of verbatim passages before dispatch so tokenizer differences cannot silently change the inputs. Store the provider, model revision, prompt and decoding mode, latency, usage, errors and model availability with each run.

| Candidate | First use | Configuration to pin |
| --- | --- | --- |
| Vertex AI | Compare a managed Gemini Flash model to the local model on identical passages. | Project, region, exact supported model ID, authentication and structured-output capability. [Google Cloud model API](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/migrate/openai/overview) |
| Baseten | Compare a hosted catalog model, or a dedicated deployment carrying a specific adapter. | Endpoint or deployment revision, served model, API key reference and schema support. [Baseten Model APIs](https://docs.baseten.co/inference/model-apis/overview) |
| vLLM | Compare serving behavior while controlling the base weights and FinGPT adapter. | Server version, model and adapter revisions, chat template, quantization and guided-decoding configuration. [vLLM serving API](https://docs.vllm.ai/en/v0.10.2/serving/openai_compatible_server.html) |

Model choice and hosting are independent experimental variables. Quantization, templates and sampling can change output even when model families match. A fair provider comparison needs the same frozen input, a recorded request and response, and endpoint-specific capability checks. The hosted options above are design candidates. This evaluation used local compute only.

## D. Paper portfolio checks

The constrained model produced six dated sentiment features. We used the selected revision's October 7, 2024 publication time and a declared 60-second inference delay for the historical reconstruction. On the synthetic NSE replay, both the price-only factor and the factor with sentiment returned 0.2413% after modeled costs. The fixture provides no evidence of a financial edge.

The separate two-currency scenario allocates independent KES and USDT books. With the example's assumed loan terms, the KES book borrowed 100 fictional shares. Its recall was blocked by zero volume and then suspension. The January 13 buy-in covered all 100 shares with the specified premium. The book incurred KES 33.5342 in loan fees and owed the KES 120 dividend. These numbers verify cash-flow and timing mechanics. Empirical short-performance work needs dated loan inventory, fees, recall terms and executable market observations.

The [study specification](../examples/kes-portfolio.json) fixes rates and assumptions. The [recorded evaluations](../results/fingpt-mt-8b/) include field-level predictions, abstentions and supporting passages.
