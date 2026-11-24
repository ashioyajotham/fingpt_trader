# Kenyan issuer dividend disclosures

The research CLI includes 42 short passages from 38 distinct issuer documents across nine NSE issuers. It runs offline without a broker, price feed or model. The included catalog uses adjudicated annotations from two readers. Model predictions are separate benchmark outputs and cannot change company actions, portfolio cash flows or orders.

## Replay one decision

```sh
python -m fingpt_research actions --as-of 2026-06-24 --symbol EQTY
python -m fingpt_research actions --as-of 2026-06-25 --symbol EQTY --status approved
python -m fingpt_research actions --as-of 2026-06-30 --status approved
python -m fingpt_research actions --as-of 2025-09-12 --symbol EABL --history
```

Equity Group reported a KES 5.75 proposal for FY2025 on March 18. Shareholders approved it on June 24. Under the disclosure clock the approval becomes visible at the start of June 25 in Nairobi. A date-only cutoff includes the entire named Nairobi day. An intraday cutoff requires a UTC offset. The default disclosure clock reconstructs history from the date on each release, applying the following midnight as a uniform availability convention. A board signature, an AGM date and a web publication can occur on different days. Historical website arrival times were unavailable. For simulations requiring witnessed availability use `--clock observed`, which withholds the whole collected snapshot until its verification bound. [Results release](https://equitygroupholdings.com/equity-group-records-historic-results-reporting-a-55-growth-in-profit-after-tax-of-kshs-75-5-billion-underscoring-the-success-of-its-strategic-transformation/) · [AGM release](https://equitygroupholdings.com/nairobi-24th-june-2026-equity-group-holdings-plc-shareholders-have-approved-all-proposed-resolutions-tabled-at-the-22nd-annual-general-meeting-agm-including-dividend-payout-board-appointments-a/)

Each event is grouped by issuer, fiscal year and distribution. Earlier observations remain queryable. A new field assertion supersedes the previous value for that field, retaining a source and date for each surviving assertion. An omitted field supplies no update. EABL's January 2025 interim announcement gave February 21 and April 30 as register and planned payment dates. The August agenda reports February 16 and payment on or about April 28. The history preserves both accounts and the date the later one entered the reconstructed record. The discrepancy's cause is unknown. A date reported as “on or about” retains its wording in the excerpts or linked notice. The covered KEGN FY2023 payment notice concerns private shareholders. Annual totals and individual installments should be examined separately. [Half-year results](https://www.eabl.com/~/media/Files/E/eabl/documents/investor/financial-result/2025/eabl-f25-half-year-results-press-ad-final.pdf) · [AGM agenda](https://www.eabl.com/~/media/Files/E/eabl/documents/investor/agm/2025/eabl-2025-agm-notice-and-agenda-english.pdf)

## Extraction benchmark

The five scored fields are status, per-share amount, currency, record date and payment date. Source selection, event linkage and passage segmentation were performed by the first reader. The benchmark evaluates extraction from selected fragments. Full-document discovery, issuer identification, splitting multi-event filings and financial returns require separate evaluation.

The first collection contains 18 development passages and 17 passages from separate issuers. Both sets were available during rule design. The seven prospective passages cover BAT Kenya and Standard Chartered Kenya. They were collected and labeled after commit `6ab5c84` froze the rules, prompt and routing choice. All three splits have disjoint issuers. An event and its later disclosures stay together. The second reader completed all 42 passages. Of 210 field decisions, the readers agreed on 209 normalized values or shared absences. On 51 of those agreements they chose different verbatim spans. The original citation anchors were retained for those fields. The table below summarizes the saved predictions rescored against the adjudicated values (full raw JSON run dumps are archived at git tag `v0.1-legacy-crypto`).

| Prospective passages, 20 reference fields | Correct | Returned | Correct with reference span | Missed (abstained) | False approvals |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rules | 18 | 19 | 18 | 1 | 0 |
| FinGPT-MT Llama 3 8B | 13 | 21 | 9 | 6 | 0 |
| Rules + FinGPT | 19 | 23 | 19 | 0 | 0 |
| Vertex Gemini 3.5 Flash-Lite | 17 | 17 | 17 | 3 | 0 |
| Rules + Vertex | 19 | 20 | 19 | 0 | 0 |

| Earlier issuer split, 54 reference fields | Correct | Returned | Correct with reference span | False approvals |
| --- | ---: | ---: | ---: | ---: |
| Rules | 54 | 54 | 54 | 0 |
| FinGPT | 33 | 56 | 31 | 2 |
| Rules + FinGPT | 54 | 60 | 54 | 0 |
| Vertex | 48 | 49 | 46 | 0 |
| Rules + Vertex | 54 | 55 | 54 | 0 |

Correct fields require equality to the normalized reference. A supported citation additionally contains the annotated span. A false approval is an approved status returned on a passage whose reference does not say approved. An abstention is a missing prediction on an expected field. Returned includes extra predictions on absent fields. Source-grounding filters require a verbatim quote. A copied quote can still support the wrong interpretation. In the prospective set, the rules read “declared and paid” as declared. Both hybrids retained that rule decision. The second reader entered KES 8 for the interim per-share amount and flagged the omitted unit. The surrounding issuer disclosure confirms the basis by reconciling KES 37 final and KES 8 interim to KES 45 per ordinary share. The catalog accepts the amount. [Issuer release, page 2](https://av.sc.com/ke/content/docs/ke-press-release-standard-chartered-fy-2024-results.pdf#page=2)

The hybrid keeps the rule values and uses the model to fill missing fields only when a core field (status, amount or currency) is missing. It routed 2 of 7 prospective passages and 2 of 17 in the other split. The comparison reused the standalone model's per-passage outputs for these routes. Timings and billing below cover the standalone calls. The model receives only the fragments, prompt and schema, without reference values, identifiers or the target event. Local decoding uses the installed FinGPT Q4_K_M checkpoint with temperature zero, JSON grammar and a 768-token completion cap. The Vertex backend requests the same task, schema, temperature and output cap through the global publisher endpoint. The service reported `gemini-3.5-flash-lite` on every call. A more specific immutable serving revision was unavailable.

The 24 Vertex calls consumed 13,812 input and 2,175 output tokens with zero reported cache or reasoning tokens. Estimated global standard list-price cost was **USD 0.00958**, before credits or taxes. Median prospective request time was 1.50 seconds for Vertex and 7.29 seconds on local Metal for FinGPT. These measurements exclude model initialization. [Google pricing](https://cloud.google.com/model-garden/pricing)

```sh
python -m fingpt_research notice-eval --split prospective
python -m fingpt_research notice-eval --split prospective --gguf /path/to/fingpt.gguf
python -m fingpt_research notice-eval --split prospective --vertex-project YOUR_PROJECT --max-calls 7
python -m tools.dividend_review > blind-review.json
python -m tools.dividend_review --returned dividend-review-return.json
```

Vertex requires an authenticated Google Cloud CLI with model access in the supplied project. This research uses on-demand requests. The review export supplies empty fields, verbatim excerpts, source links and verified PDF pages where available. Some historical issuer PDFs block automated retrieval or have moved. The locator records the direct-fetch outcome or the current PDF address, hash and matched page where verified. The reviewer can record disagreements about the evidence or source in the notes. Reconciliation reports missing reviews, changed values and different spans. The original reference file and prediction runs retain their original corpus hash. The reviewed catalog and the rescored runs use a distinct hash that includes the adjudication.
