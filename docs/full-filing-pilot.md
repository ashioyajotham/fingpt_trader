# Dividend extraction from full filings

The research CLI reads an entire local issuer PDF and returns dividend event candidates with page evidence. It requires the optional `filings` dependency. Local inference also requires the `model` extra and a GGUF checkpoint. The source argument records provenance. The command reads the supplied file offline.

```sh
python -m pip install '.[filings]'
python -m fingpt_research filing-extract --pdf issuer-release.pdf --symbol SCBK --source "$ISSUER_URL"
python -m fingpt_research filing-extract --pdf issuer-release.pdf --symbol SCBK --source "$ISSUER_URL" --gguf /path/to/fingpt.gguf
```

Outputs identify the listing, PDF hash, coverage of each original page, candidate fragments, event fields, citations and method. Coverage describes machine-readable text. A complete read still requires measurement of event recall. A partial document with zero extracted events needs manual inspection. The research report is separate from catalog adjudication and portfolio accounting.

## Frozen six-document selection

Freeze the candidate finder, rules, FinGPT prompt, grouping behavior and this protocol in Git before opening the evaluation filings. Use the SCBK FY2024 release and the EABL FY2025 agenda and polling report only for development. The existing reviewed excerpts continue to exercise the passage benchmark.

- Select six distinct complete issuer-authored PDFs of one to ten pages from at least three Nairobi-listed issuers. Use issuers outside the nine in the excerpt benchmark to exclude previously seen documents and dividend events. A dated public notice through November 24, 2026 is eligible. Verify issuer identity, downloadability and a usable text layer. Record the source URL, title, symbol, stable local filename, page count and SHA-256 of each PDF.
- Use purposeful sampling. Look for two annual or interim earnings or dividend disclosures, two meeting documents or resolution reports and two notices about other corporate business. Prefer sections likely to contain multiple distributions and notices likely to contain none. Within each group use the first documents found satisfying eligibility, aiming for distinct document layouts. Exclude a second notice on the same distribution if it would dominate the set. Log unavailable, image-only, oversized or duplicate sources encountered. Freeze the six identities before running either extractor. Keep all six even if manual reading or extraction later contradicts the expected mix.
- A reader inspects every page, records all issuer dividend events, resolves repeated discussion within the filing and marks genuine zero-event documents explicitly. Cover the reporting period or resolution under discussion, including already-paid interim installments and explicit cancellations. A purely historical comparison, dividend policy, par value, accounting balance or annual sum of already identified installments does not independently create an event. Treat an explicit recommendation to pay no dividend as an absence of a distribution, unless a specific prior distribution is explicitly cancelled. Preserve conflicting, ambiguous and approximate terms in notes. Require a page and supporting words for every asserted gold field.
- Prepare a separate first-reader inventory. Give the second reader complete originals, a blank form and stable file identities. Keep both methods' predictions private until the second reader returns the inventory. Reconcile both readers' event lists, evidence and fields before scoring. Report source-access attrition and any divergence between actual corpus composition and target composition.

## Running the comparison

```sh
python -m tools.filing_benchmark --manifest /path/to/manifest.json --out /path/to/private-predictions --gguf /path/to/fingpt.gguf
```

The runner verifies every frozen PDF, prepares the candidates once and passes that same evidence to rules and FinGPT. Reports include a digest of the ordered candidates, model identity, token usage and extraction time. Keep the original predictions when adjudicating or rescoring. Timing excludes checkpoint loading. The pilot measures event discovery and field extraction; the unit of evaluation is a full filing.
