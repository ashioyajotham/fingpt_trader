"""Passage-grounded extraction. Model output never creates a corporate action."""

import hashlib
import json
import re
from time import perf_counter

DIVIDEND_FIELDS = {
    "issuer": r"^([^\n]+? PLC)\s*$",
    "status": r"\b(approved|proposed|cancelled) (?:an? )?(?:(?:final|interim) )?dividend",
    "currency": r"\b(KShs?\.?|[A-Z]{3}) [\d,.]+ per (?:ordinary )?share",
    "amount": r"\b(?:KShs?\.?|[A-Z]{3}) ([\d,.]+) per (?:ordinary )?share",
    "ex_date": r"Ex-date:\s*(\d{4}-\d{2}-\d{2})",
    "book_closure": r"Book closure:\s*(\d{4}-\d{2}-\d{2})",
    "pay_date": r"Payment date:\s*(\d{4}-\d{2}-\d{2})",
}
STATEMENT_FIELDS = {
    "period_end": r"period ended (\d{4}-\d{2}-\d{2})",
    "revenue": r"revenue was [A-Z]{3} ([\d,]+(?:\.\d+)?)",
    "currency": r"revenue was ([A-Z]{3}) [\d,.]+",
}
CLAIM_FIELDS = {"claim": r"^([^\n]+[.])$"}


def fields(doc):
    return {"dividend_notice": DIVIDEND_FIELDS, "financial_statement": STATEMENT_FIELDS}.get(
        doc["kind"], CLAIM_FIELDS
    )


def citation(doc, value, page, quote):
    return {"value": value, "source": doc["source"], "page": page, "quote": quote,
            "attribution": doc["attribution"]}


def extract_rules(doc):
    results = {}
    for field, pattern in fields(doc).items():
        matches = [(page, match) for page, passage in enumerate(doc["pages"], 1)
                   for match in re.finditer(pattern, passage, re.I | re.M)]
        if len(matches) == 1:
            page, match = matches[0]
            value = match[1].strip()
            if field == "currency" and value.rstrip(".").upper() in {"KSH", "KSHS", "KES"}:
                value = "KES"
            if field == "status":
                value = value.lower()
            results[field] = citation(doc, value, page, match[0].strip())
    return results


class LocalFinGPT:
    """Optional, explicitly supplied FinGPT GGUF. Loading never downloads weights."""

    def __init__(self, path):
        from llama_cpp import Llama

        with path.open("rb") as stream:
            self.version = hashlib.file_digest(stream, "sha256").hexdigest()
        self.model = Llama(model_path=str(path), n_ctx=4096, seed=0, verbose=False)
        self.tokens = 0

    def generate_json(self, instruction, doc):
        prompt = "Instruction: " + instruction + "\nInput: " + json.dumps(doc["pages"], ensure_ascii=False)
        response = self.model(prompt + "\nAnswer: ", temperature=0, max_tokens=768)
        self.tokens += response["usage"]["total_tokens"]
        try:
            answer = json.loads(response["choices"][0]["text"])
        except json.JSONDecodeError:
            return {}
        return answer if isinstance(answer, dict) else {}

    def passages(self, doc):
        """Keep prompts within the GGUF context and retain original page numbers."""
        for page, text in enumerate(doc["pages"], 1):
            tokens = self.model.tokenize(text.encode("utf-8"), add_bos=False)
            for start in range(0, len(tokens), 1900):
                fragment = self.model.detokenize(tokens[start:start + 2000]).decode("utf-8", errors="ignore")
                yield page, fragment

    def sentiment(self, doc):
        scores = []
        for _, passage in self.passages(doc):
            answer = self.generate_json(
                'Classify sentiment toward the issuer. Return {"sentiment":"positive|neutral|negative"}.',
                {"pages": [passage]})
            label = answer.get("sentiment")
            if isinstance(label, str) and label in {"positive", "neutral", "negative"}:
                scores.append({"positive": 1, "neutral": 0, "negative": -1}[label])
        if scores:
            return (sum(scores) > 0) - (sum(scores) < 0)
        return None

    def extract(self, doc):
        prompt = ("Extract these fields: " + ", ".join(fields(doc)) +
                  '. Return {field: {"value": string, "page": integer, "quote": exact source text}}.'
                  " Omit unknown fields. Use lowercase status. Count pages from one.")
        matches = {field: {} for field in fields(doc)}
        for page, passage in self.passages(doc):
            candidates = self.generate_json(prompt, {"pages": [passage]})
            for field, alternatives in matches.items():
                item = candidates.get(field)
                if not isinstance(item, dict) or item.get("page") != 1:
                    continue
                quote, value = item.get("quote"), item.get("value")
                if (isinstance(quote, str) and quote.strip() and quote in passage
                        and quote in doc["pages"][page - 1] and isinstance(value, str)
                        and value.strip() and value.casefold() in quote.casefold()):
                    alternatives[value] = citation(doc, value, page, quote)
        return {field: next(iter(values.values())) for field, values in matches.items() if len(values) == 1}



def score_extractor(documents, labels, extractor):
    """Labels enumerate independently checkable field values and supporting spans."""
    start = perf_counter()
    rows = []
    totals = {"expected": 0, "returned": 0, "correct": 0, "supported": 0, "abstentions": 0}
    for doc in documents:
        predicted = extractor(doc)
        expected = labels[doc["id"]]
        for field in fields(doc):
            gold = expected.get(field)
            got = predicted.get(field)
            if gold is None and got is None:
                continue
            correct = bool(gold and got and got["value"] == gold["value"])
            supported = bool(correct and got["page"] == gold["page"] and
                             gold["span"].casefold() in got["quote"].casefold())
            totals["expected"] += int(gold is not None)
            totals["returned"] += int(got is not None)
            totals["correct"] += int(correct)
            totals["supported"] += int(supported)
            totals["abstentions"] += int(gold is not None and got is None)
            rows.append({"document": doc["id"], "field": field, "prediction": got,
                         "correct": correct, "supported": supported})
    return {**totals, "field_accuracy": totals["correct"] / len(rows) if rows else None,
            "citation_precision": totals["supported"] / totals["returned"] if totals["returned"] else None,
            "wall_seconds": perf_counter() - start, "rows": rows}
