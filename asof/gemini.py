"""Direct Google Gemini Developer API adapter (GEMINI_API_KEY) for text and scanned PDF fallback."""

import base64
import json
import os
import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .evidence import EvidenceModel

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

FINANCIAL_IMAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "currency": {"type": "string"},
        "period": {"type": "string"},
        "scale_multiplier": {"type": "number"},
        "metrics": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "enum": [
                            "revenue",
                            "profit_before_tax",
                            "profit_after_tax",
                            "eps",
                            "total_assets",
                            "total_equity",
                            "customer_deposits",
                            "loans_and_advances",
                        ],
                    },
                    "raw_value": {"type": "number"},
                    "prior_raw_value": {"type": "number"},
                    "unit": {"type": "string"},
                    "quote": {"type": "string"},
                },
                "required": ["name", "raw_value", "quote"],
            },
        },
    },
    "required": ["metrics"],
}


def _http_post_json(url, payload, headers, timeout=60):
    req = Request(url, json.dumps(payload).encode("utf-8"), headers=headers)
    try:
        with urlopen(req, timeout=timeout) as res:
            return json.load(res)
    except HTTPError as exc:
        try:
            detail = json.load(exc).get("error", {}).get("message", str(exc))
        except Exception:
            detail = str(exc)
        raise ValueError(f"Gemini API HTTP {exc.code}: {detail}") from exc


class GeminiAPIModel(EvidenceModel):
    """Bounded Gemini API fallback authenticated via x-goog-api-key header (zero extra dependencies)."""

    name = "Gemini"
    provider = "Gemini API"

    def __init__(self, api_key=None, model=DEFAULT_GEMINI_MODEL, max_calls=12, *, requester=_http_post_json):
        token = (api_key or os.environ.get("GEMINI_API_KEY", "")).strip()
        if not token:
            raise ValueError("Gemini API backend requires GEMINI_API_KEY (or set via `asof config init`)")
        if not re.fullmatch(r"[a-zA-Z0-9._-]+", model):
            raise ValueError(f"invalid Gemini model ID: {model!r}")
        if max_calls <= 0:
            raise ValueError("max_calls must be positive")
        self._api_key = token
        self.version = model
        self.endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        self.max_calls = max_calls
        self._requester = requester
        self.tokens = 0
        self.diagnostics = Counter()

    def passages(self, doc):
        yield from enumerate(doc["pages"], 1)

    def _call(self, parts, schema=None, *, max_output_tokens=768):
        if self.diagnostics["calls"] >= self.max_calls:
            raise ValueError("Gemini API request budget exhausted")
        config = {"temperature": 0, "maxOutputTokens": max_output_tokens, "responseMimeType": "application/json"}
        if schema is not None:
            config["responseJsonSchema"] = schema
        payload = {"contents": [{"role": "user", "parts": parts}], "generationConfig": config}
        headers = {"x-goog-api-key": self._api_key, "Content-Type": "application/json"}
        self.diagnostics["calls"] += 1
        result = self._requester(self.endpoint, payload, headers, 60)
        usage = result.get("usageMetadata") or {}
        self.tokens += int(usage.get("totalTokenCount", 0))
        for key in ("promptTokenCount", "candidatesTokenCount", "thoughtsTokenCount", "cachedContentTokenCount"):
            if key in usage:
                self.diagnostics[key] += int(usage[key])
        self.diagnostics["version:" + result.get("modelVersion", "unspecified")] += 1
        candidates = result.get("candidates") or []
        candidate = candidates[0] if candidates else {}
        self.diagnostics["finish:" + candidate.get("finishReason", "missing")] += 1
        res_parts = candidate.get("content", {}).get("parts") or []
        try:
            value = json.loads("".join(p.get("text", "") for p in res_parts if not p.get("thought")))
        except json.JSONDecodeError:
            self.diagnostics["invalid_json"] += 1
            return {}
        return value if isinstance(value, dict) else {}

    def generate_json(self, instruction, doc, schema=None):
        prompt = "Instruction: " + instruction + "\nInput: " + json.dumps(doc["pages"], ensure_ascii=False)
        return self._call([{"text": prompt}], schema=schema)

    def extract_financials_from_image(self, image_bytes, symbol, page_num, *, mime_type="image/png"):
        """Multimodal fallback for scanned/image-only financial statement pages (strictly schema-validated)."""
        prompt = (
            f"Extract cited financial statement headline figures for {symbol} from page {page_num} of this issuer filing. "
            "Detect the table scale multiplier (1 for unscaled, 1000 for '000/thousands, 1000000 for millions, 1000000000 for billions). "
            "Keep EPS unscaled (KES/share). Return ONLY figures explicitly printed on this page with an exact verbatim quote."
        )
        parts = [
            {"text": prompt},
            {"inline_data": {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode("ascii")}},
        ]
        parsed = self._call(parts, schema=FINANCIAL_IMAGE_SCHEMA, max_output_tokens=1024)
        self.diagnostics["multimodal_pages"] += 1
        currency = str(parsed.get("currency") or "KES").strip().upper()
        period = str(parsed.get("period") or "unknown").strip()
        try:
            scale = Decimal(str(parsed.get("scale_multiplier") or 1))
            if scale <= 0:
                scale = Decimal(1)
        except InvalidOperation:
            scale = Decimal(1)

        out = {}
        for item in parsed.get("metrics") or []:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            quote = str(item.get("quote") or "").strip()
            if not name or not quote or item.get("raw_value") is None:
                continue
            try:
                raw_val = Decimal(str(item["raw_value"]))
            except InvalidOperation:
                continue
            eff_scale = Decimal(1) if name == "eps" else scale
            scaled_val = raw_val * eff_scale
            prior_scaled = None
            if item.get("prior_raw_value") is not None:
                try:
                    prior_scaled = format(Decimal(str(item["prior_raw_value"])) * eff_scale, "f")
                except InvalidOperation:
                    prior_scaled = None
            entry = {
                "value": format(scaled_val, "f"),
                "raw_value": format(raw_val, "f"),
                "currency": currency,
                "unit": f"{currency}/share" if name == "eps" else currency,
                "scale": format(eff_scale, "f"),
                "period": period,
                "method": "multimodal_fallback",
                "citations": [{"page": page_num, "quote": quote}],
            }
            if prior_scaled is not None:
                entry["prior_value"] = prior_scaled
            out[name] = entry
        return out
