"""Explicit pay-per-request Vertex backend for the bounded notice benchmark."""
import json
import re
import subprocess
from collections import Counter
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .evidence import EvidenceModel


class VertexModel(EvidenceModel):
    name = "Vertex"
    provider = "Vertex AI"

    def __init__(self, project, model, max_calls):
        if not re.fullmatch(r"[a-z][a-z0-9-]+", project) or not re.fullmatch(r"[a-zA-Z0-9.-]+", model):
            raise ValueError("invalid Vertex project or publisher model ID")
        self.version = model
        self.endpoint = (f"https://aiplatform.googleapis.com/v1/projects/{project}/locations/global/"
                         f"publishers/google/models/{model}:generateContent")
        self.max_calls = max_calls
        self.tokens = 0
        self.diagnostics = Counter()
        self.token = subprocess.check_output(["gcloud", "auth", "print-access-token"], text=True).strip()

    def passages(self, doc):
        yield from enumerate(doc["pages"], 1)

    def generate_json(self, instruction, doc, schema=None):
        if self.diagnostics["calls"] >= self.max_calls:
            raise ValueError("Vertex request budget exhausted")
        prompt = "Instruction: " + instruction + "\nInput: " + json.dumps(doc["pages"], ensure_ascii=False)
        config = {"temperature": 0, "maxOutputTokens": 768, "responseMimeType": "application/json",
                  "responseJsonSchema": schema}
        data = {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": config}
        req = Request(self.endpoint, json.dumps(data).encode(), headers={
            "Authorization": "Bearer " + self.token, "Content-Type": "application/json"})
        self.diagnostics["calls"] += 1
        try:
            with urlopen(req, timeout=60) as res:
                result = json.load(res)
        except HTTPError as exc:
            detail = json.load(exc).get("error", {}).get("message", str(exc))
            raise ValueError(f"Vertex HTTP {exc.code}: {detail}") from exc
        usage = result["usageMetadata"]
        self.tokens += usage["totalTokenCount"]
        for key in ("promptTokenCount", "candidatesTokenCount", "thoughtsTokenCount", "cachedContentTokenCount"):
            self.diagnostics[key] += usage.get(key, 0)
        self.diagnostics["version:" + result.get("modelVersion", "unspecified")] += 1
        candidates = result.get("candidates", [])
        candidate = candidates[0] if candidates else {}
        self.diagnostics["finish:" + candidate.get("finishReason", "missing")] += 1
        parts = candidate.get("content", {}).get("parts", [])
        try:
            value = json.loads("".join(p.get("text", "") for p in parts if not p.get("thought")))
        except json.JSONDecodeError:
            self.diagnostics["invalid_json"] += 1
            return {}
        return value if isinstance(value, dict) else {}
