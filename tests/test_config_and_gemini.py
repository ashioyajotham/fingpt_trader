import json
import os
import stat

import pytest

from asof.cli import get_parser, run as run_cli
from asof.config import (
    interactive_init,
    load_effective_config,
    mask_secret,
    redact_config,
    render_startup_dashboard,
    reset_config,
    set_config_value,
    unset_config_value,
)
from asof.filings import extract_filing
from asof.gemini import GeminiAPIModel


def test_config_lifecycle_precedence_and_secret_redaction(tmp_path, monkeypatch):
    cfg_path = tmp_path / "asof-config.json"
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("ASOF_MODEL_BACKEND", raising=False)
    monkeypatch.delenv("EODHD_API_KEY", raising=False)

    # Set values via CLI `asof config set`.
    out = run_cli(get_parser().parse_args([
        "--config", str(cfg_path), "config", "set", "model.backend", "gemini",
    ]))
    assert out["settings"]["model.backend"] == {"value": "gemini", "source": "config"}
    assert stat.S_IMODE(cfg_path.stat().st_mode) == 0o600

    out = set_config_value("model.gemini_api_key", "demo-token-secret-9876", cfg_path)
    # Secrets MUST always be masked in redacted output.
    assert out["settings"]["model.gemini_api_key"]["value"] == "demo****9876"
    assert "token-secret" not in json.dumps(out)

    # Precedence: OS env > .env > config.json > default.
    work_dir = tmp_path / "project"
    work_dir.mkdir()
    (work_dir / ".env").write_text('GEMINI_API_KEY="demo-dotenv-token-2222"\nASOF_EOD_PROVIDER=eodhd\n')
    vals, srcs, _ = load_effective_config(cfg_path, cwd=work_dir)
    assert vals["model.gemini_api_key"] == "demo-dotenv-token-2222"
    assert srcs["model.gemini_api_key"] == ".env"
    assert vals["eod.provider"] == "eodhd"

    monkeypatch.setenv("GEMINI_API_KEY", "demo-osenv-token-0000")
    vals2, srcs2, _ = load_effective_config(cfg_path, cwd=work_dir)
    assert vals2["model.gemini_api_key"] == "demo-osenv-token-0000"
    assert srcs2["model.gemini_api_key"] == "env:GEMINI_API_KEY"

    # Unset and reset work cleanly.
    unset_config_value("model.backend", cfg_path)
    vals3, _, _ = load_effective_config(cfg_path, cwd=work_dir)
    assert vals3["model.backend"] == "none"

    reset_config(cfg_path)
    assert not cfg_path.exists()
    assert mask_secret("short") == "****"


def test_config_interactive_init_and_startup_dashboard(tmp_path, monkeypatch):
    cfg_path = tmp_path / "cfg.json"
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    answers = iter([
        "gemini",
        "gemini-2.5-flash",
        "demo-init-token-5678",
        "8",
        "stockanalysis",
        str(tmp_path / "uni"),
        str(tmp_path / "cat"),
    ])
    res = interactive_init(cfg_path, input_fn=lambda _p: next(answers), output_fn=lambda _m: None)
    assert res["settings"]["model.backend"]["value"] == "gemini"
    assert res["settings"]["model.name"]["value"] == "gemini-2.5-flash"
    assert res["settings"]["model.max_calls"]["value"] == 8
    assert res["settings"]["model.gemini_api_key"]["value"] == "demo****5678"

    dash = render_startup_dashboard("0.7.0", "tagline", "BANNER", config_path=cfg_path, cwd=tmp_path)
    assert "Workspace Status" in dash
    assert "gemini  (gemini-2.5-flash · key set · max 8 calls)" in dash
    assert "stockanalysis" in dash


def test_gemini_api_model_text_and_multimodal_scanned_pdf_fallback(tmp_path, monkeypatch):
    seen_headers = []

    def fake_post(url, payload, headers, timeout):
        seen_headers.append((url, headers, payload))
        parts = payload["contents"][0]["parts"]
        if any("inline_data" in p for p in parts):
            return {
                "modelVersion": "gemini-2.5-flash",
                "usageMetadata": {"totalTokenCount": 420, "promptTokenCount": 300, "candidatesTokenCount": 120},
                "candidates": [{
                    "finishReason": "STOP",
                    "content": {"parts": [{"text": json.dumps({
                        "currency": "KES",
                        "period": "2024",
                        "scale_multiplier": 1000000,
                        "metrics": [
                            {"name": "profit_after_tax", "raw_value": 24467, "prior_raw_value": 18200, "quote": "Profit after tax 24,467"},
                            {"name": "eps", "raw_value": 12.54, "quote": "Earnings per share 12.54"},
                        ],
                    })}]},
                }],
            }
        return {
            "modelVersion": "gemini-2.5-flash",
            "usageMetadata": {"totalTokenCount": 120},
            "candidates": [{
                "finishReason": "STOP",
                "content": {"parts": [{"text": json.dumps({"status": {"value": "approved", "quote": "approved final dividend"}})}]},
            }],
        }

    model = GeminiAPIModel(api_key="demo-gemini-token-6789", model="gemini-2.5-flash", max_calls=4, requester=fake_post)
    out = model.generate_json("Extract status", {"pages": ["Shareholders approved final dividend"]})
    assert out["status"]["value"] == "approved"
    # API key MUST travel via x-goog-api-key header, NEVER in the URL query string.
    assert "demo-gemini-token" not in seen_headers[0][0]
    assert seen_headers[0][1]["x-goog-api-key"] == "demo-gemini-token-6789"

    # Multimodal fallback triggers ONLY on flagged partial/unreadable pages and scales PAT while keeping EPS unscaled.
    monkeypatch.setattr("asof.filings._render_pdf_page_png", lambda _p, _n: b"\x89PNG\r\n\x1a\nfakepng")
    prepared = {
        "symbol": "KPLC",
        "source": "https://example.test/kplc-fy24.pdf",
        "pdf_sha256": "abc123",
        "coverage": {"status": "partial", "page_count": 1, "pages": [{"page": 1, "status": "partial"}]},
        "candidates": [],
        "financials": {},
    }
    report = extract_filing(prepared, model=model, fallback_only=True, pdf_path=tmp_path / "kplc.pdf")
    assert report["method"]["provider"] == "Gemini API"
    assert report["financials"]["profit_after_tax"]["value"] == "24467000000"
    assert report["financials"]["profit_after_tax"]["method"] == "multimodal_fallback"
    assert report["financials"]["eps"]["value"] == "12.54"  # EPS stays unscaled
    assert report["method"]["diagnostics"]["multimodal_pages"] == 1
