"""User-level configuration (~/.config/asof/config.json), .env loader, and startup dashboard."""

import json
import os
from pathlib import Path

MODEL_BACKENDS = ("none", "gemini", "vertex", "gguf")
EOD_PROVIDERS = ("stockanalysis", "eodhd", "csv-url")

SECRET_KEYS = {"model.gemini_api_key", "eod.eodhd_api_key"}

DEFAULTS = {
    "model.backend": "none",
    "model.name": "gemini-2.5-flash",
    "model.gemini_api_key": "",
    "model.vertex_project": "",
    "model.gguf_path": "",
    "model.max_calls": 12,
    "eod.provider": "stockanalysis",
    "eod.eodhd_api_key": "",
    "eod.csv_url": "",
    "paths.universe": "",
    "paths.catalog": "",
}

ENV_MAP = {
    "model.backend": ("ASOF_MODEL_BACKEND",),
    "model.name": ("ASOF_MODEL_NAME", "GEMINI_MODEL", "VERTEX_MODEL"),
    "model.gemini_api_key": ("GEMINI_API_KEY",),
    "model.vertex_project": ("GOOGLE_CLOUD_PROJECT",),
    "model.gguf_path": ("ASOF_GGUF_PATH",),
    "model.max_calls": ("ASOF_MAX_CALLS",),
    "eod.provider": ("ASOF_EOD_PROVIDER",),
    "eod.eodhd_api_key": ("EODHD_API_KEY",),
    "eod.csv_url": ("ASOF_EOD_CSV_URL",),
    "paths.universe": ("ASOF_UNIVERSE_DIR",),
    "paths.catalog": ("ASOF_CATALOG_DIR",),
}


def default_config_path(override=None):
    if override:
        return Path(override).expanduser()
    env_path = os.environ.get("ASOF_CONFIG", "").strip()
    if env_path:
        return Path(env_path).expanduser()
    return Path.home() / ".config" / "asof" / "config.json"


def _parse_dotenv(path):
    """Minimal stdlib .env parser (KEY=VALUE, strips quotes & comments, never mutates os.environ)."""
    values = {}
    if not path or not Path(path).is_file():
        return values
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in {"'", '"'}:
            val = val[1:-1]
        if key and val:
            values[key] = val
    return values


def _flatten(nested, prefix=""):
    flat = {}
    if not isinstance(nested, dict):
        return flat
    for k, v in nested.items():
        full = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(_flatten(v, full))
        else:
            flat[full] = v
    return flat


def _unflatten(flat):
    nested = {}
    for key, val in sorted(flat.items()):
        parts = key.split(".")
        cur = nested
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = val
    return nested


def _validate_value(key, raw):
    if key not in DEFAULTS:
        raise ValueError(f"unknown config key {key!r}; valid keys: {', '.join(sorted(DEFAULTS))}")
    if key == "model.backend":
        val = str(raw).strip().lower()
        if val not in MODEL_BACKENDS:
            raise ValueError(f"model.backend must be one of {', '.join(MODEL_BACKENDS)}")
        return val
    if key == "eod.provider":
        val = str(raw).strip().lower()
        if val not in EOD_PROVIDERS:
            raise ValueError(f"eod.provider must be one of {', '.join(EOD_PROVIDERS)}")
        return val
    if key == "model.max_calls":
        try:
            num = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("model.max_calls must be a positive integer") from exc
        if num <= 0:
            raise ValueError("model.max_calls must be a positive integer")
        return num
    return str(raw).strip()


def read_config_file(config_path=None):
    path = default_config_path(config_path)
    if not path.is_file():
        return {}, path
    raw = json.loads(path.read_text(encoding="utf-8"))
    flat = _flatten(raw)
    cleaned = {}
    for k, v in flat.items():
        if k in DEFAULTS and v not in (None, ""):
            cleaned[k] = _validate_value(k, v)
    return cleaned, path


def write_config_file(flat_values, config_path=None):
    path = default_config_path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nested = _unflatten({k: v for k, v in flat_values.items() if k in DEFAULTS and v not in (None, "")})
    path.write_text(json.dumps(nested, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return path


def load_effective_config(config_path=None, *, cwd=None):
    """Resolve effective config with precedence: OS env > .env > config.json > defaults."""
    work_dir = Path(cwd) if cwd else Path.cwd()
    file_vals, resolved_path = read_config_file(config_path)
    dotenv_vals = _parse_dotenv(work_dir / ".env")

    values = dict(DEFAULTS)
    sources = {k: "default" for k in DEFAULTS}

    for k, v in file_vals.items():
        values[k] = v
        sources[k] = "config"

    for k, env_names in ENV_MAP.items():
        for name in env_names:
            if name in dotenv_vals:
                values[k] = _validate_value(k, dotenv_vals[name])
                sources[k] = ".env"
                break

    for k, env_names in ENV_MAP.items():
        for name in env_names:
            env_val = os.environ.get(name, "").strip()
            if env_val:
                values[k] = _validate_value(k, env_val)
                sources[k] = f"env:{name}"
                break

    # Convenience inference: if backend wasn't explicitly set but a Gemini key or Vertex project is in env/.env,
    # keep backend="none" unless explicitly configured so rules-only stays default and surprise charges never happen.
    return values, sources, resolved_path


def mask_secret(value):
    text = str(value or "").strip()
    if not text:
        return None
    if len(text) <= 8:
        return "****"
    return f"{text[:4]}****{text[-4:]}"


def redact_config(values, sources, config_path):
    """Return a JSON-safe redacted config summary with provenance for each key."""
    entries = {}
    for key in sorted(DEFAULTS):
        val = values[key]
        if key in SECRET_KEYS:
            display = mask_secret(val)
        else:
            display = val if val != "" else None
        entries[key] = {"value": display, "source": sources[key]}
    return {
        "config_file": str(config_path),
        "exists": Path(config_path).is_file(),
        "settings": entries,
    }


def set_config_value(key, value, config_path=None):
    validated = _validate_value(key, value)
    current, path = read_config_file(config_path)
    current[key] = validated
    write_config_file(current, path)
    vals, srcs, _ = load_effective_config(path)
    return redact_config(vals, srcs, path)


def unset_config_value(key, config_path=None):
    if key not in DEFAULTS:
        raise ValueError(f"unknown config key {key!r}; valid keys: {', '.join(sorted(DEFAULTS))}")
    current, path = read_config_file(config_path)
    current.pop(key, None)
    write_config_file(current, path)
    vals, srcs, _ = load_effective_config(path)
    return redact_config(vals, srcs, path)


def reset_config(config_path=None):
    path = default_config_path(config_path)
    if path.is_file():
        path.unlink()
    vals, srcs, _ = load_effective_config(path)
    return redact_config(vals, srcs, path)


def interactive_init(config_path=None, *, input_fn=input, output_fn=print):
    """Interactive terminal setup wizard for model fallback, EOD provider, and local paths."""
    current, path = read_config_file(config_path)
    vals, _, _ = load_effective_config(path)

    def ask(prompt, default=""):
        suffix = f" [{default}]" if default else ""
        ans = input_fn(f"{prompt}{suffix}: ").strip()
        return ans if ans else str(default)

    output_fn(f"Configuring AsOf user settings ({path})")
    backend = ask("Model fallback backend (none / gemini / vertex / gguf)", vals["model.backend"]).lower()
    backend = _validate_value("model.backend", backend)
    current["model.backend"] = backend

    if backend == "gemini":
        current["model.name"] = ask("Gemini model ID", vals["model.name"] or "gemini-2.5-flash")
        key_hint = "leave blank to use GEMINI_API_KEY env/.env"
        raw_key = input_fn(f"Gemini API key ({key_hint}): ").strip()
        if raw_key:
            current["model.gemini_api_key"] = raw_key
    elif backend == "vertex":
        proj = ask("Google Cloud project ID", vals["model.vertex_project"])
        if proj:
            current["model.vertex_project"] = proj
        current["model.name"] = ask("Vertex Gemini model ID", vals["model.name"] or "gemini-3.5-flash-lite")
    elif backend == "gguf":
        gguf = ask("Path to local .gguf model file", vals["model.gguf_path"])
        if gguf:
            current["model.gguf_path"] = gguf

    if backend != "none":
        current["model.max_calls"] = _validate_value("model.max_calls", ask("Max fallback calls per filing", vals["model.max_calls"]))

    provider = ask("NSE EOD sync provider (stockanalysis / eodhd / csv-url)", vals["eod.provider"]).lower()
    provider = _validate_value("eod.provider", provider)
    current["eod.provider"] = provider
    if provider == "eodhd":
        raw_eod = input_fn("EODHD API key (leave blank to use EODHD_API_KEY env/.env): ").strip()
        if raw_eod:
            current["eod.eodhd_api_key"] = raw_eod
    elif provider == "csv-url":
        csv_url = ask("Default NSE daily CSV URL", vals["eod.csv_url"])
        if csv_url:
            current["eod.csv_url"] = csv_url

    uni = ask("Default local NSE universe directory (optional)", vals["paths.universe"] or "./local-data/universe")
    if uni:
        current["paths.universe"] = uni
    cat = ask("Default local catalog directory (optional)", vals["paths.catalog"] or "./local-data/catalog")
    if cat:
        current["paths.catalog"] = cat

    write_config_file(current, path)
    eff_vals, eff_srcs, _ = load_effective_config(path)
    return redact_config(eff_vals, eff_srcs, path)


def _describe_universe(path_str, work_dir):
    candidate = Path(path_str).expanduser() if path_str else (work_dir / "local-data" / "universe")
    manifest = candidate / "manifest.json"
    if not manifest.is_file():
        return "not configured  (run: asof nse-sync --universe ./local-data/universe)"
    try:
        meta = json.loads(manifest.read_text(encoding="utf-8"))
        tickers = meta.get("tickers", {})
        last_days = [info.get("last_session") for info in tickers.values() if info.get("last_session")]
        latest = max(last_days) if last_days else "unknown"
        tag = "" if path_str else " [auto-detected]"
        return f"{candidate}  ({len(tickers)} tickers · as of {latest}){tag}"
    except (OSError, json.JSONDecodeError, TypeError):
        return f"{candidate}  (unreadable manifest)"


def _describe_catalog(path_str, work_dir):
    candidate = Path(path_str).expanduser() if path_str else (work_dir / "local-data" / "catalog")
    notices = candidate / "notices.json"
    if notices.is_file():
        try:
            count = len(json.loads(notices.read_text(encoding="utf-8")))
            tag = "" if path_str else " [auto-detected]"
            return f"{candidate}  ({count} notices · writable){tag}"
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    return "bundled read-only  (42 reviewed NSE notices)"


def _describe_model(values):
    backend = values["model.backend"]
    if backend == "none":
        return "none  (deterministic rules-only · zero API calls)"
    if backend == "gemini":
        key_state = "key set" if values["model.gemini_api_key"] else "missing GEMINI_API_KEY"
        return f"gemini  ({values['model.name']} · {key_state} · max {values['model.max_calls']} calls)"
    if backend == "vertex":
        proj = values["model.vertex_project"] or "missing project"
        return f"vertex  ({values['model.name']} · project={proj} · max {values['model.max_calls']} calls)"
    gguf = values["model.gguf_path"] or "missing path"
    return f"gguf  ({gguf})"


def _describe_eod(values):
    provider = values["eod.provider"]
    if provider == "stockanalysis":
        return "stockanalysis  (free NASE raw OHLCV · no key required)"
    if provider == "eodhd":
        key_state = "key set" if values["eod.eodhd_api_key"] else "missing EODHD_API_KEY"
        return f"eodhd  ({key_state})"
    return f"csv-url  ({values['eod.csv_url'] or 'no default URL'})"


def render_startup_dashboard(version, tagline, banner, config_path=None, *, cwd=None):
    """Render the bare `asof` startup dashboard (Option A: banner + status + starter commands)."""
    work_dir = Path(cwd) if cwd else Path.cwd()
    values, _, resolved_cfg = load_effective_config(config_path, cwd=work_dir)
    uni_desc = _describe_universe(values["paths.universe"], work_dir)
    cat_desc = _describe_catalog(values["paths.catalog"], work_dir)
    model_desc = _describe_model(values)
    eod_desc = _describe_eod(values)
    cfg_note = str(resolved_cfg) + (" (loaded)" if resolved_cfg.is_file() else " (defaults)")

    lines = [
        banner,
        f"v{version}  ·  config: {cfg_note}",
        "",
        "Workspace Status",
        f"  Universe : {uni_desc}",
        f"  Catalog  : {cat_desc}",
        f"  Model    : {model_desc}",
        f"  EOD Sync : {eod_desc}",
        "",
        "Quick Start",
        "  asof config init                               Configure Gemini / Vertex / GGUF & paths",
        "  asof nse-sync --universe ./local-data/universe Append latest closed NSE EOD sessions",
        "  asof serve                                     Launch the 5-tab local web workstation",
        "  asof factor-backtest --factor dividend_yield   Run a point-in-time NSE factor backtest",
        "  asof filing-extract --pdf FILE --symbol SCBK --source URL",
        "",
        "Run `asof --help` for all commands.",
    ]
    return "\n".join(lines)
