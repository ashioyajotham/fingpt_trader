"""Command line interface for reproducible research and paper portfolios."""

import argparse
import json
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from . import TAGLINE, __version__
from .config import (
    interactive_init,
    load_effective_config,
    redact_config,
    render_startup_dashboard,
    reset_config,
    set_config_value,
    unset_config_value,
)
from .data import FIXTURES, instant, load_dataset, number
from .evidence import LocalFinGPT, extract_rules, evaluate_documents
from .experiments import compare, dataset_hash, load_features, model_features, simulate
from .forward import ForwardBook
from .paper import Assumptions

BANNER = (
    "    ___          ____  ____\n"
    "   /   |  _____ / __ \\/ __/\n"
    "  / /| | / ___// / / / /_\n"
    " / ___ |(__  )/ /_/ / __/\n"
    "/_/  |_/____/ \\____/_/\n"
    f"\n{TAGLINE}"
)


def encode(value):
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def emit(value, streaming=False):
    print(json.dumps(value, default=encode, sort_keys=True, indent=None if streaming else 2), flush=True)


def get_parser():
    parser = argparse.ArgumentParser(
        prog="asof",
        description=BANNER,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"asof {__version__} — {TAGLINE}")
    parser.add_argument("--config", type=Path, help="optional path to user config.json (default: ~/.config/asof/config.json)")
    commands = parser.add_subparsers(dest="command", required=False)

    cfg_cmd = commands.add_parser("config", help="view or configure Gemini API, Vertex AI, GGUF, EOD sync, and default paths")
    cfg_sub = cfg_cmd.add_subparsers(dest="config_action")
    cfg_sub.add_parser("show", help="show effective configuration with redacted secrets")
    cfg_sub.add_parser("init", help="interactive setup wizard for model fallback, EOD provider, and local paths")
    cfg_set = cfg_sub.add_parser("set", help="set one config key (e.g. model.backend gemini)")
    cfg_set.add_argument("key")
    cfg_set.add_argument("value")
    cfg_unset = cfg_sub.add_parser("unset", help="remove one key from user config")
    cfg_unset.add_argument("key")
    cfg_sub.add_parser("reset", help="delete user config file and revert to defaults")

    from .dividends import STATUSES
    actions = commands.add_parser("actions", help="query the sourced dividend catalog")
    actions.add_argument("--as-of", required=True, help="inclusive Nairobi day or timestamp with offset")
    actions.add_argument("--symbol")
    actions.add_argument("--status", choices=STATUSES)
    actions.add_argument("--history", action="store_true")
    actions.add_argument("--clock", choices=("disclosure", "observed"), default="disclosure")
    actions.add_argument("--catalog", type=Path, help="optional local catalog directory (default: configured or bundled)")
    filing = commands.add_parser("filing-extract", help="extract cited dividend events and financials from a complete issuer PDF")
    filing.add_argument("--pdf", type=Path, required=True)
    filing.add_argument("--symbol", required=True)
    filing.add_argument("--source", required=True, help="immutable issuer URL or source identifier")
    filing_backend = filing.add_mutually_exclusive_group()
    filing_backend.add_argument("--gemini", action="store_true", help="use direct Gemini API fallback (GEMINI_API_KEY)")
    filing_backend.add_argument("--vertex-project", help="optional hosted Vertex AI Gemini fallback for uncertain candidates")
    filing_backend.add_argument("--gguf", type=Path, help="optional local GGUF fallback for uncertain candidates")
    filing_backend.add_argument("--rules-only", action="store_true", help="force deterministic rules only (ignore configured model)")
    filing.add_argument("--gemini-model", help="Gemini API model ID (default: configured or gemini-2.5-flash)")
    filing.add_argument("--gemini-api-key", help="optional Gemini API key override")
    filing.add_argument("--vertex-model", help="Vertex Gemini model ID (default: configured or gemini-3.5-flash-lite)")
    filing.add_argument("--max-calls", type=int, help="maximum hosted model fallback requests (default: configured or 12)")
    fin_ext = commands.add_parser("financials-extract", help="extract cited revenue, PAT, EPS, and balance-sheet figures from a PDF")
    fin_ext.add_argument("--pdf", type=Path, required=True)
    fin_ext.add_argument("--symbol", required=True)
    fin_ext.add_argument("--source", required=True, help="immutable issuer URL or source identifier")
    promote = commands.add_parser("catalog-add", help="promote filing-extract output into a local dividend catalog")
    promote.add_argument("--report", type=Path, required=True, help="JSON report produced by filing-extract")
    promote.add_argument("--notice-date", required=True, help="issuer disclosure date (YYYY-MM-DD)")
    promote.add_argument("--catalog", type=Path, required=True, help="target local catalog directory")
    fin_add = commands.add_parser("financials-add", help="promote extracted financials into a local point-in-time catalog")
    fin_add.add_argument("--report", type=Path, required=True, help="JSON report from filing-extract or financials-extract")
    fin_add.add_argument("--notice-date", required=True, help="issuer disclosure date (YYYY-MM-DD)")
    fin_add.add_argument("--catalog", type=Path, required=True, help="target local catalog directory")
    fin_q = commands.add_parser("financials", help="query persisted financial statements as of a date")
    fin_q.add_argument("--as-of", required=True, help="inclusive Nairobi day or timestamp with offset")
    fin_q.add_argument("--catalog", type=Path, help="local catalog directory (default: configured paths.catalog)")
    fin_q.add_argument("--symbol")
    fin_q.add_argument("--metric")
    fin_q.add_argument("--history", action="store_true")
    fin_q.add_argument("--clock", choices=("disclosure", "observed"), default="disclosure")
    nse_imp = commands.add_parser("nse-import", help="import NSE Daily Price List PDF/CSV file(s) into dataset or universe")
    nse_imp.add_argument("--input", "--pdf", dest="input_path", type=Path, required=True,
                         help="NSE daily price list PDF/CSV file or directory")
    nse_imp.add_argument("--symbol", help="single NSE ticker symbol, e.g. SCOM, EQTY, SCBK")
    nse_imp.add_argument("--out", type=Path, help="target single-listing dataset directory")
    nse_imp.add_argument("--universe", type=Path, help="bulk-import multi-ticker CSV(s) into a universe directory")
    nse_imp.add_argument("--symbols", help="optional comma-separated ticker filter for --universe")
    nse_imp.add_argument("--sectors", type=Path, help="optional NSE market sectors CSV (auto-detected in --input dir)")
    nse_imp.add_argument("--issuer", help="override issuer legal name (single-listing mode)")
    nse_imp.add_argument("--isin", help="override security ISIN (single-listing mode)")
    nse_imp.add_argument("--session", help="override session date (YYYY-MM-DD) for single file")
    nse_imp.add_argument("--source", help="provenance URL for the price list")
    nse_imp.add_argument("--ocr-cmd", help="optional OCR shell template for image-only PDFs using {in} and {out}")
    nse_sync = commands.add_parser("nse-sync", help="append completed daily NSE EOD bars from a configured provider")
    nse_sync.add_argument("--universe", type=Path, help="target multi-ticker universe directory (default: configured paths.universe)")
    nse_sync.add_argument("--provider", choices=("stockanalysis", "eodhd", "csv-url"), help="EOD provider (default: configured or stockanalysis)")
    nse_sync.add_argument("--symbols", help="optional comma-separated tickers (default: existing universe or 12 blue-chips)")
    nse_sync.add_argument("--range", dest="range_code", choices=("1M", "6M", "1Y", "5Y", "MAX"), default="6M",
                          help="history window for stockanalysis provider (default: 6M)")
    nse_sync.add_argument("--since", help="optional inclusive start date (YYYY-MM-DD)")
    nse_sync.add_argument("--until", help="optional inclusive end date (YYYY-MM-DD, capped at last closed NSE session)")
    nse_sync.add_argument("--api-key", help="optional API key (or set EODHD_API_KEY / `asof config`)")
    nse_sync.add_argument("--url", dest="csv_url", help="remote NSE daily CSV URL when --provider csv-url")
    nse_sync.add_argument("--overwrite", action="store_true", help="allow overwriting already-stored sessions")
    nse_sync.add_argument("--delay", type=float, default=0.15, help="seconds between per-ticker requests (default: 0.15)")
    bench = commands.add_parser("notice-eval", help="evaluate frozen dividend passages")
    bench.add_argument("--split", choices=("development", "holdout", "prospective"), default="prospective")
    backend = bench.add_mutually_exclusive_group()
    backend.add_argument("--gemini", action="store_true", help="use direct Gemini API (GEMINI_API_KEY)")
    backend.add_argument("--gguf", type=Path)
    backend.add_argument("--vertex-project", help="explicitly enable paid Vertex requests")
    backend.add_argument("--rules-only", action="store_true", help="force deterministic rules only")
    bench.add_argument("--gemini-model", help="Gemini API model ID")
    bench.add_argument("--gemini-api-key", help="optional Gemini API key override")
    bench.add_argument("--vertex-model", help="Vertex Gemini model ID")
    bench.add_argument("--max-calls", type=int, help="maximum hosted model requests")
    basket = commands.add_parser("portfolio", help="aggregate separately funded listings in a base currency")
    basket.add_argument("--spec", type=Path, required=True)
    fac = commands.add_parser("factors", help="evaluate cross-sectional point-in-time factors (Rank-IC and spreads)")
    fac.add_argument("--universe", type=Path, help="multi-ticker universe directory (default: configured paths.universe)")
    fac.add_argument("--catalog", type=Path, help="optional local dividend/financials catalog directory")
    fac.add_argument("--horizons", default="1,5,20", help="comma-separated forward session horizons")
    fac.add_argument("--min-adv", type=float, default=0.0, help="minimum 20-day KES turnover filter")
    fac.add_argument("--blend", help="optional composite factor blend, e.g. 'dividend_yield:0.5,momentum_20d:-0.5'")
    fbt = commands.add_parser("factor-backtest", help="run a point-in-time cross-sectional factor portfolio backtest")
    fbt.add_argument("--universe", type=Path, help="multi-ticker universe directory (default: configured paths.universe)")
    fbt.add_argument("--catalog", type=Path, help="optional local dividend/financials catalog directory")
    fbt.add_argument("--factor", default="momentum_20d",
                     choices=("momentum_20d", "momentum_60d", "liquidity_adv20", "dividend_yield", "earnings_yield"))
    fbt.add_argument("--factor-blend", help="optional composite factor blend, e.g. 'dividend_yield:0.5,momentum_20d:-0.5'")
    fbt.add_argument("--direction", default="top", choices=("top", "bottom"),
                     help="long highest factor scores (top) or lowest scores (bottom, e.g. reversal)")
    fbt.add_argument("--top-n", type=int, default=5, help="number of ranked listings held per rebalance")
    fbt.add_argument("--weighting", default="equal", choices=("equal", "rank"))
    fbt.add_argument("--rebalance", default="monthly", choices=("daily", "weekly", "monthly"))
    fbt.add_argument("--min-adv", type=float, default=0.0, help="minimum 20-day KES turnover filter")
    fbt.add_argument("--cash", type=number, default=Decimal("1000000"))
    fbt.add_argument("--fee-bps", type=number, default=Decimal("10"))
    fbt.add_argument("--slippage-bps", type=number, default=Decimal("5"))
    fbt.add_argument("--participation", type=number, default=Decimal("0.05"))
    fbt.add_argument("--max-weight", type=number, default=Decimal("0.35"))
    fbt.add_argument("--max-sector-weight", type=number, help="optional maximum aggregate weight per sector, e.g. 0.40")
    fbt.add_argument("--wht-pct", type=number, default=Decimal("5"), help="dividend withholding tax percent (default: 5)")
    fbt.add_argument("--no-dividends", action="store_true", help="disable catalog cash dividend crediting (price-only)")
    fbt.add_argument("--start", help="optional inclusive start session (YYYY-MM-DD)")
    fbt.add_argument("--end", help="optional inclusive end session (YYYY-MM-DD)")
    web = commands.add_parser("serve", help="start the local AsOf web workstation")
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=int, default=8765)
    web.add_argument("--catalog", type=Path, help="optional local dividend/financials catalog directory")
    web.add_argument("--universe", type=Path, help="optional multi-ticker universe directory for NSE Board & Factor Lab")
    web.add_argument("--allow-remote", action="store_true", help="explicitly permit binding to a non-loopback address")
    for name in ("replay", "evidence", "evaluate", "features", "paper", "compare"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--dataset", default="nse",
                         help="nse, binance, or a local dataset directory (default: nse)")
        cmd.add_argument("--as-of", type=instant, help="inclusive ISO timestamp with UTC offset")
        if name in {"evaluate", "evidence", "features"}:
            cmd.add_argument("--structured", action="store_true", help="use schema-constrained local model decoding")
            cmd.add_argument("--gguf", type=Path, required=name == "features",
                             help="local FinGPT weights for optional inference")
        if name == "evaluate":
            cmd.add_argument("--corpus", type=Path, help="standalone directory with source documents and reference labels")
            cmd.add_argument("--labels", type=Path, help="reviewed JSON references (default: dataset labels.json)")
        if name == "features":
            cmd.add_argument("--model-available-at", type=instant, required=True)
            cmd.add_argument("--latency-seconds", type=int, required=True)
        if name in {"paper", "compare"}:
            cmd.add_argument("--features", type=Path, help="timestamped model outputs from the features command")
            cmd.add_argument("--cash", type=number, default=Assumptions.cash)
            cmd.add_argument("--fee-bps", type=number, default=Assumptions.fee_bps)
            cmd.add_argument("--slippage-bps", type=number, default=Assumptions.slippage_bps)
            cmd.add_argument("--participation", type=number, default=Assumptions.participation)
            cmd.add_argument("--max-exposure", type=number, default=Assumptions.max_exposure)
        if name == "paper":
            cmd.add_argument("--policy", choices=["cash", "buy_hold", "price", "price_ml", "short_hold"], default="cash")
            cmd.add_argument("--borrow", type=Path, help="listing-specific JSON borrow history")
            cmd.add_argument("--follow", action="store_true",
                             help="consume subsequent append-only dataset directory paths on stdin, one per line")
    return parser


def _resolve_model(args, cfg_vals, *, default_max_calls=12):
    if getattr(args, "rules_only", False):
        return None
    max_calls = args.max_calls if getattr(args, "max_calls", None) is not None else (cfg_vals["model.max_calls"] or default_max_calls)
    if getattr(args, "gemini", False):
        from .gemini import GeminiAPIModel
        return GeminiAPIModel(
            api_key=getattr(args, "gemini_api_key", None) or cfg_vals["model.gemini_api_key"],
            model=getattr(args, "gemini_model", None) or cfg_vals["model.name"] or "gemini-2.5-flash",
            max_calls=max_calls,
        )
    if getattr(args, "vertex_project", None):
        from .vertex import VertexModel
        return VertexModel(
            args.vertex_project,
            getattr(args, "vertex_model", None) or cfg_vals["model.name"] or "gemini-3.5-flash-lite",
            max_calls,
        )
    if getattr(args, "gguf", None):
        return LocalFinGPT(args.gguf, structured=True)

    # Fall back to configured user backend (default is "none" -> deterministic rules-only).
    backend = cfg_vals["model.backend"]
    if backend == "gemini":
        from .gemini import GeminiAPIModel
        return GeminiAPIModel(
            api_key=cfg_vals["model.gemini_api_key"],
            model=cfg_vals["model.name"] or "gemini-2.5-flash",
            max_calls=max_calls,
        )
    if backend == "vertex":
        from .vertex import VertexModel
        if not cfg_vals["model.vertex_project"]:
            raise ValueError("model.backend=vertex requires model.vertex_project (set via `asof config init`)")
        return VertexModel(
            cfg_vals["model.vertex_project"],
            cfg_vals["model.name"] or "gemini-3.5-flash-lite",
            max_calls,
        )
    if backend == "gguf":
        if not cfg_vals["model.gguf_path"]:
            raise ValueError("model.backend=gguf requires model.gguf_path (set via `asof config init`)")
        return LocalFinGPT(Path(cfg_vals["model.gguf_path"]).expanduser(), structured=True)
    return None


def _cfg_path(cli_path, cfg_vals, key, *, required_flag=None):
    if cli_path is not None:
        return cli_path
    configured = cfg_vals.get(key, "")
    if configured:
        return Path(configured).expanduser()
    if required_flag:
        raise ValueError(f"{required_flag} is required (or set {key} via `asof config init`)")
    return None


def run(args):
    cfg_override = getattr(args, "config", None)
    if args.command is None:
        print(render_startup_dashboard(__version__, TAGLINE, BANNER, config_path=cfg_override))
        return None
    if args.command == "config":
        action = args.config_action or "show"
        if action == "show":
            vals, srcs, path = load_effective_config(cfg_override)
            return redact_config(vals, srcs, path)
        if action == "init":
            return interactive_init(cfg_override)
        if action == "set":
            return set_config_value(args.key, args.value, cfg_override)
        if action == "unset":
            return unset_config_value(args.key, cfg_override)
        if action == "reset":
            return reset_config(cfg_override)
        raise ValueError(f"unknown config action: {action}")

    cfg_vals, _, _ = load_effective_config(cfg_override)
    if args.command == "actions":
        from .dividends import CORPUS, actions
        cat = _cfg_path(args.catalog, cfg_vals, "paths.catalog") or CORPUS
        return actions(args.as_of, args.symbol, args.status, args.history, args.clock, root=cat)
    if args.command == "filing-extract":
        from .filings import extract_filing, prepare_filing
        prepared = prepare_filing(args.pdf, args.symbol, args.source)
        model = _resolve_model(args, cfg_vals, default_max_calls=12)
        return extract_filing(prepared, model=model, fallback_only=True, pdf_path=args.pdf)
    if args.command == "financials-extract":
        from .filings import extract_filing_financials
        return extract_filing_financials(args.pdf, args.symbol, args.source)
    if args.command == "catalog-add":
        from .dividends import promote_filing
        report = json.loads(args.report.read_text())
        return promote_filing(report, args.notice_date, args.catalog)
    if args.command == "financials-add":
        from .financials import promote_financials
        report = json.loads(args.report.read_text())
        return promote_financials(report, args.notice_date, args.catalog)
    if args.command == "financials":
        from .financials import query_financials
        cat = _cfg_path(args.catalog, cfg_vals, "paths.catalog", required_flag="--catalog")
        return query_financials(
            args.as_of, root=cat, symbol=args.symbol, metric=args.metric,
            history=args.history, clock=args.clock,
        )
    if args.command == "nse-import":
        from .nse_pricelist import import_nse_pricelist, import_nse_universe
        if args.universe:
            syms = [s.strip() for s in args.symbols.split(",") if s.strip()] if args.symbols else None
            return import_nse_universe(args.input_path, args.universe, symbols=syms,
                                       source_url=args.source, sectors_csv=args.sectors)
        if not args.symbol or not args.out:
            raise ValueError("nse-import requires either --universe DIR or both --symbol and --out")
        return import_nse_pricelist(
            args.input_path, args.symbol, args.out,
            issuer=args.issuer, isin=args.isin, session_date=args.session,
            source_url=args.source, ocr_cmd=args.ocr_cmd,
        )
    if args.command == "nse-sync":
        from .nse_sync import sync_nse_universe
        uni = _cfg_path(args.universe, cfg_vals, "paths.universe", required_flag="--universe")
        provider = args.provider or cfg_vals["eod.provider"] or "stockanalysis"
        api_key = args.api_key or cfg_vals["eod.eodhd_api_key"] or None
        csv_url = args.csv_url or cfg_vals["eod.csv_url"] or None
        syms = [s.strip() for s in args.symbols.split(",") if s.strip()] if args.symbols else None
        return sync_nse_universe(
            uni,
            provider=provider,
            symbols=syms,
            range_code=args.range_code,
            since=args.since,
            until=args.until,
            api_key=api_key,
            csv_url=csv_url,
            overwrite=args.overwrite,
            delay_seconds=args.delay,
        )
    if args.command == "notice-eval":
        from .dividend_eval import evaluate
        model = _resolve_model(args, cfg_vals, default_max_calls=24)
        return evaluate(args.split, model)
    if args.command == "portfolio":
        from .portfolio import portfolio
        return portfolio(args.spec)
    if args.command == "factors":
        from .factors import evaluate_factors, load_universe
        uni = _cfg_path(args.universe, cfg_vals, "paths.universe", required_flag="--universe")
        cat = _cfg_path(args.catalog, cfg_vals, "paths.catalog")
        horizons = tuple(int(h.strip()) for h in args.horizons.split(",") if h.strip())
        if not horizons or any(h <= 0 for h in horizons):
            raise ValueError("horizons must be positive integers")
        return evaluate_factors(load_universe(uni), catalog_dir=cat,
                                horizons=horizons, min_adv=args.min_adv, blend=args.blend)
    if args.command == "factor-backtest":
        from .factor_backtest import FactorBacktestConfig, backtest_factor
        from .factors import load_universe
        uni = _cfg_path(args.universe, cfg_vals, "paths.universe", required_flag="--universe")
        cat = _cfg_path(args.catalog, cfg_vals, "paths.catalog")
        cfg = FactorBacktestConfig(
            factor=args.factor, factor_blend=args.factor_blend, direction=args.direction,
            top_n=args.top_n, weighting=args.weighting, rebalance=args.rebalance, min_adv=args.min_adv,
            cash=args.cash, fee_bps=args.fee_bps, slippage_bps=args.slippage_bps,
            participation=args.participation, max_weight=args.max_weight,
            max_sector_weight=args.max_sector_weight,
            include_dividends=not args.no_dividends, wht_pct=args.wht_pct,
            start=args.start, end=args.end,
        )
        return backtest_factor(load_universe(uni), cfg, catalog_dir=cat)
    if args.command == "serve":
        from .dividends import CORPUS
        from .web import serve
        cat = _cfg_path(args.catalog, cfg_vals, "paths.catalog") or CORPUS
        uni = _cfg_path(args.universe, cfg_vals, "paths.universe")
        serve(host=args.host, port=args.port, catalog_root=cat,
              universe_root=uni, allow_remote=args.allow_remote)
        return None
    if getattr(args, "structured", False) and not args.gguf:
        raise ValueError("--structured requires a local model")
    if args.command == "evaluate" and args.corpus:
        if args.as_of:
            raise ValueError("standalone excerpt evaluation does not use a market replay clock")
        documents = json.loads((args.corpus / "documents.json").read_text())
        labels = json.loads((args.labels or args.corpus / "labels.json").read_text())
        model = LocalFinGPT(args.gguf, structured=args.structured) if args.gguf else None
        return {"corpus": args.corpus.name, "extractors": evaluate_documents(documents, labels, model)}
    root = FIXTURES / args.dataset if args.dataset in {"binance", "nse", "nse-public-sample"} else Path(args.dataset)
    dataset = load_dataset(root)
    full_dataset = dataset
    if args.as_of:
        from dataclasses import replace

        dataset = replace(dataset, bars=[b for b in dataset.bars if b.available_at <= args.as_of],
                          documents=[d for d in dataset.documents if d["available_at"] <= args.as_of])
    if args.command == "replay":
        return dataset.report(None)
    if args.command in {"evidence", "evaluate", "features"}:
        model = LocalFinGPT(args.gguf, structured=args.structured) if args.gguf else None
        if args.command == "features":
            return model_features(model, dataset, args.model_available_at, args.latency_seconds)
        if args.command == "evidence":
            extract = model.extract if model else extract_rules
            return [{"id": d["id"], "kind": d["kind"], "source": d["source"],
                     "attribution": d["attribution"], "available_at": d["available_at"],
                     "extractor": model.version if model else "rules-v1",
                     "fields": extract(d)} for d in dataset.documents]
        labels = json.loads((args.labels or root / "labels.json").read_text())
        return {"dataset": dataset.meta["source"], "extractors": evaluate_documents(dataset.documents, labels, model)}
    config = Assumptions(args.cash, args.fee_bps, args.slippage_bps, args.participation, args.max_exposure)
    features = load_features(args.features, full_dataset)
    if args.as_of:
        features = [f for f in features if f.observed_at <= args.as_of]
    if args.command == "compare":
        return compare(dataset, config, features, dataset_hash(root))
    if args.policy == "price_ml" and not args.features:
        raise ValueError("price_ml requires a timestamped feature input")
    borrow = None
    if args.borrow:
        from .borrow import BorrowSchedule
        history = json.loads(args.borrow.read_text())
        borrow = BorrowSchedule(history["instrument"], history["records"])
    if args.follow:
        if args.as_of:
            raise ValueError("use an initial snapshot directory for forward simulation")
        book = ForwardBook(dataset, args.policy, config, features, borrow)
        emit(book.engine.report(), streaming=True)
        for line in sys.stdin:
            if line.strip():
                next_dataset = load_dataset(Path(line.strip()))
                if book.advance(next_dataset, load_features(args.features, next_dataset)):
                    emit(book.engine.report(), streaming=True)
        return None
    return simulate(dataset, args.policy, config, features, borrow=borrow).report()


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    parser = get_parser()
    try:
        result = run(parser.parse_args(args))
        if result is not None:
            emit(result)
    except (ValueError, KeyError, OSError, TypeError, ImportError, IndexError) as exc:
        parser.error(str(exc))
    return 0


def backtest():
    return main(["compare", *sys.argv[1:]])
