"""Command line interface for reproducible research and paper portfolios."""

import argparse
import json
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from .data import FIXTURES, instant, load_dataset, number
from .evidence import LocalFinGPT, extract_rules, evaluate_documents
from .experiments import compare, dataset_hash, load_features, model_features, simulate
from .forward import ForwardBook
from .paper import Assumptions


def encode(value):
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def emit(value, streaming=False):
    print(json.dumps(value, default=encode, sort_keys=True, indent=None if streaming else 2), flush=True)


def get_parser():
    parser = argparse.ArgumentParser(description="Point-in-time market research and paper simulation")
    commands = parser.add_subparsers(dest="command", required=True)
    from .dividends import STATUSES
    actions = commands.add_parser("actions", help="query the sourced dividend catalog")
    actions.add_argument("--as-of", required=True, help="inclusive Nairobi day or timestamp with offset")
    actions.add_argument("--symbol")
    actions.add_argument("--status", choices=STATUSES)
    actions.add_argument("--history", action="store_true")
    actions.add_argument("--clock", choices=("disclosure", "observed"), default="disclosure")
    actions.add_argument("--catalog", type=Path, help="optional local catalog directory (default: bundled fixture)")
    filing = commands.add_parser("filing-extract", help="extract cited dividend events from a complete issuer PDF")
    filing.add_argument("--pdf", type=Path, required=True)
    filing.add_argument("--symbol", required=True)
    filing.add_argument("--source", required=True, help="immutable issuer URL or source identifier")
    filing_backend = filing.add_mutually_exclusive_group()
    filing_backend.add_argument("--vertex-project", help="optional hosted Gemini fallback for uncertain candidates")
    filing_backend.add_argument("--gguf", type=Path, help="optional local GGUF fallback for uncertain candidates")
    filing.add_argument("--vertex-model", default="gemini-3.5-flash-lite")
    filing.add_argument("--max-calls", type=int, default=12, help="maximum hosted Gemini fallback requests")
    promote = commands.add_parser("catalog-add", help="promote filing-extract output into a local dividend catalog")
    promote.add_argument("--report", type=Path, required=True, help="JSON report produced by filing-extract")
    promote.add_argument("--notice-date", required=True, help="issuer disclosure date (YYYY-MM-DD)")
    promote.add_argument("--catalog", type=Path, required=True, help="target local catalog directory")
    nse_imp = commands.add_parser("nse-import", help="import NSE Daily Price List PDF(s) into an nse-daily dataset")
    nse_imp.add_argument("--pdf", type=Path, required=True, help="NSE daily price list PDF file or directory")
    nse_imp.add_argument("--symbol", required=True, help="NSE ticker symbol, e.g. SCOM, EQTY, SCBK")
    nse_imp.add_argument("--out", type=Path, required=True, help="target dataset directory")
    nse_imp.add_argument("--issuer", help="override issuer legal name")
    nse_imp.add_argument("--isin", help="override security ISIN")
    nse_imp.add_argument("--session", help="override session date (YYYY-MM-DD) for single PDF")
    nse_imp.add_argument("--source", help="provenance URL for the price list")
    bench = commands.add_parser("notice-eval", help="evaluate frozen dividend passages")
    bench.add_argument("--split", choices=("development", "holdout", "prospective"), default="prospective")
    backend = bench.add_mutually_exclusive_group()
    backend.add_argument("--gguf", type=Path)
    backend.add_argument("--vertex-project", help="explicitly enable paid Vertex requests")
    bench.add_argument("--vertex-model", default="gemini-3.5-flash-lite")
    bench.add_argument("--max-calls", type=int, default=24, help="maximum Vertex requests")
    basket = commands.add_parser("portfolio", help="aggregate separately funded listings in a base currency")
    basket.add_argument("--spec", type=Path, required=True)
    for name in ("replay", "evidence", "evaluate", "features", "paper", "compare"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--dataset", default="binance" if name in {"replay", "paper"} else "nse",
                         help="nse, binance, or a local dataset directory")
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


def run(args):
    if args.command == "actions":
        from .dividends import CORPUS, actions
        return actions(args.as_of, args.symbol, args.status, args.history, args.clock,
                       root=args.catalog or CORPUS)
    if args.command == "filing-extract":
        from .filings import extract_filing, prepare_filing
        prepared = prepare_filing(args.pdf, args.symbol, args.source)
        model = None
        if args.vertex_project:
            from .vertex import VertexModel
            model = VertexModel(args.vertex_project, args.vertex_model, args.max_calls)
        elif args.gguf:
            model = LocalFinGPT(args.gguf, structured=True)
        return extract_filing(prepared, model=model, fallback_only=True)
    if args.command == "catalog-add":
        from .dividends import promote_filing
        report = json.loads(args.report.read_text())
        return promote_filing(report, args.notice_date, args.catalog)
    if args.command == "nse-import":
        from .nse_pricelist import import_nse_pricelist
        return import_nse_pricelist(
            args.pdf, args.symbol, args.out,
            issuer=args.issuer, isin=args.isin, session_date=args.session, source_url=args.source,
        )
    if args.command == "notice-eval":
        from .dividend_eval import evaluate
        model = None
        if args.gguf:
            model = LocalFinGPT(args.gguf, structured=True)
        elif args.vertex_project:
            from .vertex import VertexModel
            model = VertexModel(args.vertex_project, args.vertex_model, args.max_calls)
        return evaluate(args.split, model)
    if args.command == "portfolio":
        from .portfolio import portfolio
        return portfolio(args.spec)
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
    if not args or (args[0].startswith("-") and args[0] not in {"--help", "-h"}):
        args.insert(0, "replay")
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
