"""Command line interface for reproducible research and paper portfolios."""

import argparse
import json
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from .data import FIXTURES, instant, load_dataset, number
from .evidence import LocalFinGPT, extract_rules, score_extractor
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
    for name in ("replay", "evidence", "evaluate", "features", "paper", "compare"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--dataset", default="binance" if name in {"replay", "paper"} else "nse",
                         help="nse, binance, or a local dataset directory")
        cmd.add_argument("--as-of", type=instant, help="inclusive ISO timestamp with UTC offset")
        if name in {"evaluate", "evidence", "features"}:
            cmd.add_argument("--gguf", type=Path, required=name == "features",
                             help="local FinGPT weights for optional inference")
        if name == "evaluate":
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
            cmd.add_argument("--policy", choices=["cash", "buy_hold", "price", "price_ml"], default="cash")
            cmd.add_argument("--follow", action="store_true",
                             help="consume subsequent append-only dataset directory paths on stdin, one per line")
    return parser


def run(args):
    root = FIXTURES / args.dataset if args.dataset in {"binance", "nse"} else Path(args.dataset)
    dataset = load_dataset(root)
    full_dataset = dataset
    if args.as_of:
        from dataclasses import replace

        dataset = replace(dataset, bars=[b for b in dataset.bars if b.available_at <= args.as_of],
                          documents=[d for d in dataset.documents if d["available_at"] <= args.as_of])
    if args.command == "replay":
        return dataset.report(None)
    if args.command in {"evidence", "evaluate", "features"}:
        model = LocalFinGPT(args.gguf) if args.gguf else None
        if args.command == "features":
            return model_features(model, dataset, args.model_available_at, args.latency_seconds)
        if args.command == "evidence":
            extract = model.extract if model else extract_rules
            return [{"id": d["id"], "kind": d["kind"], "source": d["source"],
                     "attribution": d["attribution"], "available_at": d["available_at"],
                     "extractor": model.version if model else "rules-v1",
                     "fields": extract(d)} for d in dataset.documents]
        labels = json.loads((args.labels or root / "labels.json").read_text())
        scores = {"rules": score_extractor(dataset.documents, labels, extract_rules)}
        if model:
            scores["FinGPT"] = score_extractor(dataset.documents, labels, model.extract)
            scores["FinGPT"].update(model_sha256=model.version, tokens=model.tokens)
        return {"dataset": dataset.meta["source"], "extractors": scores}
    config = Assumptions(args.cash, args.fee_bps, args.slippage_bps, args.participation, args.max_exposure)
    features = load_features(args.features, full_dataset)
    if args.as_of:
        features = [f for f in features if f.observed_at <= args.as_of]
    if args.command == "compare":
        return compare(dataset, config, features, dataset_hash(root))
    if args.policy == "price_ml" and not args.features:
        raise ValueError("price_ml requires a timestamped feature input")
    if args.follow:
        if args.as_of:
            raise ValueError("use an initial snapshot directory for forward simulation")
        book = ForwardBook(dataset, args.policy, config, features)
        emit(book.engine.report(), streaming=True)
        for line in sys.stdin:
            if line.strip():
                next_dataset = load_dataset(Path(line.strip()))
                if book.advance(next_dataset, load_features(args.features, next_dataset)):
                    emit(book.engine.report(), streaming=True)
        return None
    return simulate(dataset, args.policy, config, features).report()


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
