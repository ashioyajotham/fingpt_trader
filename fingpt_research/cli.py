import argparse
import json
from datetime import datetime
from decimal import Decimal

from .data import FIXTURES, instant, load_dataset


def encode(value):
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Offline point-in-time market research")
    parser.add_argument("command", choices=["replay"], nargs="?", default="replay")
    parser.add_argument("--dataset", default=str(FIXTURES / "binance"))
    parser.add_argument("--as-of", help="inclusive ISO timestamp with UTC offset")
    args = parser.parse_args(argv)
    try:
        dataset = load_dataset(args.dataset)
        cutoff = instant(args.as_of) if args.as_of else None
        print(json.dumps(dataset.report(cutoff), default=encode, sort_keys=True, indent=2))
    except (ValueError, KeyError, OSError, TypeError) as exc:
        parser.error(str(exc))
    return 0
