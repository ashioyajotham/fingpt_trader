"""Aggregate separately funded paper books using point-in-time FX marks."""

import json
from bisect import bisect_right
from collections import defaultdict
from dataclasses import replace
from decimal import Decimal

from .borrow import BorrowSchedule
from .data import FIXTURES, instant, load_dataset, number
from .experiments import load_features
from .paper import Assumptions, PaperEngine
from .risk import drawdown


class FX:
    def __init__(self, base, rows, max_age_hours):
        self.base = base
        self.max_age = number(max_age_hours) * 3600
        if self.max_age <= 0:
            raise ValueError("FX maximum age must be positive")
        self.quotes = defaultdict(list)
        for row in rows:
            observed, available = instant(row["observed_at"]), instant(row["available_at"])
            rate = number(row["base_per_unit"])
            if (row["currency"] == base or observed > available or rate <= 0 or not row["source"]):
                raise ValueError("invalid FX quote, time or provenance")
            self.quotes[row["currency"]].append((available, observed, rate))
        self.times = {}
        for ccy, quotes in self.quotes.items():
            quotes.sort()
            times = [q[0] for q in quotes]
            if len(set(times)) != len(times) or any(a[1] > b[1] for a, b in zip(quotes, quotes[1:])):
                raise ValueError("ambiguous or out-of-order FX quote")
            self.times[ccy] = times

    def rate(self, currency, stamp):
        if currency == self.base:
            return Decimal(1)
        idx = bisect_right(self.times.get(currency, []), stamp) - 1
        if idx < 0:
            raise ValueError(f"missing FX rate for {currency} at {stamp}")
        _, observed, rate = self.quotes[currency][idx]
        if (stamp - observed).total_seconds() > self.max_age:
            raise ValueError(f"stale FX rate for {currency} at {stamp}")
        return rate


def portfolio(spec_path):
    spec = json.loads(spec_path.read_text())
    start, end = instant(spec["start"]), instant(spec["end"])
    capital = number(spec["initial_base_cash"])
    if start >= end or capital <= 0:
        raise ValueError("invalid study interval or initial cash")
    fx = FX(spec["base_currency"], spec["fx"], spec["max_fx_age_hours"])
    weights = [number(row["weight"]) for row in spec["sleeves"]]
    if min(weights, default=0) <= 0 or sum(weights) != 1:
        raise ValueError("positive sleeve weights must sum to one")
    max_mark_age = number(spec["max_mark_age_hours"]) * 3600
    if max_mark_age <= 0:
        raise ValueError("maximum mark age must be positive")
    costs = Assumptions(**{k: number(v) for k, v in spec.get("assumptions", {}).items()})
    books, currencies = {}, {}
    events = defaultdict(list)
    for row, weight in zip(spec["sleeves"], weights):
        folder = row["dataset"]
        root = FIXTURES / folder if folder in {"nse", "binance", "nse-public-sample"} else spec_path.parent / folder
        dataset = load_dataset(root)
        ticker, currency = dataset.meta["instrument"], dataset.meta["currency"]
        if ticker in books:
            raise ValueError("one independently funded sleeve per listing")
        local_cash = capital * weight / fx.rate(currency, start)
        schedule = BorrowSchedule(ticker, row["borrow"]) if "borrow" in row else None
        features = load_features(spec_path.parent / row["features"], dataset) if "features" in row else []
        if row["policy"] == "price_ml" and not row.get("features"):
            raise ValueError("price_ml requires model features")
        books[ticker] = PaperEngine(dataset.meta, row["policy"], replace(costs, cash=local_cash), schedule)
        currencies[ticker] = currency
        for bar in dataset.bars:
            if start < bar.close_time and start < bar.available_at <= end:
                events[bar.available_at].append((ticker, bar, dataset.documents, features))
    times = {start, end, *events}
    for currency in set(currencies.values()):
        times.update(t for t in fx.times.get(currency, []) if start < t <= end)
    ledger = []
    peak = capital
    for stamp in sorted(times):
        for ticker, bar, docs, features in events[stamp]:
            books[ticker].observe(bar, docs, features)
        nav, gross, net = Decimal(0), Decimal(0), Decimal(0)
        for ticker, book in books.items():
            rate = fx.rate(currencies[ticker], stamp)
            if book.units and (stamp - book.last_mark.close_time).total_seconds() > max_mark_age:
                raise ValueError(f"stale position mark for {ticker} at {stamp}")
            mark = book.units * book.last_mark.close if book.last_mark else 0
            local_nav = book.ledger[-1]["nav"] if book.ledger else book.cash
            nav += local_nav * rate
            net += mark * rate
            gross += abs(mark) * rate
        peak = max(peak, nav)
        ledger.append({"as_of": stamp, "nav": nav, "drawdown": drawdown(nav, peak),
                       "gross_exposure": gross / nav if nav > 0 else None,
                       "net_exposure": net / nav if nav > 0 else None})
    return {"base_currency": fx.base, "initial_nav": capital, "net_return": ledger[-1]["nav"] / capital - 1,
            "maximum_drawdown": max(r["drawdown"] for r in ledger), "ledger": ledger,
            "sleeves": {k: {"currency": currencies[k], **b.report()} for k, b in books.items()}}
