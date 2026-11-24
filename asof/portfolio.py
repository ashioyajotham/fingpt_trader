"""Aggregate multi-asset paper books using point-in-time FX marks, rebalancing, and risk attribution."""

import json
from bisect import bisect_right
from collections import defaultdict
from dataclasses import replace
from decimal import Decimal

from .borrow import BorrowSchedule
from .data import FIXTURES, instant, load_dataset, number
from .experiments import load_features
from .paper import Assumptions, PaperEngine
from .risk import benchmark_attribution, concentration_metrics, drawdown, sharpe_ratio, sortino_ratio


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


def _rebalance_bucket(stamp, schedule):
    if schedule == "daily":
        return stamp.date().isoformat()
    if schedule == "weekly":
        iso = stamp.isocalendar()
        return f"{iso.year}-W{iso.week:02d}"
    if schedule == "monthly":
        return f"{stamp.year}-{stamp.month:02d}"
    raise ValueError(f"unsupported rebalance schedule: {schedule}")


def _resolve_root(folder, spec_path):
    return FIXTURES / folder if folder in {"nse", "binance", "nse-public-sample"} else spec_path.parent / folder


def _book_local_nav(book):
    if book.last_mark is None:
        return book.cash
    receivable = book.ledger[-1]["receivable"] if book.ledger else Decimal(0)
    return book.cash + book.units * book.last_mark.close + receivable


def _rebalance_cash(books, currencies, target_weights, fx, stamp):
    """Transfer available free cash across sleeves toward target base-currency weights (NAV-conserving)."""
    rates = {t: fx.rate(currencies[t], stamp) for t in books}
    base_navs = {t: _book_local_nav(b) * rates[t] for t, b in books.items()}
    total_nav = sum(base_navs.values(), Decimal(0))
    if total_nav <= 0:
        return None
    donors, receivers = {}, {}
    for t, book in books.items():
        target = total_nav * target_weights[t]
        delta = target - base_navs[t]
        if delta < 0:
            free_base = max(Decimal(0), book.cash * rates[t])
            offer = min(-delta, free_base)
            if offer > 0:
                donors[t] = offer
        elif delta > 0:
            receivers[t] = delta
    pool = min(sum(donors.values(), Decimal(0)), sum(receivers.values(), Decimal(0)))
    if pool <= 0:
        return None
    donor_total = sum(donors.values(), Decimal(0))
    recv_total = sum(receivers.values(), Decimal(0))
    transfers = {}
    for t, offer in donors.items():
        give_base = pool * (offer / donor_total)
        books[t].cash -= give_base / rates[t]
        transfers[t] = -give_base
    for t, need in receivers.items():
        take_base = pool * (need / recv_total)
        books[t].cash += take_base / rates[t]
        transfers[t] = take_base
    return {"as_of": stamp, "transferred_base": pool, "transfers": transfers}


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
    max_single = number(spec["max_single_weight"]) if "max_single_weight" in spec else None
    if max_single is not None:
        if not 0 < max_single <= 1:
            raise ValueError("max_single_weight must be in (0, 1]")
        if any(w > max_single for w in weights):
            raise ValueError("sleeve weight exceeds max_single_weight")
    rebalance = spec.get("rebalance")
    if rebalance is not None and rebalance not in {"daily", "weekly", "monthly"}:
        raise ValueError("rebalance must be daily, weekly, or monthly")
    max_mark_age = number(spec["max_mark_age_hours"]) * 3600
    if max_mark_age <= 0:
        raise ValueError("maximum mark age must be positive")
    costs = Assumptions(**{k: number(v) for k, v in spec.get("assumptions", {}).items()})
    books, currencies, target_weights = {}, {}, {}
    events = defaultdict(list)
    for row, weight in zip(spec["sleeves"], weights):
        dataset = load_dataset(_resolve_root(row["dataset"], spec_path))
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
        target_weights[ticker] = weight
        for bar in dataset.bars:
            if start < bar.close_time and start < bar.available_at <= end:
                events[bar.available_at].append((ticker, bar, dataset.documents, features))

    bench_book, bench_ccy = None, None
    bench_events = defaultdict(list)
    if "benchmark" in spec:
        bench_ds = load_dataset(_resolve_root(spec["benchmark"], spec_path))
        bench_ccy = bench_ds.meta["currency"]
        bench_cash = capital / fx.rate(bench_ccy, start)
        bench_book = PaperEngine(bench_ds.meta, "buy_hold", replace(costs, cash=bench_cash))
        for bar in bench_ds.bars:
            if start < bar.close_time and start < bar.available_at <= end:
                bench_events[bar.available_at].append((bar, bench_ds.documents))

    times = {start, end, *events, *bench_events}
    for currency in set(currencies.values()) | ({bench_ccy} if bench_ccy else set()):
        times.update(t for t in fx.times.get(currency, []) if start < t <= end)

    ledger, bench_navs, rebalances = [], [], []
    peak = capital
    last_bucket = _rebalance_bucket(start, rebalance) if rebalance else None
    final_weights = {}
    for stamp in sorted(times):
        for ticker, bar, docs, features in events[stamp]:
            books[ticker].observe(bar, docs, features)
        if bench_book is not None:
            for bar, docs in bench_events[stamp]:
                bench_book.observe(bar, docs)
        if rebalance and stamp > start:
            bucket = _rebalance_bucket(stamp, rebalance)
            if bucket != last_bucket:
                event = _rebalance_cash(books, currencies, target_weights, fx, stamp)
                if event:
                    rebalances.append(event)
                last_bucket = bucket
        nav, gross, net = Decimal(0), Decimal(0), Decimal(0)
        sleeve_base = {}
        for ticker, book in books.items():
            rate = fx.rate(currencies[ticker], stamp)
            if book.units and (stamp - book.last_mark.close_time).total_seconds() > max_mark_age:
                raise ValueError(f"stale position mark for {ticker} at {stamp}")
            mark = book.units * book.last_mark.close if book.last_mark else Decimal(0)
            local_nav = _book_local_nav(book)
            base_nav = local_nav * rate
            sleeve_base[ticker] = base_nav
            nav += base_nav
            net += mark * rate
            gross += abs(mark) * rate
        if nav > 0:
            final_weights = {t: float(v / nav) for t, v in sleeve_base.items()}
        if bench_book is not None:
            b_rate = fx.rate(bench_ccy, stamp)
            bench_navs.append(_book_local_nav(bench_book) * b_rate)
        peak = max(peak, nav)
        ledger.append({"as_of": stamp, "nav": nav, "drawdown": drawdown(nav, peak),
                       "gross_exposure": gross / nav if nav > 0 else None,
                       "net_exposure": net / nav if nav > 0 else None})

    nav_series = [row["nav"] for row in ledger]
    sleeve_reports = {k: {"currency": currencies[k], **b.report()} for k, b in books.items()}
    return {
        "base_currency": fx.base,
        "initial_nav": capital,
        "net_return": ledger[-1]["nav"] / capital - 1,
        "maximum_drawdown": max(r["drawdown"] for r in ledger),
        "sharpe_ratio": sharpe_ratio(nav_series),
        "sortino_ratio": sortino_ratio(nav_series),
        "concentration": concentration_metrics(final_weights),
        "liquidity": {k: rep["days_to_exit"] for k, rep in sleeve_reports.items()},
        "rebalances": rebalances,
        "benchmark_attribution": benchmark_attribution(nav_series, bench_navs) if bench_book is not None else None,
        "ledger": ledger,
        "sleeves": sleeve_reports,
    }
