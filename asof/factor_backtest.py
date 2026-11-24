"""Point-in-time cross-sectional factor portfolio backtester (zero look-ahead, total-return, broker-free)."""

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from math import isfinite
from pathlib import Path

from .data import number
from .dividends import NAIROBI, load_corpus
from .factors import FACTOR_NAMES, compute_composite_scores, compute_factor_panel, parse_factor_blend
from .risk import (
    benchmark_attribution,
    concentration_metrics,
    days_to_exit,
    drawdown,
    sharpe_ratio,
    sortino_ratio,
)

ZERO = Decimal("0")
ONE = Decimal("1")
HUNDRED = Decimal("100")
BPS = Decimal("10000")
REBALANCE_SCHEDULES = ("daily", "weekly", "monthly")
DIRECTIONS = ("top", "bottom")
WEIGHTINGS = ("equal", "rank")


@dataclass(frozen=True)
class FactorBacktestConfig:
    factor: str = "momentum_20d"
    factor_blend: dict | str | None = None
    direction: str = "top"
    top_n: int = 5
    weighting: str = "equal"
    rebalance: str = "monthly"
    min_adv: float = 0.0
    cash: Decimal = Decimal("1000000")
    fee_bps: Decimal = Decimal("10")
    slippage_bps: Decimal = Decimal("5")
    participation: Decimal = Decimal("0.05")
    max_weight: Decimal = Decimal("0.35")
    max_sector_weight: Decimal | None = None
    include_dividends: bool = True
    wht_pct: Decimal = Decimal("5")
    start: str | None = None
    end: str | None = None

    def __post_init__(self):
        blend = parse_factor_blend(self.factor_blend)
        object.__setattr__(self, "factor_blend", blend)
        if blend is None and self.factor not in FACTOR_NAMES:
            raise ValueError(f"unknown factor {self.factor!r}; expected one of {', '.join(FACTOR_NAMES)}")
        if self.direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {', '.join(DIRECTIONS)}")
        if self.weighting not in WEIGHTINGS:
            raise ValueError(f"weighting must be one of {', '.join(WEIGHTINGS)}")
        if self.rebalance not in REBALANCE_SCHEDULES:
            raise ValueError(f"rebalance must be one of {', '.join(REBALANCE_SCHEDULES)}")
        if self.top_n <= 0:
            raise ValueError("top_n must be positive")
        if self.min_adv < 0 or not isfinite(self.min_adv):
            raise ValueError("min_adv must be non-negative and finite")
        if (self.cash <= 0 or not 0 < self.participation <= 1 or not 0 < self.max_weight <= 1
                or not 0 <= self.fee_bps < BPS or not 0 <= self.slippage_bps < BPS):
            raise ValueError("invalid initial capital, cost, participation, or max_weight")
        if self.max_sector_weight is not None and not 0 < self.max_sector_weight <= 1:
            raise ValueError("max_sector_weight must be in (0, 1]")
        if not 0 <= self.wht_pct < HUNDRED:
            raise ValueError("wht_pct must be in [0, 100)")
        if self.start:
            date.fromisoformat(self.start)
        if self.end:
            date.fromisoformat(self.end)
        if self.start and self.end and self.start > self.end:
            raise ValueError("start session must not exceed end session")


def _bucket(session, schedule):
    day = date.fromisoformat(session)
    if schedule == "daily":
        return session
    if schedule == "weekly":
        iso = day.isocalendar()
        return f"{iso.year}-W{iso.week:02d}"
    return f"{day.year}-{day.month:02d}"


def _select_symbols(ranked_symbols, top_n, max_weight, *, sectors=None, max_sector_weight=None):
    """Select up to top_n symbols in rank order while respecting per-sector slot headroom."""
    if not ranked_symbols:
        return []
    if max_sector_weight is None or max_sector_weight >= ONE or not sectors:
        return list(ranked_symbols[:top_n])
    slot = min(max_weight, ONE / Decimal(top_n))
    selected = []
    sec_counts = defaultdict(int)
    deferred = []
    for sym in ranked_symbols:
        if len(selected) >= top_n:
            break
        sec = sectors.get(sym, "Unclassified")
        count = sec_counts[sec]
        if count == 0 or Decimal(count + 1) * slot <= max_sector_weight:
            selected.append(sym)
            sec_counts[sec] = count + 1
        else:
            deferred.append(sym)
    # If universe lacks enough non-capped sectors to fill top_n, backfill safely (weights still hard-capped).
    for sym in deferred:
        if len(selected) >= top_n:
            break
        selected.append(sym)
    return selected


def _target_weights(ranked_symbols, weighting, max_weight, *, sectors=None, max_sector_weight=None):
    """Allocate long-only target weights enforcing both per-symbol max_weight and per-sector max_sector_weight."""
    n = len(ranked_symbols)
    if n == 0:
        return {}
    if weighting == "rank":
        raw = {sym: Decimal(n - idx) for idx, sym in enumerate(ranked_symbols)}
    else:
        raw = {sym: ONE for sym in ranked_symbols}
    total = sum(raw.values(), ZERO)
    weights = {sym: val / total for sym, val in raw.items()}
    locked = set()

    for _ in range(n * 3 + 4):
        changed = False
        # 1. Enforce per-symbol cap.
        for sym in list(weights):
            if weights[sym] > max_weight:
                weights[sym] = max_weight
                locked.add(sym)
                changed = True

        # 2. Enforce per-sector cap.
        if max_sector_weight is not None and sectors:
            by_sec = defaultdict(list)
            for sym in weights:
                by_sec[sectors.get(sym, "Unclassified")].append(sym)
            for sec, syms in by_sec.items():
                sec_sum = sum((weights[s] for s in syms), ZERO)
                if sec_sum > max_sector_weight:
                    scale = max_sector_weight / sec_sum
                    for s in syms:
                        weights[s] = weights[s] * scale
                        locked.add(s)
                    changed = True

        # 3. Redistribute leftover capital to unlocked symbols whose sector still has headroom.
        left = ONE - sum(weights.values(), ZERO)
        if left <= Decimal("1e-12"):
            break
        sec_totals = defaultdict(lambda: ZERO)
        if sectors:
            for s, w in weights.items():
                sec_totals[sectors.get(s, "Unclassified")] += w
        eligible = []
        for s in weights:
            if s in locked:
                continue
            if max_sector_weight is not None and sectors:
                if sec_totals[sectors.get(s, "Unclassified")] >= max_sector_weight - Decimal("1e-12"):
                    continue
            eligible.append(s)
        if not eligible:
            break
        elig_sum = sum((weights[s] for s in eligible), ZERO)
        if elig_sum <= 0:
            break
        for s in eligible:
            weights[s] += left * (weights[s] / elig_sum)
        changed = True
        if not changed:
            break

    return {s: w for s, w in weights.items() if w > ZERO}


def _load_catalog_dividend_events(catalog_dir):
    """Extract point-in-time approved/paid dividend payment schedules from a local dividend catalog."""
    if not catalog_dir:
        return {}
    root = Path(catalog_dir)
    if not all((root / f"{name}.json").exists() for name in ("notices", "membership", "reference")):
        return {}
    docs, groups, labels, _ = load_corpus(root, reviewed=True)
    state = {}
    first_approved = {}
    for doc in sorted(docs, key=lambda d: (d["notice_date"], d["id"])):
        key = doc["id"]
        event = groups[key]["event"]
        if not event:
            continue
        nday = date.fromisoformat(doc["notice_date"])
        avail_day = (nday + timedelta(days=1)).isoformat()
        cur = state.setdefault(event, {"symbol": doc["symbol"].upper(), "fields": {}, "notice_date": doc["notice_date"]})
        cur["notice_date"] = doc["notice_date"]
        cur["fields"].update(labels[key])
        st = (cur["fields"].get("status") or {}).get("value")
        amt = (cur["fields"].get("amount") or {}).get("value")
        if st in {"approved", "paid"} and amt and event not in first_approved:
            first_approved[event] = (avail_day, doc["notice_date"])

    by_sym = defaultdict(list)
    for event, info in sorted(state.items()):
        if event not in first_approved:
            continue
        f = info["fields"]
        if (f.get("status") or {}).get("value") == "cancelled":
            continue
        raw_amt = (f.get("amount") or {}).get("value")
        if not raw_amt:
            continue
        amount = number(raw_amt)
        if amount <= 0:
            continue
        avail_day, appr_notice = first_approved[event]
        rec_date = (f.get("record_date") or {}).get("value")
        pay_raw = (f.get("payment_date") or {}).get("value")
        if rec_date:
            elig_date = rec_date
        elif pay_raw:
            elig_date = min(appr_notice, pay_raw)
        else:
            elig_date = appr_notice
        pay_date = max(pay_raw, avail_day) if pay_raw else avail_day
        by_sym[info["symbol"]].append({
            "event": event,
            "symbol": info["symbol"],
            "available_day": avail_day,
            "elig_date": elig_date,
            "pay_date": pay_date,
            "amount": amount,
        })
    return by_sym


def _eligible_units_before(held_history, cutoff_day):
    """Return the last recorded share count strictly prior to cutoff_day."""
    eligible = ZERO
    for sess, qty in held_history:
        if sess >= cutoff_day:
            break
        eligible = qty
    return eligible


def _yearly_breakdown(ledger, bench_ledger, div_by_session, initial_cash):
    """Compute calendar-year return, drawdown, Sharpe, benchmark attribution, and net dividends."""
    by_year = defaultdict(list)
    for row, brow in zip(ledger, bench_ledger):
        by_year[row["session"][:4]].append((row, brow))
    out = []
    prev_nav = initial_cash
    prev_bnav = initial_cash
    for year in sorted(by_year):
        pairs = by_year[year]
        s_navs = [prev_nav, *(r["nav"] for r, _ in pairs)]
        b_navs = [prev_bnav, *(b["nav"] for _, b in pairs)]
        peak = prev_nav
        max_dd = ZERO
        for r, _ in pairs:
            peak = max(peak, r["nav"])
            if peak > 0:
                max_dd = max(max_dd, drawdown(r["nav"], peak))
        end_nav = pairs[-1][0]["nav"]
        end_bnav = pairs[-1][1]["nav"]
        s_ret = (end_nav / prev_nav - ONE) if prev_nav > 0 else ZERO
        b_ret = (end_bnav / prev_bnav - ONE) if prev_bnav > 0 else ZERO
        div_net = sum((div_by_session.get(r["session"], ZERO) for r, _ in pairs), ZERO)
        out.append({
            "year": year,
            "sessions": len(pairs),
            "start_nav": prev_nav,
            "end_nav": end_nav,
            "net_return": s_ret,
            "benchmark_return": b_ret,
            "active_return": s_ret - b_ret,
            "max_drawdown": max_dd,
            "sharpe_ratio": sharpe_ratio(s_navs),
            "dividends_net": div_net,
        })
        prev_nav = end_nav
        prev_bnav = end_bnav
    return out


def _simulate_weights(universe, sessions, bars_by_session, target_schedule, cfg, *, catalog_divs=None):
    """Simulate cash-first next-session execution with multi-day order carry-over and PIT cash dividends."""
    lots = {sym: number(ds.meta["lot_size"]) for sym, ds in universe.items()}
    sectors = {sym: ds.meta.get("sector") or "Unclassified" for sym, ds in universe.items()}
    cash = cfg.cash
    units = {sym: ZERO for sym in universe}
    last_close = {}
    recent_vols = {sym: [] for sym in universe}
    held_history = {sym: [] for sym in universe}
    pending_targets = None
    pending_orders = {}
    paid_divs = set()
    applied_splits = set()
    peak = cfg.cash
    worst_dd = ZERO
    total_fees = ZERO
    total_dividends_net = ZERO
    total_wht_paid = ZERO
    turnover = ZERO
    trades = []
    dividend_events = []
    div_by_session = defaultdict(lambda: ZERO)
    ledger = []
    wht_frac = cfg.wht_pct / HUNDRED

    for session in sessions:
        today_bars = bars_by_session[session]
        for sym, bar in today_bars.items():
            if bar.close > 0:
                last_close[sym] = bar.close
            recent_vols[sym].append(bar.volume)

        # 0. Apply dataset-level splits before trading (conserving NAV & scaling pending orders).
        for sym, bar in today_bars.items():
            for doc in universe[sym].documents:
                if "split" not in doc or doc["available_at"] > bar.close_time:
                    continue
                action = doc["split"]
                key = (sym, action["id"])
                if key in applied_splits or action["status"] != "approved" or action["ex_date"] > session:
                    continue
                applied_splits.add(key)
                if units[sym] > 0:
                    num, den = number(action["numerator"]), number(action["denominator"])
                    lot = lots[sym]
                    old_u = units[sym]
                    new_u = ((old_u * num) // (den * lot)) * lot
                    cash_in_lieu = ((old_u * num - new_u * den) * bar.close) / den
                    units[sym] = new_u
                    cash += cash_in_lieu
                    if sym in pending_orders:
                        po = pending_orders[sym]
                        psign = ONE if po >= 0 else -ONE
                        pending_orders[sym] = psign * ((abs(po) * num) // (den * lot)) * lot

        # 1. Accrue and settle point-in-time cash dividends (strictly zero look-ahead).
        receivable = ZERO
        if cfg.include_dividends and catalog_divs:
            for sym, evs in catalog_divs.items():
                for ev in evs:
                    if ev["available_day"] > session or ev["event"] in paid_divs:
                        continue
                    elig = _eligible_units_before(held_history[sym], ev["elig_date"])
                    if elig <= 0:
                        if ev["pay_date"] <= session:
                            paid_divs.add(ev["event"])
                        continue
                    gross = elig * ev["amount"]
                    wht = gross * wht_frac
                    net = gross - wht
                    if ev["pay_date"] <= session:
                        cash += net
                        total_dividends_net += net
                        total_wht_paid += wht
                        div_by_session[session] += net
                        paid_divs.add(ev["event"])
                        dividend_events.append({
                            "session": session, "event": ev["event"], "symbol": sym,
                            "eligible_units": elig, "amount_per_share": ev["amount"],
                            "gross": gross, "wht": wht, "net": net,
                        })
                    else:
                        receivable += net

        # 2. Convert newly queued rebalance targets into concrete lot deltas once, then carry unfilled slices.
        if pending_targets is not None:
            pre_nav = cash + receivable + sum(units[s] * last_close[s] for s in units if units[s] and s in last_close)
            pending_orders = {}
            all_syms = sorted(set(pending_targets) | {s for s, q in units.items() if q > 0})
            for sym in all_syms:
                mark = today_bars[sym].close if sym in today_bars and today_bars[sym].close > 0 else last_close.get(sym)
                if not mark or mark <= 0:
                    continue
                lot = lots[sym]
                w = pending_targets.get(sym, ZERO)
                desired = ((w * pre_nav / mark) // lot) * lot if pre_nav > 0 and w > 0 else ZERO
                delta = desired - units[sym]
                if delta != 0:
                    pending_orders[sym] = delta
            pending_targets = None

        # 3. Work down pending_orders (sells first so freed cash funds buys; unfilled slices carry to t+2, ...).
        if pending_orders:
            sells = [(sym, pending_orders[sym]) for sym in sorted(pending_orders) if pending_orders[sym] < 0]
            buys = [(sym, pending_orders[sym]) for sym in sorted(pending_orders) if pending_orders[sym] > 0]

            for sym, delta in sells:
                bar = today_bars.get(sym)
                if bar is None or bar.status != "trading" or bar.volume <= 0 or bar.close <= 0:
                    continue
                lot = lots[sym]
                price = bar.close * (ONE - cfg.slippage_bps / BPS)
                fee_per_unit = price * cfg.fee_bps / BPS
                cap = min(abs(delta), bar.volume * cfg.participation, units[sym])
                qty = (max(ZERO, cap) // lot) * lot
                if qty > 0:
                    fee = qty * fee_per_unit
                    cash += qty * price - fee
                    units[sym] -= qty
                    total_fees += fee
                    turnover += qty * price
                    trades.append({
                        "session": session, "symbol": sym, "side": "sell",
                        "quantity": qty, "price": price, "fee": fee,
                    })
                    rem = delta + qty
                    if rem < 0 and units[sym] > 0:
                        pending_orders[sym] = rem
                    else:
                        pending_orders.pop(sym, None)
                elif units[sym] <= 0:
                    pending_orders.pop(sym, None)

            for sym, delta in buys:
                bar = today_bars.get(sym)
                if bar is None or bar.status != "trading" or bar.volume <= 0 or bar.close <= 0:
                    continue
                lot = lots[sym]
                price = bar.close * (ONE + cfg.slippage_bps / BPS)
                fee_per_unit = price * cfg.fee_bps / BPS
                unit_cost = price + fee_per_unit
                afford = cash / unit_cost if unit_cost > 0 else ZERO
                cap = min(delta, bar.volume * cfg.participation, afford)
                qty = (max(ZERO, cap) // lot) * lot
                if qty > 0:
                    fee = qty * fee_per_unit
                    cash -= qty * price + fee
                    units[sym] += qty
                    total_fees += fee
                    turnover += qty * price
                    trades.append({
                        "session": session, "symbol": sym, "side": "buy",
                        "quantity": qty, "price": price, "fee": fee,
                    })
                    rem = delta - qty
                    if rem >= lot:
                        pending_orders[sym] = rem
                    else:
                        pending_orders.pop(sym, None)

        # 4. Mark portfolio to market at today's close and record end-of-day share ownership.
        for sym in universe:
            if units[sym] > 0 or (held_history[sym] and held_history[sym][-1][1] != ZERO):
                held_history[sym].append((session, units[sym]))

        invested = sum(units[s] * last_close[s] for s in units if units[s] and s in last_close)
        nav = cash + invested + receivable
        peak = max(peak, nav)
        dd = drawdown(nav, peak) if peak > 0 else ZERO
        worst_dd = max(worst_dd, dd)
        pos_count = sum(1 for q in units.values() if q > 0)
        ledger.append({
            "session": session,
            "nav": nav,
            "cash": cash,
            "invested": invested,
            "receivable": receivable,
            "gross_exposure": (invested / nav) if nav > 0 else None,
            "drawdown": dd,
            "positions_count": pos_count,
        })

        # 5. Queue new target weights decided at today's close for next trading session.
        if session in target_schedule:
            pending_targets = target_schedule[session]

    final_nav = ledger[-1]["nav"] if ledger else cfg.cash
    holdings = []
    final_weights = {}
    sector_exposure = defaultdict(float)
    for sym in sorted(units):
        qty = units[sym]
        if qty <= 0 or sym not in last_close:
            continue
        mv = qty * last_close[sym]
        w = float(mv / final_nav) if final_nav > 0 else 0.0
        sec = sectors[sym]
        final_weights[sym] = w
        sector_exposure[sec] += w
        holdings.append({
            "symbol": sym,
            "sector": sec,
            "units": qty,
            "price": last_close[sym],
            "market_value": mv,
            "weight": w,
            "days_to_exit": days_to_exit(qty, recent_vols[sym][-20:], cfg.participation),
        })
    return {
        "final_nav": final_nav,
        "max_drawdown": worst_dd,
        "total_fees": total_fees,
        "total_dividends_net": total_dividends_net,
        "total_wht_paid": total_wht_paid,
        "dividend_events": dividend_events,
        "div_by_session": div_by_session,
        "turnover": turnover,
        "trades": trades,
        "ledger": ledger,
        "holdings": holdings,
        "final_weights": final_weights,
        "sector_exposure": dict(sorted(sector_exposure.items())),
    }


def backtest_factor(universe, cfg=None, *, catalog_dir=None):
    """Run a point-in-time cross-sectional factor portfolio backtest against an equal-weight universe benchmark."""
    cfg = cfg or FactorBacktestConfig()
    panel, _ = compute_factor_panel(universe, catalog_dir=catalog_dir, min_adv=cfg.min_adv)
    sectors = {sym: ds.meta.get("sector") or "Unclassified" for sym, ds in universe.items()}
    catalog_divs = _load_catalog_dividend_events(catalog_dir) if cfg.include_dividends else {}

    bars_by_session = {}
    for sym, ds in universe.items():
        for bar in ds.bars:
            if cfg.start and bar.session < cfg.start:
                continue
            if cfg.end and bar.session > cfg.end:
                continue
            bars_by_session.setdefault(bar.session, {})[sym] = bar

    sessions = sorted(bars_by_session)
    if len(sessions) < 2:
        raise ValueError("factor backtest requires at least 2 trading sessions in the selected window")

    reverse = cfg.direction == "top"
    active_factor = "composite" if cfg.factor_blend else cfg.factor
    strat_schedule = {}
    bench_schedule = {}
    rebalance_log = []
    last_bucket = None

    for session in sessions[:-1]:
        bucket = _bucket(session, cfg.rebalance)
        if bucket == last_bucket:
            continue
        cross = panel.get(session, {})
        if cfg.factor_blend:
            comp_scores = compute_composite_scores(cross, cfg.factor_blend)
            scored = [(sym, val) for sym, val in comp_scores.items() if isfinite(val)]
        else:
            scored = [
                (sym, fdict[cfg.factor])
                for sym, fdict in cross.items()
                if cfg.factor in fdict and fdict[cfg.factor] is not None and isfinite(fdict[cfg.factor])
            ]
        if not scored:
            continue
        last_bucket = bucket
        scored.sort(key=lambda item: (-item[1] if reverse else item[1], item[0]))
        ranked_syms = [sym for sym, _ in scored]
        selected = _select_symbols(
            ranked_syms, cfg.top_n, cfg.max_weight,
            sectors=sectors, max_sector_weight=cfg.max_sector_weight,
        )
        weights = _target_weights(
            selected, cfg.weighting, cfg.max_weight,
            sectors=sectors, max_sector_weight=cfg.max_sector_weight,
        )
        strat_schedule[session] = weights

        eligible_all = sorted(cross.keys())
        bench_schedule[session] = _target_weights(eligible_all, "equal", ONE)
        rebalance_log.append({
            "signal_session": session,
            "eligible_count": len(scored),
            "selected": selected,
            "target_weights": {s: format(w, "f") for s, w in weights.items()},
        })

    if not strat_schedule:
        raise ValueError(
            f"no rebalance sessions had valid {active_factor!r} observations "
            f"(min_adv={cfg.min_adv}, sessions={len(sessions)})"
        )

    strat = _simulate_weights(universe, sessions, bars_by_session, strat_schedule, cfg, catalog_divs=catalog_divs)
    bench = _simulate_weights(universe, sessions, bars_by_session, bench_schedule, cfg, catalog_divs=catalog_divs)

    strat_navs = [cfg.cash, *(r["nav"] for r in strat["ledger"])]
    bench_navs = [cfg.cash, *(r["nav"] for r in bench["ledger"])]

    net_return = strat["final_nav"] / cfg.cash - ONE
    start_day = date.fromisoformat(sessions[0])
    end_day = date.fromisoformat(sessions[-1])
    years = (end_day - start_day).days / 365.25
    cagr = (float(strat["final_nav"] / cfg.cash) ** (1.0 / years) - 1.0) if years >= 0.25 and strat["final_nav"] > 0 else None
    has_divs = bool(cfg.include_dividends and catalog_divs)

    return {
        "mode": "factor_backtest",
        "return_basis": "total_return_net_of_costs_and_wht" if has_divs else "price_return_net_of_costs",
        "price_basis": "raw_close_or_vwap",
        "factor": active_factor,
        "factor_blend": cfg.factor_blend,
        "direction": cfg.direction,
        "top_n": cfg.top_n,
        "weighting": cfg.weighting,
        "rebalance": cfg.rebalance,
        "min_adv_kes": cfg.min_adv,
        "start_session": sessions[0],
        "end_session": sessions[-1],
        "sessions": len(sessions),
        "rebalance_count": len(rebalance_log),
        "trades_count": len(strat["trades"]),
        "assumptions": {
            "cash": cfg.cash,
            "fee_bps": cfg.fee_bps,
            "slippage_bps": cfg.slippage_bps,
            "participation": cfg.participation,
            "max_weight": cfg.max_weight,
            "max_sector_weight": cfg.max_sector_weight,
            "include_dividends": cfg.include_dividends,
            "wht_pct": cfg.wht_pct,
        },
        "initial_nav": cfg.cash,
        "final_nav": strat["final_nav"],
        "net_return": net_return,
        "cagr": cagr,
        "max_drawdown": strat["max_drawdown"],
        "sharpe_ratio": sharpe_ratio(strat_navs),
        "sortino_ratio": sortino_ratio(strat_navs),
        "turnover_over_initial_nav": strat["turnover"] / cfg.cash,
        "total_fees": strat["total_fees"],
        "total_dividends_net": strat["total_dividends_net"],
        "total_wht_paid": strat["total_wht_paid"],
        "dividend_events_count": len(strat["dividend_events"]),
        "dividend_events": strat["dividend_events"],
        "concentration": concentration_metrics(strat["final_weights"]),
        "sector_exposure": strat["sector_exposure"],
        "benchmark_attribution": benchmark_attribution(strat_navs, bench_navs),
        "yearly": _yearly_breakdown(strat["ledger"], bench["ledger"], strat["div_by_session"], cfg.cash),
        "final_holdings": strat["holdings"],
        "rebalances": rebalance_log,
        "ledger": [
            {**row, "benchmark_nav": b_row["nav"]}
            for row, b_row in zip(strat["ledger"], bench["ledger"])
        ],
    }
