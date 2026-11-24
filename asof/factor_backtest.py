"""Point-in-time cross-sectional factor portfolio backtester (zero look-ahead, broker-free)."""

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from math import isfinite

from .data import number
from .factors import FACTOR_NAMES, compute_factor_panel
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
BPS = Decimal("10000")
REBALANCE_SCHEDULES = ("daily", "weekly", "monthly")
DIRECTIONS = ("top", "bottom")
WEIGHTINGS = ("equal", "rank")


@dataclass(frozen=True)
class FactorBacktestConfig:
    factor: str = "momentum_20d"
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
    start: str | None = None
    end: str | None = None

    def __post_init__(self):
        if self.factor not in FACTOR_NAMES:
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


def _target_weights(ranked_symbols, weighting, max_weight):
    """Allocate capped long-only target weights across selected symbols (sum <= 1)."""
    n = len(ranked_symbols)
    if n == 0:
        return {}
    if weighting == "rank":
        raw = {sym: Decimal(n - idx) for idx, sym in enumerate(ranked_symbols)}
    else:
        raw = {sym: ONE for sym in ranked_symbols}
    total = sum(raw.values(), ZERO)
    weights = {sym: val / total for sym, val in raw.items()}
    # Iteratively redistribute weight above max_weight among uncapped names.
    capped = {}
    remaining = dict(weights)
    while remaining:
        over = {s: w for s, w in remaining.items() if w > max_weight}
        if not over:
            capped.update(remaining)
            break
        for s in over:
            capped[s] = max_weight
            del remaining[s]
        left = ONE - sum(capped.values(), ZERO)
        if left <= 0 or not remaining:
            for s in remaining:
                capped[s] = ZERO
            break
        rem_sum = sum(remaining.values(), ZERO)
        remaining = {s: (w / rem_sum) * left for s, w in remaining.items()}
    return {s: w for s, w in capped.items() if w > 0}


def _simulate_weights(universe, sessions, bars_by_session, target_schedule, cfg):
    """Simulate cash-first next-session execution with fees, slippage, lots, and volume participation caps."""
    lots = {sym: number(ds.meta["lot_size"]) for sym, ds in universe.items()}
    cash = cfg.cash
    units = {sym: ZERO for sym in universe}
    last_close = {}
    recent_vols = {sym: [] for sym in universe}
    pending_targets = None
    peak = cfg.cash
    worst_dd = ZERO
    total_fees = ZERO
    turnover = ZERO
    trades = []
    ledger = []

    for session in sessions:
        today_bars = bars_by_session[session]
        for sym, bar in today_bars.items():
            if bar.close > 0:
                last_close[sym] = bar.close
            recent_vols[sym].append(bar.volume)

        # 1. Execute pending targets decided at a prior session's close (zero look-ahead).
        if pending_targets is not None:
            pre_nav = cash + sum(units[s] * last_close[s] for s in units if units[s] and s in last_close)
            sells, buys = [], []
            all_syms = sorted(set(pending_targets) | {s for s, q in units.items() if q > 0})
            for sym in all_syms:
                bar = today_bars.get(sym)
                if bar is None or bar.status != "trading" or bar.volume <= 0 or bar.close <= 0:
                    continue
                lot = lots[sym]
                w = pending_targets.get(sym, ZERO)
                desired = ((w * pre_nav / bar.close) // lot) * lot if pre_nav > 0 and w > 0 else ZERO
                delta = desired - units[sym]
                if delta < 0:
                    sells.append((sym, bar, lot, delta))
                elif delta > 0:
                    buys.append((sym, bar, lot, delta))

            # Execute sells first so freed cash is available for buys on the same session.
            for sym, bar, lot, delta in sells:
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

            for sym, bar, lot, delta in buys:
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
            pending_targets = None

        # 2. Mark portfolio to market at today's close.
        invested = sum(units[s] * last_close[s] for s in units if units[s] and s in last_close)
        nav = cash + invested
        peak = max(peak, nav)
        dd = drawdown(nav, peak) if peak > 0 else ZERO
        worst_dd = max(worst_dd, dd)
        pos_count = sum(1 for q in units.values() if q > 0)
        ledger.append({
            "session": session,
            "nav": nav,
            "cash": cash,
            "invested": invested,
            "gross_exposure": (invested / nav) if nav > 0 else None,
            "drawdown": dd,
            "positions_count": pos_count,
        })

        # 3. Queue new target weights decided at today's close for next trading session.
        if session in target_schedule:
            pending_targets = target_schedule[session]

    final_nav = ledger[-1]["nav"] if ledger else cfg.cash
    holdings = []
    final_weights = {}
    for sym in sorted(units):
        qty = units[sym]
        if qty <= 0 or sym not in last_close:
            continue
        mv = qty * last_close[sym]
        w = float(mv / final_nav) if final_nav > 0 else 0.0
        final_weights[sym] = w
        holdings.append({
            "symbol": sym,
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
        "turnover": turnover,
        "trades": trades,
        "ledger": ledger,
        "holdings": holdings,
        "final_weights": final_weights,
    }


def backtest_factor(universe, cfg=None, *, catalog_dir=None):
    """Run a point-in-time cross-sectional factor portfolio backtest against an equal-weight universe benchmark."""
    cfg = cfg or FactorBacktestConfig()
    panel, _ = compute_factor_panel(universe, catalog_dir=catalog_dir, min_adv=cfg.min_adv)

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
    strat_schedule = {}
    bench_schedule = {}
    rebalance_log = []
    last_bucket = None

    for session in sessions[:-1]:
        bucket = _bucket(session, cfg.rebalance)
        if bucket == last_bucket:
            continue
        cross = panel.get(session, {})
        scored = [
            (sym, fdict[cfg.factor])
            for sym, fdict in cross.items()
            if cfg.factor in fdict and fdict[cfg.factor] is not None and isfinite(fdict[cfg.factor])
        ]
        if not scored:
            continue
        last_bucket = bucket
        scored.sort(key=lambda item: (-item[1] if reverse else item[1], item[0]))
        selected = [sym for sym, _ in scored[:cfg.top_n]]
        weights = _target_weights(selected, cfg.weighting, cfg.max_weight)
        strat_schedule[session] = weights

        # Equal-weight benchmark across all liquidity-eligible names on the same rebalance session.
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
            f"no rebalance sessions had valid {cfg.factor!r} observations "
            f"(min_adv={cfg.min_adv}, sessions={len(sessions)})"
        )

    strat = _simulate_weights(universe, sessions, bars_by_session, strat_schedule, cfg)
    bench = _simulate_weights(universe, sessions, bars_by_session, bench_schedule, cfg)

    strat_navs = [cfg.cash, *(r["nav"] for r in strat["ledger"])]
    bench_navs = [cfg.cash, *(r["nav"] for r in bench["ledger"])]

    net_return = strat["final_nav"] / cfg.cash - ONE
    start_day = date.fromisoformat(sessions[0])
    end_day = date.fromisoformat(sessions[-1])
    years = (end_day - start_day).days / 365.25
    cagr = (float(strat["final_nav"] / cfg.cash) ** (1.0 / years) - 1.0) if years >= 0.25 and strat["final_nav"] > 0 else None

    return {
        "mode": "factor_backtest",
        "return_basis": "price_return_net_of_costs",
        "price_basis": "raw_close_or_vwap",
        "factor": cfg.factor,
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
        "concentration": concentration_metrics(strat["final_weights"]),
        "benchmark_attribution": benchmark_attribution(strat_navs, bench_navs),
        "final_holdings": strat["holdings"],
        "rebalances": rebalance_log,
        "ledger": [
            {**row, "benchmark_nav": b_row["nav"]}
            for row, b_row in zip(strat["ledger"], bench["ledger"])
        ],
    }
