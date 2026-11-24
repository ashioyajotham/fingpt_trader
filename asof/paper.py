"""Single-listing paper ledger. A strategy observes each close after pending fills."""

from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal, ROUND_CEILING

from .data import number
from .risk import days_to_exit, drawdown, exposures, sharpe_ratio

ZERO = Decimal("0")
ONE = Decimal("1")
BPS = Decimal("10000")


@dataclass(frozen=True)
class Signal:
    instrument: str
    producer: str
    version: str
    observed_at: datetime
    horizon_sessions: int
    value: Decimal
    evidence: tuple


@dataclass(frozen=True)
class ApprovedIntent:
    instrument: str
    decided_at: datetime
    quantity: Decimal


@dataclass(frozen=True)
class Assumptions:
    cash: Decimal = Decimal("100000")
    fee_bps: Decimal = Decimal("10")
    slippage_bps: Decimal = Decimal("5")
    participation: Decimal = Decimal("0.01")
    max_exposure: Decimal = ONE

    def __post_init__(self):
        if (self.cash <= 0 or not 0 < self.participation <= 1 or not 0 < self.max_exposure <= 1
                or not 0 <= self.fee_bps < BPS or not 0 <= self.slippage_bps < BPS):
            raise ValueError("invalid initial capital, cost, participation or exposure")


def approve(signal, bar, nav, units, lot, config, allow_short=False):
    """Map a research value to a long-only target and size in whole lots."""
    floor = -config.max_exposure if allow_short else ZERO
    weight = min(config.max_exposure, max(floor, signal.value))
    target = (weight * nav / bar.close // lot) * lot
    return ApprovedIntent(signal.instrument, signal.observed_at, target - units)


def baseline(name):
    """Return a signal producer. Custom callables use the same four arguments."""
    if name not in {"cash", "buy_hold", "price", "price_ml", "short_hold"}:
        raise ValueError("unknown portfolio policy")

    def produce(instrument, bar, previous, features):
        evidence = [bar.session]
        if name in {"price", "price_ml"}:
            bullish = previous is not None and bar.close > previous.close
            if previous:
                evidence.append(previous.session)
            if name == "price_ml":
                # A one-session news feature only influences its first eligible close.
                eligible = [f for f in features if previous is None or previous.available_at < f.observed_at]
                if eligible:
                    bullish = bullish and sum(f.value for f in eligible) >= 0
                    evidence.extend(ref for f in eligible for ref in f.evidence)
            weight = ONE if bullish else ZERO
        elif name == "short_hold":
            weight = -ONE
        else:
            weight = ONE if name == "buy_hold" else ZERO
        return Signal(instrument, name, "1", bar.available_at, 1, weight, tuple(evidence))

    return produce


class PaperEngine:
    def __init__(self, meta, policy, config, borrow=None):
        self.producer = baseline(policy) if isinstance(policy, str) else policy
        self.instrument = meta["instrument"]
        if borrow is not None and borrow.instrument != self.instrument:
            raise ValueError("borrow schedule listing mismatch")
        if policy == "short_hold" and borrow is None:
            raise ValueError("short_hold requires a dated borrow schedule")
        self.borrow = borrow
        self.last_mark = None
        self.borrow_cost = ZERO
        self.lot = number(meta["lot_size"])
        self.policy = policy if isinstance(policy, str) else policy.__name__
        self.config = config
        self.cash = config.cash
        self.peak = config.cash
        self.worst = ZERO
        self.units = ZERO
        self.pending = None
        self.previous = None
        self.held = {}
        self.paid = {}
        self.applied_splits = set()
        self.split_events = []
        self.volumes = []
        self.ledger = []
        self.fills = []
        self.cash_events = []
        self.signals = []

    def apply_splits(self, bar, documents):
        """Apply approved stock splits and bonus issues once on/after ex-date, conserving NAV."""
        actions = {}
        for doc in documents:
            if "split" in doc and doc["available_at"] <= bar.close_time:
                action = doc["split"]
                actions[action["id"]] = action
        for key, action in actions.items():
            if key in self.applied_splits or action["status"] != "approved" or action["ex_date"] > bar.session:
                continue
            self.applied_splits.add(key)
            if not self.units:
                continue
            num, den = number(action["numerator"]), number(action["denominator"])
            ratio = num / den
            old_units = self.units
            sign = ONE if old_units > 0 else -ONE
            new_units = sign * ((abs(old_units) * num) // (den * self.lot)) * self.lot
            cash_in_lieu = ((old_units * num - new_units * den) * bar.close) / den
            self.units = new_units
            self.cash += cash_in_lieu
            if self.pending and self.pending.decided_at < bar.close_time:
                psign = ONE if self.pending.quantity >= 0 else -ONE
                adj_qty = psign * ((abs(self.pending.quantity) * num) // (den * self.lot)) * self.lot
                self.pending = ApprovedIntent(self.pending.instrument, self.pending.decided_at, adj_qty)
            self.split_events.append({
                "action": key, "session": bar.session, "kind": action["kind"], "ratio": ratio,
                "old_units": old_units, "new_units": new_units, "cash_in_lieu": cash_in_lieu,
            })

    def dividends(self, bar, documents):
        actions = {}
        for doc in documents:
            if "dividend" in doc and doc["available_at"] <= bar.close_time:
                action = doc["dividend"]
                actions[action["id"]] = action
        receivable = ZERO
        for key, action in actions.items():
            if action["ex_date"] > bar.session:
                continue
            eligible = ZERO
            for session, quantity in self.held.items():
                if session >= action["ex_date"]:
                    break
                eligible = quantity
            due = eligible * number(action["amount"]) if action["status"] == "approved" else ZERO
            paid = self.paid.get(key, ZERO)
            if action["pay_date"] <= bar.session:
                delta = due - paid
                if delta:
                    self.cash += delta
                    self.cash_events.append({"action": key, "session": bar.session, "amount": delta})
                self.paid[key] = due
            else:
                receivable += due - paid
        return receivable

    def fill(self, bar, receivable):
        order = self.pending
        if bar.price_type != "close" or bar.volume <= 0 or bar.status != "trading":
            return
        term = self.borrow.at_close(bar.close_time) if self.borrow else None
        capacity = term.quantity if term and bar.close_time < term.expires else ZERO
        recalled = max(ZERO, -self.units - capacity)
        if recalled:
            # A fractional locate reduction must cover the next whole trading lot.
            requested = (recalled / self.lot).to_integral_value(rounding=ROUND_CEILING) * self.lot
            decided = term.starts if bar.close_time < term.expires else term.expires
        elif order and order.instrument == self.instrument and order.decided_at < bar.close_time:
            requested, decided = order.quantity, order.decided_at
        else:
            return
        self.pending = None
        if not requested:
            return
        cfg = self.config
        sign = ONE if requested > 0 else -ONE
        premium = term.buy_in_bps if recalled else ZERO
        price = bar.close * (1 + (sign * cfg.slippage_bps + premium) / BPS)
        fee_per_unit = price * cfg.fee_bps / BPS
        limit = min(abs(requested), bar.volume * cfg.participation)
        if self.units * sign < 0:
            # Reducing risk remains possible after a gap breaches a limit.
            limit = min(limit, abs(self.units))
        else:
            nav = self.cash + self.units * bar.close + receivable
            headroom = cfg.max_exposure * nav - abs(self.units) * bar.close
            drag = sign * (price - bar.close) + fee_per_unit
            limit = min(limit, headroom / (bar.close + cfg.max_exposure * drag))
            if sign > 0:
                limit = min(limit, self.cash / (price + fee_per_unit))
            else:
                limit = min(limit, capacity + self.units)
        quantity = sign * (max(ZERO, limit) // self.lot) * self.lot
        if quantity:
            fee = abs(quantity) * fee_per_unit
            self.cash -= quantity * price + fee
            self.units += quantity
            self.fills.append({"session": bar.session, "decided_at": decided, "filled_at": bar.close_time,
                               "quantity": quantity, "price": price, "fee": fee,
                               "reason": "forced_buy_in" if recalled else "policy"})

    def observe(self, bar, documents=(), features=(), *, trading=True):
        if self.last_mark and (bar.session <= self.last_mark.session or bar.close_time <= self.last_mark.close_time
                               or bar.available_at < self.last_mark.available_at):
            raise ValueError("paper sessions must advance exactly once")
        fee = ZERO
        if self.borrow and self.units < 0:
            fee = self.borrow.charge(self.last_mark.close_time, bar.close_time, -self.units * self.last_mark.close)
            self.cash -= fee
            self.borrow_cost += fee
        self.last_mark = bar
        self.volumes.append(bar.volume)
        self.apply_splits(bar, documents)
        accrued = self.dividends(bar, documents)
        self.fill(bar, accrued)
        position = self.units * bar.close
        nav = self.cash + position + accrued
        gross, net = exposures([position], nav) if nav > 0 else (None, None)
        self.peak = max(self.peak, nav)
        dd = drawdown(nav, self.peak)
        self.worst = max(self.worst, dd)
        self.held[bar.session] = self.units
        self.ledger.append({"session": bar.session, "cash": self.cash, "quantity": self.units,
                            "receivable": accrued, "nav": nav, "gross_exposure": gross, "net_exposure": net,
                            "borrow_fee": fee, "insolvent": nav <= 0, "drawdown": dd, "max_drawdown": self.worst})
        if nav <= 0 or bar.status != "trading" or not bar.volume:
            return
        visible = tuple(f for f in features if f.instrument == self.instrument and f.observed_at <= bar.available_at)
        signal = self.producer(self.instrument, bar, self.previous, visible)
        self.previous = bar
        if trading and signal is not None:
            if (not isinstance(signal, Signal) or signal.instrument != self.instrument or
                    signal.observed_at != bar.available_at or not signal.evidence or not signal.value.is_finite()):
                raise ValueError("policy produced an invalid research signal")
            self.signals.append(signal)
            self.pending = approve(signal, bar, nav, self.units, self.lot, self.config, self.borrow is not None)

    def report(self):
        nav = self.ledger[-1]["nav"] if self.ledger else self.config.cash
        turnover = sum(abs(f["quantity"]) * f["price"] for f in self.fills)
        nav_series = [self.config.cash, *(row["nav"] for row in self.ledger)]
        dte = days_to_exit(self.units, self.volumes[-20:], self.config.participation)
        return {"mode": "paper", "policy": self.policy, "assumptions": asdict(self.config),
                "net_return": nav / self.config.cash - 1, "max_drawdown": self.worst,
                "sharpe_ratio": sharpe_ratio(nav_series), "days_to_exit": dte,
                "currency_borrow_fees": self.borrow_cost, "turnover_over_initial_nav": turnover / self.config.cash,
                "ledger": self.ledger, "fills": self.fills, "dividends": self.cash_events,
                "splits": self.split_events, "signals": [asdict(s) for s in self.signals]}
