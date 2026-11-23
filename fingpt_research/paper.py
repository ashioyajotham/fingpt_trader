"""Single-listing paper ledger. A strategy observes each close after pending fills."""

from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal

from .data import number
from .risk import drawdown, exposures

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


def approve(signal, bar, nav, units, lot, config):
    """Map a research value to a long-only target and size in whole lots."""
    weight = min(config.max_exposure, max(ZERO, signal.value))
    target = (weight * nav / bar.close // lot) * lot
    return ApprovedIntent(signal.instrument, signal.observed_at, target - units)


def baseline(name):
    """Return a signal producer. Custom callables use the same four arguments."""
    if name not in {"cash", "buy_hold", "price", "price_ml"}:
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
        else:
            weight = ONE if name == "buy_hold" else ZERO
        return Signal(instrument, name, "1", bar.available_at, 1, weight, tuple(evidence))

    return produce


class PaperEngine:
    def __init__(self, meta, policy, config):
        self.producer = baseline(policy) if isinstance(policy, str) else policy
        self.instrument = meta["instrument"]
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
        self.ledger = []
        self.fills = []
        self.cash_events = []
        self.signals = []

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
        if (order is None or order.instrument != self.instrument or order.decided_at >= bar.close_time
                or bar.price_type != "close" or not order.quantity or bar.volume == 0 or bar.status != "trading"):
            return
        cfg = self.config
        buying = order.quantity > 0
        sign = ONE if buying else -ONE
        price = bar.close * (1 + sign * cfg.slippage_bps / BPS)
        limit = min(abs(order.quantity), bar.volume * cfg.participation)
        if buying:
            unit_cost = price * (1 + cfg.fee_bps / BPS)
            nav = self.cash + self.units * bar.close + receivable
            # Fees and impact reduce NAV as the buy fills.
            headroom = cfg.max_exposure * nav - self.units * bar.close
            risk_cap = headroom / (bar.close + cfg.max_exposure * (unit_cost - bar.close))
            limit = min(limit, self.cash / unit_cost, risk_cap)
        else:
            limit = min(limit, self.units)
        quantity = sign * (max(ZERO, limit) // self.lot) * self.lot
        if quantity:
            fee = abs(quantity) * price * cfg.fee_bps / BPS
            self.cash -= quantity * price + fee
            self.units += quantity
            self.fills.append({"session": bar.session, "decided_at": order.decided_at,
                               "filled_at": bar.close_time, "quantity": quantity, "price": price, "fee": fee})
        self.pending = None

    def observe(self, bar, documents=(), features=(), *, trading=True):
        if self.ledger and self.ledger[-1]["session"] >= bar.session:
            raise ValueError("paper sessions must advance exactly once")
        accrued = self.dividends(bar, documents)
        self.fill(bar, accrued)
        position = self.units * bar.close
        nav = self.cash + position + accrued
        gross, net = exposures([position], nav)
        self.peak = max(self.peak, nav)
        dd = drawdown(nav, self.peak)
        self.worst = max(self.worst, dd)
        self.held[bar.session] = self.units
        self.ledger.append({"session": bar.session, "cash": self.cash, "quantity": self.units,
                            "receivable": accrued, "nav": nav, "gross_exposure": gross, "net_exposure": net,
                            "drawdown": dd, "max_drawdown": self.worst})
        if bar.status != "trading" or not bar.volume:
            return
        visible = tuple(f for f in features if f.instrument == self.instrument and f.observed_at <= bar.available_at)
        signal = self.producer(self.instrument, bar, self.previous, visible)
        self.previous = bar
        if trading and signal is not None:
            if (not isinstance(signal, Signal) or signal.instrument != self.instrument or
                    signal.observed_at != bar.available_at or not signal.evidence or not signal.value.is_finite()):
                raise ValueError("policy produced an invalid research signal")
            self.signals.append(signal)
            self.pending = approve(signal, bar, nav, self.units, self.lot, self.config)

    def report(self):
        nav = self.ledger[-1]["nav"] if self.ledger else self.config.cash
        turnover = sum(abs(f["quantity"]) * f["price"] for f in self.fills)
        return {"mode": "paper", "policy": self.policy, "assumptions": asdict(self.config),
                "net_return": nav / self.config.cash - 1, "max_drawdown": self.worst,
                "turnover_over_initial_nav": turnover / self.config.cash, "ledger": self.ledger,
                "fills": self.fills, "dividends": self.cash_events,
                "signals": [asdict(s) for s in self.signals]}
