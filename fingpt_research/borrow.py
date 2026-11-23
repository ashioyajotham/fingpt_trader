"""Point-in-time borrow limits for paper simulation. Rates are decimal ACT/365."""

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from .data import instant, number


@dataclass(frozen=True)
class BorrowTerms:
    starts: datetime
    expires: datetime
    quantity: Decimal
    annual_rate: Decimal
    buy_in_bps: Decimal
    source: str


class BorrowSchedule:
    def __init__(self, instrument, records):
        self.instrument = instrument
        self.terms = []
        for r in records:
            starts = max(instant(r["effective_at"]), instant(r["available_at"]))
            term = BorrowTerms(starts, instant(r["expires_at"]), number(r["max_units"]),
                               number(r["annual_rate"]), number(r["buy_in_bps"]), r["source"])
            if (term.expires <= starts or term.quantity < 0 or term.annual_rate < 0
                    or not 0 <= term.buy_in_bps < 10000 or not term.source):
                raise ValueError("invalid borrow availability, fee, source or expiry")
            self.terms.append(term)
        self.terms.sort(key=lambda t: t.starts)
        self.starts = [t.starts for t in self.terms]
        if len(set(self.starts)) != len(self.starts):
            raise ValueError("simultaneous borrow revisions require consolidation")

    def at_close(self, stamp):
        # A message arriving exactly at the close cannot trigger a fill at that close.
        idx = bisect_left(self.starts, stamp) - 1
        return self.terms[idx] if idx >= 0 else None

    def charge(self, start, end, notional):
        """Previous closing mark with rate changes integrated over the holding period."""
        if not notional or start == end:
            return Decimal(0)
        cuts = self.starts[bisect_right(self.starts, start):bisect_left(self.starts, end)]
        charge = Decimal(0)
        for stop in [*cuts, end]:
            idx = bisect_right(self.starts, start) - 1
            if idx < 0:
                raise ValueError("unpriced outstanding loan")
            seconds = Decimal(str((stop - start).total_seconds()))
            charge += notional * self.terms[idx].annual_rate * seconds / (365 * 86400)
            start = stop
        return charge
