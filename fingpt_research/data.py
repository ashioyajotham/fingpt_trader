"""Market observations. All adapters return closed, raw bars for one listing."""

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"


def instant(value):
    stamp = datetime.fromisoformat(value)
    if stamp.tzinfo is None:
        raise ValueError(f"timestamp requires a UTC offset: {value}")
    return stamp.astimezone(UTC)


def number(value):
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"invalid decimal: {value}") from exc
    if not result.is_finite():
        raise ValueError(f"non-finite number: {value}")
    return result


@dataclass(frozen=True)
class Bar:
    session: str
    close_time: datetime
    available_at: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    status: str = "trading"

    def __post_init__(self):
        if self.available_at < self.close_time:
            raise ValueError("closed bar available before close")
        if not 0 < self.low <= min(self.open, self.close) <= max(self.open, self.close) <= self.high:
            raise ValueError(f"invalid OHLC for {self.session}")
        if self.volume < 0 or self.status not in {"trading", "suspended"}:
            raise ValueError(f"invalid volume or status for {self.session}")


def binance_bars(root, meta):
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    capture = instant(meta["ingested_at"])
    for row in json.loads((root / "klines.json").read_text()):
        start = epoch + timedelta(milliseconds=row[0])
        end = epoch + timedelta(milliseconds=row[6])
        if start.time() != datetime.min.time() or end - start != timedelta(days=1, milliseconds=-1):
            raise ValueError("expected UTC daily spot klines in milliseconds")
        if end <= capture:
            yield Bar(start.date().isoformat(), end, end, *map(number, row[1:6]))


@dataclass
class Dataset:
    meta: dict
    bars: list
    documents: list

    def report(self, cutoff):
        from dataclasses import asdict

        return {"dataset": self.meta, "observations": [
            asdict(bar) for bar in self.bars if cutoff is None or bar.available_at <= cutoff
        ], "documents": [
            doc for doc in self.documents if cutoff is None or doc["available_at"] <= cutoff
        ]}


def load_dataset(root):
    root = Path(root)
    meta = json.loads((root / "manifest.json").read_text())
    for key in ("instrument", "venue", "currency", "source", "calendar_version", "ingested_at"):
        if not meta.get(key):
            raise ValueError(f"missing manifest field: {key}")
    captured = instant(meta["ingested_at"])
    if number(meta["lot_size"]) <= 0 or meta["price_basis"] != "raw":
        raise ValueError("positive lot size and raw prices required")
    if meta["adapter"] != "binance-spot-daily":
        raise ValueError("unsupported adapter")
    bars = list(binance_bars(root, meta))
    if not bars or any(a.session >= b.session or a.available_at >= b.available_at
                       for a, b in zip(bars, bars[1:])):
        raise ValueError("expected unique daily bars in observation order")
    if bars[-1].available_at > captured:
        raise ValueError("observation becomes available after capture")
    return Dataset(meta, bars, [])
