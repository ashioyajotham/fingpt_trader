"""Point-in-time cross-sectional factor computation, Spearman Rank-IC, and NSE universe snapshots."""
from bisect import bisect_right
from collections import defaultdict
from datetime import date, datetime, time, timedelta
from math import isfinite, sqrt
from pathlib import Path
import json

from .data import load_dataset
from .dividends import NAIROBI, load_corpus
from .financials import load_financials
from .risk import days_to_exit

FACTOR_NAMES = ("momentum_20d", "momentum_60d", "liquidity_adv20", "dividend_yield", "earnings_yield")


def load_universe(root):
    """Load all validated datasets from a universe directory (via manifest.json or subdirectories)."""
    base = Path(root)
    index_path = base / "manifest.json"
    datasets = {}
    if index_path.exists():
        meta = json.loads(index_path.read_text(encoding="utf-8"))
        for sym, info in sorted(meta.get("tickers", {}).items()):
            datasets[sym.upper()] = load_dataset(base / info["path"])
    else:
        for child in sorted(base.iterdir()):
            if child.is_dir() and (child / "manifest.json").exists():
                datasets[child.name.upper()] = load_dataset(child)
    if not datasets:
        raise ValueError(f"no valid datasets found in universe directory: {base}")
    return datasets


def _average_ranks(values):
    """Assign 1-based average ranks with exact tie handling."""
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg
        i = j + 1
    return ranks


def spearman_rank_ic(xs, ys):
    """Exact Spearman rank correlation coefficient between two equal-length sequences."""
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    rx = _average_ranks(xs)
    ry = _average_ranks(ys)
    n = len(rx)
    mx = sum(rx) / n
    my = sum(ry) / n
    dx = [v - mx for v in rx]
    dy = [v - my for v in ry]
    vx = sum(v * v for v in dx)
    vy = sum(v * v for v in dy)
    if vx <= 0 or vy <= 0:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / sqrt(vx * vy)


def _index_dividends(catalog_dir):
    """Index dividend disclosures by symbol in Nairobi disclosure-clock order (zero look-ahead)."""
    if not catalog_dir:
        return {}
    root = Path(catalog_dir)
    if not all((root / f"{name}.json").exists() for name in ("notices", "membership", "reference")):
        return {}
    docs, groups, labels, _ = load_corpus(root, reviewed=True)
    by_sym = defaultdict(list)
    for doc in sorted(docs, key=lambda d: (d["notice_date"], d["id"])):
        key = doc["id"]
        event = groups[key]["event"]
        if not event:
            continue
        nday = date.fromisoformat(doc["notice_date"])
        available = datetime.combine(nday + timedelta(days=1), time.min, NAIROBI)
        by_sym[doc["symbol"].upper()].append((available, nday, event, labels[key]))
    return by_sym


def _trailing_dividend_per_share(events_list, as_of_time, session_day):
    """Sum latest approved/paid per-share dividends disclosed within trailing 365 days as of close_time."""
    if not events_list:
        return None
    state = {}
    for available, nday, event, fields in events_list:
        if available > as_of_time:
            break
        cur = state.setdefault(event, {"notice_date": nday, "fields": {}})
        cur["notice_date"] = nday
        cur["fields"].update(fields)
    window_start = session_day - timedelta(days=365)
    total = 0.0
    found = False
    for info in state.values():
        f = info["fields"]
        status = (f.get("status") or {}).get("value")
        amount = (f.get("amount") or {}).get("value")
        if status in {"approved", "paid"} and amount and window_start < info["notice_date"] <= session_day:
            total += float(amount)
            found = True
    return total if found else None


def _index_eps(catalog_dir):
    """Index point-in-time EPS disclosures by symbol in Nairobi disclosure-clock order."""
    if not catalog_dir:
        return {}
    root = Path(catalog_dir)
    if not (root / "financials.json").exists():
        return {}
    rows, _ = load_financials(root)
    by_sym = defaultdict(list)
    for rec in sorted(rows, key=lambda r: (r["notice_date"], r["id"])):
        eps_item = rec.get("metrics", {}).get("eps")
        if not eps_item:
            continue
        nday = date.fromisoformat(rec["notice_date"])
        available = datetime.combine(nday + timedelta(days=1), time.min, NAIROBI)
        by_sym[rec["symbol"].upper()].append((available, float(eps_item["value"])))
    indexed = {}
    for sym, items in by_sym.items():
        items.sort(key=lambda pair: pair[0])
        indexed[sym] = ([p[0] for p in items], [p[1] for p in items])
    return indexed


def _latest_eps(eps_entry, as_of_time):
    if not eps_entry:
        return None
    times, values = eps_entry
    idx = bisect_right(times, as_of_time) - 1
    return values[idx] if idx >= 0 else None


def compute_factor_panel(universe, *, catalog_dir=None, min_adv=0.0):
    """Compute strictly point-in-time cross-sectional factors keyed by session -> symbol -> factor dict."""
    div_idx = _index_dividends(catalog_dir)
    eps_idx = _index_eps(catalog_dir)
    panel = defaultdict(dict)
    closes_by_sym = {}

    for sym, dataset in sorted(universe.items()):
        bars = dataset.bars
        closes = [float(b.close) for b in bars]
        turnovers = [float(b.close * b.volume) for b in bars]
        closes_by_sym[sym] = {b.session: (i, closes[i]) for i, b in enumerate(bars)}
        for i, bar in enumerate(bars):
            adv_window = turnovers[max(0, i - 19):i + 1]
            adv20 = sum(adv_window) / len(adv_window) if adv_window else 0.0
            if min_adv > 0 and adv20 < min_adv:
                continue
            row = {"liquidity_adv20": adv20}
            if i >= 20 and closes[i - 20] > 0:
                row["momentum_20d"] = (closes[i] / closes[i - 20]) - 1.0
            if i >= 60 and closes[i - 60] > 0:
                row["momentum_60d"] = (closes[i] / closes[i - 60]) - 1.0
            if closes[i] > 0:
                dps = _trailing_dividend_per_share(div_idx.get(sym), bar.close_time, date.fromisoformat(bar.session))
                if dps is not None:
                    row["dividend_yield"] = dps / closes[i]
                eps = _latest_eps(eps_idx.get(sym), bar.close_time)
                if eps is not None:
                    row["earnings_yield"] = eps / closes[i]
            panel[bar.session][sym] = row
    return panel, closes_by_sym


def parse_factor_blend(spec):
    """Parse a composite factor blend spec ('dividend_yield:0.5,momentum_20d:-0.5' or dict)."""
    if spec is None:
        return None
    if isinstance(spec, str):
        raw = spec.strip()
        if not raw:
            return None
        parsed = {}
        for part in raw.split(","):
            piece = part.strip()
            if not piece:
                continue
            if ":" not in piece:
                raise ValueError(f"invalid factor blend item {piece!r}; expected 'factor:weight'")
            name, weight_str = piece.split(":", 1)
            name = name.strip()
            weight = float(weight_str.strip())
            parsed[name] = parsed.get(name, 0.0) + weight
    elif isinstance(spec, dict):
        parsed = {str(k).strip(): float(v) for k, v in spec.items()}
    else:
        raise ValueError("factor blend must be a mapping or 'factor:weight,...' string")
    if not parsed:
        return None
    for name, weight in parsed.items():
        if name not in FACTOR_NAMES:
            raise ValueError(f"unknown factor {name!r} in blend; expected one of {', '.join(FACTOR_NAMES)}")
        if not isfinite(weight) or weight == 0:
            raise ValueError(f"blend weight for {name!r} must be finite and non-zero")
    return parsed


def compute_composite_scores(cross, blend):
    """Compute cross-sectional percentile-rank weighted composite scores for tickers with all blend factors."""
    if not blend or not cross:
        return {}
    syms = sorted(
        sym for sym, fdict in cross.items()
        if all(f in fdict and fdict[f] is not None and isfinite(fdict[f]) for f in blend)
    )
    if not syms:
        return {}
    n = len(syms)
    total_abs = sum(abs(w) for w in blend.values())
    scores = {sym: 0.0 for sym in syms}
    for fname, weight in blend.items():
        vals = [cross[sym][fname] for sym in syms]
        ranks = _average_ranks(vals)
        for sym, rk in zip(syms, ranks):
            pct = (rk - 0.5) / n
            scores[sym] += (weight / total_abs) * pct
    return scores


def evaluate_factors(universe, *, catalog_dir=None, horizons=(1, 5, 20), min_adv=0.0, blend=None):
    """Evaluate cross-sectional factors (and optional composite blend) using Spearman Rank-IC and spreads."""
    parsed_blend = parse_factor_blend(blend)
    panel, closes_by_sym = compute_factor_panel(universe, catalog_dir=catalog_dir, min_adv=min_adv)
    bar_closes = {sym: [float(b.close) for b in ds.bars] for sym, ds in universe.items()}
    sessions = sorted(panel)

    factor_list = list(FACTOR_NAMES)
    if parsed_blend:
        for session in sessions:
            comp = compute_composite_scores(panel[session], parsed_blend)
            for sym, val in comp.items():
                panel[session][sym]["composite"] = val
        factor_list.append("composite")

    results = {}
    for factor in factor_list:
        by_horizon = {}
        for h in horizons:
            ics, spreads, sizes = [], [], []
            for session in sessions:
                pairs = []
                for sym, fdict in panel[session].items():
                    fval = fdict.get(factor)
                    if fval is None or not isfinite(fval):
                        continue
                    idx, c0 = closes_by_sym[sym][session]
                    series = bar_closes[sym]
                    if idx + h >= len(series) or c0 <= 0:
                        continue
                    fwd_ret = (series[idx + h] / c0) - 1.0
                    if isfinite(fwd_ret):
                        pairs.append((fval, fwd_ret))
                if len(pairs) < 3:
                    continue
                xs = [p[0] for p in pairs]
                ys = [p[1] for p in pairs]
                ic = spearman_rank_ic(xs, ys)
                if ic is None:
                    continue
                ics.append(ic)
                sizes.append(len(pairs))
                ordered = sorted(pairs, key=lambda p: p[0])
                k = max(1, len(ordered) // 3)
                bottom = sum(p[1] for p in ordered[:k]) / k
                top = sum(p[1] for p in ordered[-k:]) / k
                spreads.append(top - bottom)
            if not ics:
                by_horizon[str(h)] = {
                    "evaluated_sessions": 0, "avg_universe_size": 0.0,
                    "mean_ic": None, "ic_std": None, "ic_ir": None, "hit_rate": None, "mean_spread": None,
                }
            else:
                n = len(ics)
                mean_ic = sum(ics) / n
                var_ic = sum((v - mean_ic) ** 2 for v in ics) / (n - 1) if n > 1 else 0.0
                std_ic = sqrt(var_ic) if var_ic > 0 else None
                by_horizon[str(h)] = {
                    "evaluated_sessions": n,
                    "avg_universe_size": sum(sizes) / n,
                    "mean_ic": mean_ic,
                    "ic_std": std_ic,
                    "ic_ir": (mean_ic / std_ic) if std_ic else None,
                    "hit_rate": sum(1 for v in ics if v > 0) / n,
                    "mean_spread": sum(spreads) / len(spreads),
                }
        results[factor] = by_horizon
    out = {
        "tickers": sorted(universe),
        "sessions": len(sessions),
        "horizons": list(horizons),
        "min_adv_kes": float(min_adv),
        "factors": results,
    }
    if parsed_blend:
        out["blend"] = parsed_blend
    return out


def universe_snapshot(universe, *, catalog_dir=None, stale_days=5, position_kes=100000.0):
    """Build a multi-ticker NSE Market & Liquidity Board summary as of each listing's latest session."""
    div_idx = _index_dividends(catalog_dir)
    eps_idx = _index_eps(catalog_dir)
    latest_overall = max(date.fromisoformat(ds.bars[-1].session) for ds in universe.values())
    rows = []
    for sym, ds in sorted(universe.items()):
        bars = ds.bars
        last = bars[-1]
        close = float(last.close)
        tail = bars[-20:]
        vols = [float(b.volume) for b in tail]
        adv_shares = sum(vols) / len(vols) if vols else 0.0
        adv_kes = sum(float(b.close * b.volume) for b in tail) / len(tail) if tail else 0.0
        qty_for_pos = position_kes / close if close > 0 else 0.0
        dte = days_to_exit(qty_for_pos, vols, 0.01) if qty_for_pos > 0 else None
        last_day = date.fromisoformat(last.session)
        stale = (latest_overall - last_day).days > stale_days or float(last.volume) <= 0
        dps = _trailing_dividend_per_share(div_idx.get(sym), last.close_time, last_day)
        eps = _latest_eps(eps_idx.get(sym), last.close_time)
        rows.append({
            "symbol": sym,
            "instrument": ds.meta["instrument"],
            "issuer": ds.meta["issuer"],
            "sector": ds.meta.get("sector") or "Unclassified",
            "currency": ds.meta["currency"],
            "sessions": len(bars),
            "latest_session": last.session,
            "close": format(last.close, "f"),
            "price_type": last.price_type,
            "volume": format(last.volume, "f"),
            "adv20_shares": adv_shares,
            "adv20_kes": adv_kes,
            "days_to_exit_100k": dte,
            "stale": stale,
            "dividend_yield": (dps / close) if (dps is not None and close > 0) else None,
            "earnings_yield": (eps / close) if (eps is not None and close > 0) else None,
        })
    return {"as_of_session": latest_overall.isoformat(), "tickers_count": len(rows), "rows": rows}
