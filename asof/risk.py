"""Signed position values, drawdown, risk-adjusted returns, concentration, and liquidity metrics."""

from math import isfinite, sqrt


def exposures(values, nav):
    if not isfinite(nav) or nav <= 0 or any(not isfinite(v) for v in values):
        raise ValueError("exposure requires finite position values and positive NAV")
    return sum(abs(v) for v in values) / nav, sum(values) / nav


def drawdown(nav, peak):
    if peak <= 0:
        raise ValueError("drawdown requires a positive high-water mark")
    return max(0, (peak - nav) / peak)


def _period_returns(nav_series):
    values = [float(v) for v in nav_series]
    if len(values) < 3 or any(not isfinite(v) or v <= 0 for v in values):
        return None
    return [(b / a) - 1.0 for a, b in zip(values, values[1:])]


def sharpe_ratio(nav_series, periods_per_year=252):
    """Annualized Sharpe ratio (zero risk-free baseline) from a sequence of positive NAV marks."""
    returns = _period_returns(nav_series)
    if returns is None:
        return None
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    if variance <= 0:
        return None
    return (mean / sqrt(variance)) * sqrt(periods_per_year)


def sortino_ratio(nav_series, periods_per_year=252):
    """Annualized Sortino ratio using downside deviation below zero."""
    returns = _period_returns(nav_series)
    if returns is None:
        return None
    mean = sum(returns) / len(returns)
    downside_sq = [min(0.0, r) ** 2 for r in returns]
    downside_var = sum(downside_sq) / len(returns)
    if downside_var <= 0:
        return None
    return (mean / sqrt(downside_var)) * sqrt(periods_per_year)


def concentration_metrics(weights):
    """Return max absolute weight, Herfindahl-Hirschman Index (HHI), and effective position count."""
    vals = [abs(float(w)) for w in (weights.values() if isinstance(weights, dict) else weights)]
    total = sum(vals)
    if total <= 0:
        return {"max_weight": 0.0, "hhi": 0.0, "effective_positions": 0.0}
    norm = [v / total for v in vals]
    hhi = sum(w * w for w in norm)
    return {
        "max_weight": max(vals),
        "hhi": hhi,
        "effective_positions": (1.0 / hhi) if hhi > 0 else 0.0,
    }


def benchmark_attribution(portfolio_navs, benchmark_navs, periods_per_year=252):
    """Compute active return, tracking error, information ratio, and beta against a benchmark NAV series."""
    if len(portfolio_navs) != len(benchmark_navs) or len(portfolio_navs) < 3:
        return None
    p_vals = [float(v) for v in portfolio_navs]
    b_vals = [float(v) for v in benchmark_navs]
    if any(not isfinite(v) or v <= 0 for v in (*p_vals, *b_vals)):
        return None
    rp = [(b / a) - 1.0 for a, b in zip(p_vals, p_vals[1:])]
    rb = [(b / a) - 1.0 for a, b in zip(b_vals, b_vals[1:])]
    ra = [p - b for p, b in zip(rp, rb)]
    n = len(rp)
    mean_p = sum(rp) / n
    mean_b = sum(rb) / n
    mean_a = sum(ra) / n

    var_a = sum((x - mean_a) ** 2 for x in ra) / (n - 1)
    te = sqrt(var_a) * sqrt(periods_per_year) if var_a > 0 else None
    ir = (mean_a * periods_per_year) / te if te else None

    var_b = sum((x - mean_b) ** 2 for x in rb) / (n - 1)
    cov_pb = sum((p - mean_p) * (b - mean_b) for p, b in zip(rp, rb)) / (n - 1)
    beta = (cov_pb / var_b) if var_b > 0 else None

    p_ret = (p_vals[-1] / p_vals[0]) - 1.0
    b_ret = (b_vals[-1] / b_vals[0]) - 1.0
    return {
        "portfolio_return": p_ret,
        "benchmark_return": b_ret,
        "active_return": p_ret - b_ret,
        "tracking_error": te,
        "information_ratio": ir,
        "beta": beta,
    }


def days_to_exit(quantity, volumes, participation):
    """Estimated trading sessions required to liquidate a position at a given ADV participation rate."""
    qty = abs(float(quantity))
    part = float(participation)
    if qty == 0:
        return 0.0
    if not isfinite(part) or part <= 0:
        raise ValueError("participation must be positive")
    recent = [float(v) for v in volumes if isfinite(float(v)) and float(v) >= 0]
    if not recent:
        return None
    adv = sum(recent) / len(recent)
    if adv <= 0:
        return None
    return qty / (adv * part)
