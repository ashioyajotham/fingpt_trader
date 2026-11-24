"""Signed position values, drawdown, risk-adjusted returns, and liquidity metrics."""

from math import isfinite, sqrt


def exposures(values, nav):
    if not isfinite(nav) or nav <= 0 or any(not isfinite(v) for v in values):
        raise ValueError("exposure requires finite position values and positive NAV")
    return sum(abs(v) for v in values) / nav, sum(values) / nav


def drawdown(nav, peak):
    if peak <= 0:
        raise ValueError("drawdown requires a positive high-water mark")
    return max(0, (peak - nav) / peak)


def sharpe_ratio(nav_series, periods_per_year=252):
    """Annualized Sharpe ratio (zero risk-free baseline) from a sequence of positive NAV marks."""
    values = [float(v) for v in nav_series]
    if len(values) < 3 or any(not isfinite(v) or v <= 0 for v in values):
        return None
    returns = [(b / a) - 1.0 for a, b in zip(values, values[1:])]
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    if variance <= 0:
        return None
    return (mean / sqrt(variance)) * sqrt(periods_per_year)


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
