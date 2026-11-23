"""Signed position values and NAV must be expressed in the same currency."""

from math import isfinite


def exposures(values, nav):
    if not isfinite(nav) or nav <= 0 or any(not isfinite(v) for v in values):
        raise ValueError("exposure requires finite position values and positive NAV")
    return sum(abs(v) for v in values) / nav, sum(values) / nav


def drawdown(nav, peak):
    if peak <= 0:
        raise ValueError("drawdown requires a positive high-water mark")
    return max(0, (peak - nav) / peak)
