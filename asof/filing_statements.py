"""Extract cited income-statement, EPS, and balance-sheet figures from complete issuer PDFs."""
import re
from decimal import Decimal, InvalidOperation

PERIOD_INLINE = re.compile(
    r"\b(?:(?:financial\s+|fiscal\s+)?year\s+(?:ended|ending)\s+(?:\d{1,2}(?:st|nd|rd|th)?\s+)?"
    r"[A-Za-z]+\s+|FY\s*)(20\d{2})\b", re.I)
HEADER_YEARS = re.compile(r"\b(?:FY\s*|\d{1,2}[./-]\d{1,2}[./-])?(20\d{2})\b", re.I)
NUM_CELL = re.compile(r"^(?:\([\d,]+(?:\.\d+)?\)|-?[\d,]+(?:\.\d+)?%?|-)$")

# Scale headers must NOT have a numeric quantity between currency and unit (rejects prose like "KShs 109.6bn").
SCALE_PATTERNS = (
    (re.compile(r"\b(?:kshs?|shs|kes)\.?\s*(?:'|’)?(?:billions?|bn)\b", re.I), Decimal("1000000000")),
    (re.compile(r"\b(?:kshs?|shs|kes)\.?\s*(?:'|’)?(?:millions?|mn|m)\b", re.I), Decimal("1000000")),
    (re.compile(r"\b(?:kshs?|shs|kes)\.?\s*(?:'|’)?(?:000s?|thousands?)\b", re.I), Decimal("1000")),
)
STACKED_CURRENCY = re.compile(r"\b(?:kshs?|shs|kes)\.?\b", re.I)
STACKED_UNIT = (
    (re.compile(r"\b(?:billions?|bn)\b", re.I), Decimal("1000000000")),
    (re.compile(r"\b(?:millions?|mn)\b", re.I), Decimal("1000000")),
    (re.compile(r"(?:'000|’000|\bthousands?\b)", re.I), Decimal("1000")),
)

# Ordered so more specific labels match before shorter prefixes.
LINE_ITEMS = (
    ("revenue", re.compile(
        r"^(?:total\s+operating\s+income|total\s+revenue|net\s+revenue|service\s+revenue|total\s+income|revenue|turnover)\b",
        re.I), False),
    ("profit_before_tax", re.compile(
        r"^(?:profit\s*/?\s*\(loss\)\s+before\s+(?:income\s+)?tax(?:ation)?(?:\s+and\s+exceptional\s+items)?"
        r"|profit\s+before\s+(?:income\s+)?tax(?:ation)?(?:\s+and\s+exceptional\s+items)?)\b",
        re.I), False),
    ("profit_after_tax", re.compile(
        r"^(?:profit\s*/?\s*\(loss\)\s+after\s+tax(?:ation)?(?:\s+and\s+exceptional\s+items)?"
        r"|profit\s+after\s+tax(?:ation)?(?:\s+and\s+exceptional\s+items)?"
        r"|profit\s+for\s+the\s+(?:year|period)|net\s+profit)\b",
        re.I), False),
    ("eps", re.compile(
        r"^(?:(?:basic(?:\s+(?:and|&)\s+diluted)?\s+)?earnings\s+per\s+share"
        r"(?:\s*[-–]\s*basic(?:\s*(?:and|&)\s*diluted)?)?)\b",
        re.I), True),
    ("total_assets", re.compile(r"^total\s+assets\b", re.I), False),
    ("total_equity", re.compile(
        r"^(?:total\s+(?:shareholders['’]?\s+)?(?:equity|funds)|shareholders['’]?\s+(?:equity|funds))\b",
        re.I), False),
    ("loans_and_advances", re.compile(
        r"^(?:net\s+)?loans\s+and\s+advances(?:\s+to\s+customers)?\b", re.I), False),
    ("customer_deposits", re.compile(
        r"^(?:customer\s+deposits|deposits\s+from\s+customers)\b", re.I), False),
)


def _parse_amount(token):
    cleaned = token.strip()
    if not cleaned or cleaned == "-" or cleaned.endswith("%"):
        return None
    negative = cleaned.startswith("(") and cleaned.endswith(")")
    if negative:
        cleaned = cleaned[1:-1]
    cleaned = cleaned.replace(",", "")
    try:
        val = Decimal(cleaned)
    except InvalidOperation:
        return None
    return -val if negative else val


def _is_header_like(line):
    """True for compact table/section headers, False for multi-word prose sentences."""
    words = [w for w in line.split() if re.search(r"[A-Za-z]", w)]
    return len(words) <= 10 and not line.rstrip().endswith(".")


def _detect_scale(lines, idx):
    """Inspect a header line (or column chunk) and next line for currency + scale unit."""
    chunks = [c.strip() for c in re.split(r"\s{6,}", lines[idx]) if c.strip()]
    for chunk in chunks:
        if not _is_header_like(chunk):
            continue
        for pattern, mult in SCALE_PATTERNS:
            if pattern.search(chunk):
                return mult
    line = lines[idx].strip()
    if _is_header_like(line) and STACKED_CURRENCY.search(line) and idx + 1 < len(lines):
        nxt = lines[idx + 1].strip()
        if _is_header_like(nxt):
            for pattern, mult in STACKED_UNIT:
                if pattern.search(nxt):
                    return mult
    return None


def _detect_periods(line):
    if not _is_header_like(line):
        return None
    years = HEADER_YEARS.findall(line)
    distinct = []
    for y in years:
        if y not in distinct:
            distinct.append(y)
    if len(distinct) >= 2 and (
        " vs " not in line.lower()
        or re.search(r"20\d{2}\s{2,}(?:FY\s*|\d{1,2}[./-]\d{1,2}[./-])?20\d{2}", line, re.I)
    ):
        return distinct[0], distinct[1]
    return None


def _norm_decimal(val):
    return format(val.normalize(), "f")


def extract_financials(pages):
    """Extract cited financial-statement metrics across all physical pages of a filing."""
    default_period = None
    if pages:
        m = PERIOD_INLINE.search(pages[0])
        if m:
            default_period = m[1]

    metrics = {}
    for page_num, text in enumerate(pages, 1):
        lines = text.splitlines()
        scale = Decimal("1")
        current_period = default_period
        prior_period = None

        for idx, raw in enumerate(lines):
            stripped = raw.strip()
            if not stripped:
                continue
            detected_scale = _detect_scale(lines, idx)
            if detected_scale is not None:
                scale = detected_scale
                continue
            periods = _detect_periods(stripped)
            if periods and not any(pat.match(stripped) for _, pat, _ in LINE_ITEMS):
                current_period, prior_period = periods
                continue

            for key, pattern, is_per_share in LINE_ITEMS:
                if key in metrics:
                    continue
                m = pattern.match(stripped)
                if not m:
                    continue
                tail = stripped[m.end():].strip()
                # Strip leading unit/qualifier tags like "(KShs)" or "(Shs)" — never numeric losses like "(1,250)".
                tail = re.sub(r"^\([A-Za-z][^)]*\)\s*", "", tail)
                tokens = tail.split()
                # Table rows begin immediately with numeric cells (rejects prose sentences, tolerates 2-column text to the right).
                if not tokens or not NUM_CELL.match(tokens[0]):
                    continue
                leading = []
                for tok in tokens:
                    if NUM_CELL.match(tok):
                        leading.append(tok)
                    else:
                        break
                nums = [_parse_amount(t) for t in leading]
                nums = [n for n in nums if n is not None]
                if not nums:
                    continue
                row_scale = Decimal("1") if is_per_share else scale
                curr_raw = nums[0]
                curr_val = curr_raw * row_scale
                entry = {
                    "value": _norm_decimal(curr_val),
                    "raw_value": _norm_decimal(curr_raw),
                    "currency": "KES",
                    "unit": "KES/share" if is_per_share else "KES",
                    "scale": _norm_decimal(row_scale),
                    "period": current_period,
                    "citations": [{"page": page_num, "quote": stripped}],
                }
                if len(nums) >= 2:
                    prior_raw = nums[1]
                    entry["prior_period"] = prior_period
                    entry["prior_raw_value"] = _norm_decimal(prior_raw)
                    entry["prior_value"] = _norm_decimal(prior_raw * row_scale)
                metrics[key] = entry
                break

    return metrics
