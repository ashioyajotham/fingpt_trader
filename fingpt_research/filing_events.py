"""Find and link dividend mentions in the page text of a complete issuer filing."""
import re
from collections import defaultdict

from .dividend_eval import DATE, ITEM, MONEY, normalize

PERIOD = re.compile(
    r"\b(?:(?:financial\s+|fiscal\s+)?year\s+(?:ended|ending)\s+(?:\d{1,2}(?:st|nd|rd|th)?\s+)?"
    r"[A-Za-z]+\s+|FY\s*)(20\d{2})\b", re.I)
TYPE = re.compile(r"\b(interim|final|special)\s+(?:cash\s+)?dividend\b", re.I)
UNIT = re.compile(r"\b(?:per|for (?:each|every)|a) (?:ordinary )?share\b", re.I)
CURRENCY = re.compile(MONEY, re.I)
RECORD = re.compile(r"\b(?:close of business|record date|book closure|books close|register(?:ed)?)"
                    r"[^.;]{0,65}?(" + DATE + ")", re.I)
PAYMENT = re.compile(r"\b(?:paid|payable|payment date|payment on)[^.;]{0,55}?(" + DATE + ")", re.I)
ACTIONS = (
    ("cancelled", r"\b(?:withdrawn|cancelled)\b"),
    ("paid", r"\b(?:was (?:declared and )?paid|already paid|previously paid|has been paid|have been paid|were paid|company paid)\b"),
    ("approved", r"\b(?:(?:shareholders|directors|board) (?:have |has )?(?:approved|endorsed)"
                 r"|has approved|duly approved|resolved by majority)\b"),
    ("proposed", r"\b(?:recommend(?:ed|ing|s)?|propos(?:ed|ing|es)|to approve|to declare)\b"),
    ("declared", r"\b(?:declared|announced)\b"),
)
FIELD_NAMES = ("distribution_type", "period", "status", "amount", "currency", "record_date", "payment_date")
REJECT = re.compile(r"\b(?:no dividend|not (?:recommend(?:ing)?|declare|approved)|dividend policy)\b", re.I)
SENTENCES = re.compile(r"(?<=[.!?])\s+(?=[A-Z])|\s+(?:and|while|but)\s+"
                       r"(?=(?:to (?:approve|confirm|declare) |(?:an?|the) (?:final|interim|special) dividend))")
SCHEMA = {"type": "object", "properties": {name: ITEM for name in FIELD_NAMES}, "additionalProperties": False}
PROMPT = (
    "Extract one cash dividend from the first passage, using the labeled supporting context where needed. "
    "Ignore totals, historical comparators, percentages and par values. Omit unsupported fields. Use "
    "distribution_type interim/final/special, period as a four-digit fiscal year, status proposed/approved/"
    "declared/paid/cancelled, amount per share, currency ISO code, record_date and payment_date YYYY-MM-DD. "
    "A future payment is not paid. Each returned field must contain value and a short exact quote from "
    "one input passage. Use an empty object if there is no event."
)


class Passage:
    """A whitespace-normalized reading view with offsets back to the literal PDF line fragments."""

    def __init__(self, fragments):
        words, offsets = [], []
        position = 0
        for index, part in enumerate(fragments):
            for match in re.finditer(r"\S+", part["text"]):
                words.append(match[0])
                end = position + len(match[0])
                offsets.append((position, end, index, match.start()))
                position = end + 1
        self.text = " ".join(words)
        self.fragments = fragments
        self.offsets = offsets

    def span(self, start, end):
        ranges = {}
        for left, right, i, offset in self.offsets:
            if right <= start:
                continue
            if left >= end:
                break
            a, b = offset + max(0, start-left), offset + min(right, end)-left
            if i not in ranges:
                ranges[i] = [a, b]
            else:
                ranges[i][1] = b
        return [{"page": self.fragments[i]["page"], "text": self.fragments[i]["text"][a:b]} for i,(a,b) in ranges.items()]

    def cite(self, start, end):
        return [{"page": f["page"], "quote": f["text"]} for f in self.span(start, end)]


def _columns(page):
    """Read columns in order where repeated wide gutters separate two bodies of prose."""
    lines = page.splitlines()
    if re.search(r"\bverdict\b", page, re.I):
        return [lines]
    separators = []
    for line in lines:
        gaps = [g for g in re.finditer(r"(?<=\S) {18,}(?=\S)", line) if g.end() >= 65]
        separators.extend(gaps)
    # Multiple lines must share a physical gutter. Left indents and table figures alone are insufficient.
    if len(separators) < 3:
        return [lines]
    coverage = defaultdict(int)
    for gap in separators:
        for pos in range(gap.start()+1, gap.end()):
            coverage[pos] += 1
    mid = max(coverage, key=coverage.get)
    if coverage[mid] < max(3, len(separators)//2):
        return [lines]
    left, right = [], []
    for line in lines:
        if len(line) > mid and line[mid].isspace() and (mid == 0 or line[mid-1].isspace()):
            left.append(line[:mid].rstrip())
            right.append(line[mid:].strip())
        else:
            left.append(line)
            right.append("")
    return [left, right]


def _blocks(lines, page):
    """A polling verdict belongs to its numbered resolution or individual subresolution."""
    body, verdict, parent = [], None, None
    for line in lines:
        raw = line.strip()
        if not raw:
            # White space is common inside a table row and across newspaper columns.
            continue
        numbered = re.match(r"^\d{1,2}(?:[.)]\s+|\s{2,})", raw)
        sub = re.match(r"^\(?[a-z][.)]\s+", raw)
        heading = len(raw.split()) <= 5 and raw[0].isupper() and not re.search(r"[.,;:]|\d", raw)
        if numbered or sub or heading:
            if body:
                yield body, verdict
            body = []
            if numbered or heading:
                parent = None
            verdict = parent
        row_vote = re.search(r"\s{3,}((?:PASSED|CARRIED|REJECTED|FAILED))\s*$", raw, re.I)
        if row_vote:
            verdict = {"page": page, "text": row_vote[1]}
            raw = re.split(r"\s{3,}(?=[\d,]+\s{2,}[\d.]+%)", raw)[0]
            raw = re.sub(r"\s{3,}(PASSED|CARRIED|REJECTED|FAILED)\s*$", "", raw, flags=re.I)
            if numbered and re.fullmatch(r"\d+[.)]?\s+dividends?", raw, re.I):
                parent = verdict
        if raw:
            body.append({"page": page, "text": raw})
    if body:
        yield body, verdict


COMPARISON = re.compile(r"\b(?:up from|down from|compared (?:with|to)|versus|in the prior year|in the previous year"
                        r"|last year's|prior-year|previous year's|in comparison)\b", re.I)
CANCELLED = re.compile(ACTIONS[0][1], re.I)


def _parts(view):
    """Group a dividend sentence with its dates, or with an explicit per-share basis in a following total."""
    breaks = [0, *(m.end() for m in SENTENCES.finditer(view.text)), len(view.text)]
    pending = None
    for a,b in zip(breaks,breaks[1:]):
        text = view.text[a:b]
        keyword = re.search(r"\bdividends?\b", text, re.I)
        total = re.search(r"\b(?:total|annual aggregate) dividend", text, re.I)
        comparator = COMPARISON.search(text)
        if keyword and comparator and comparator.start() <= keyword.start():
            continue
        if keyword and comparator:
            b = a + comparator.start()
            text = text[:comparator.start()]
        actionable = keyword and not total and (not REJECT.search(text) or CANCELLED.search(text)) and (
            TYPE.search(text) or CURRENCY.search(text) or
            any(re.search(pat, text, re.I) for _, pat in ACTIONS))
        if actionable:
            if pending:
                yield pending
            pending = [view.span(a,b), []]
        elif pending and total and UNIT.search(text):
            pending[1].extend(view.span(a,b))
        elif pending and re.match(r"(?:the |this |such )?(?:dividend|payment|books|shareholders|it)\b", text, re.I):
            pending[0].extend(view.span(a,b))
        elif pending:
            yield pending
            pending = None
    if pending:
        yield pending


def discover(pages):
    candidates = []
    filing_period = []
    if pages:
        # A results dateline can establish a fiscal year for later narrative. Prefer explicit reporting language.
        first = Passage([{"page": 1, "text": line} for line in pages[0].splitlines() if line.strip()])
        match = re.search(r"(?:results|statements) for the " + PERIOD.pattern, first.text, re.I)
        if match:
            filing_period = first.span(*match.span())
    for page, text in enumerate(pages, 1):
        # Join only a recognizable continuation at the very top of the next page.
        # A fresh numbered item or a new distribution starts its own candidate.
        nonempty = [line.strip() for line in text.splitlines() if line.strip()]
        if candidates and nonempty and re.match(
                r"^(?:to shareholders|registered at|at the close of business|on the register|per ordinary share"
                r"|(?:the|this) dividend (?:will be|is) payable)\b", nonempty[0], re.I):
            last = candidates[-1]
            focus = [f for f in last["fragments"] if f["role"] == "event"]
            if focus[-1]["page"] == page-1 and pages[page-2].rstrip().endswith(focus[-1]["text"]):
                for line in nonempty:
                    if re.match(r"^\(?[\da-z]+[.)]\s+", line) or TYPE.search(line):
                        break
                    last["fragments"].append({"page": page, "text": line, "role": "event"})
                    if line.endswith("."):
                        break
        for column in _columns(text):
            for lines, vote in _blocks(column, page):
                for focus, units in _parts(Passage(lines)):
                    fragments = []
                    for role, parts in (("event", focus), ("units", units), ("verdict", [vote] if vote else []),
                                        ("period", filing_period)):
                        fragments.extend({**part, "role": role} for part in parts)
                    candidates.append({"candidate_id": f"candidate-{len(candidates)+1:03d}", "fragments": fragments})
    return candidates


def _roles(candidate):
    groups = defaultdict(list)
    for part in candidate["fragments"]:
        groups[part.get("role", "event")].append(part)
    return {role: Passage(parts) for role, parts in groups.items()}


def _field(value, citations):
    return {"value": value, "citations": citations}


def _rule_fields(roles):
    event = roles["event"]
    text = event.text
    fields = {}
    kind = TYPE.search(text)
    if kind:
        fields["distribution_type"] = _field(kind[1].lower(), event.cite(*kind.span()))
    for label, pattern in ACTIONS:
        hit = re.search(pattern, text, re.I)
        if hit:
            fields["status"] = _field(label, event.cite(*hit.span()))
            break
    verdict = roles.get("verdict")
    if verdict and fields.get("status", {}).get("value") != "paid":
        if verdict.text.upper() in {"PASSED", "CARRIED"}:
            anchors = fields.get("distribution_type", {}).get("citations", [])
            fields["status"] = _field("approved", anchors + verdict.cite(0, len(verdict.text)))
        else:
            fields.pop("status", None)
    for part in (event, roles.get("period")):
        hit = PERIOD.search(part.text) if part else None
        if hit:
            fields["period"] = _field(hit[1], part.cite(*hit.span()))
            break
    for m in CURRENCY.finditer(text):
        tail = text[m.end():]
        if re.match(r"\s*(?:billion|million|thousand|bn\b|mn\b|%)", tail, re.I):
            continue
        # Skip par values and a prior-year number inside comparative parentheses.
        if re.search(r"(?:ordinary )?share of\s*$|\(\s*20\d{2}\s*:\s*$", text[:m.start()], re.I):
            continue
        units = event
        unit = UNIT.search(tail[:95])
        if unit and CURRENCY.search(tail[:unit.start()]):
            # A second amount occurs before this unit. It may be a total or prior-year comparator.
            if not re.match(r"^\s*\(20\d{2}:", tail):
                continue
        if unit:
            start, end = m.end()+unit.start(), m.end()+unit.end()
        else:
            units = roles.get("units")
            unit = UNIT.search(units.text) if units else None
            if not unit:
                continue
            start, end = unit.span()
        amount = normalize("amount", m["amount"])
        if amount:
            citations = event.cite(*m.span()) + units.cite(start, end)
            fields["amount"] = _field(amount, citations)
            fields["currency"] = _field(normalize("currency", m["currency"]), event.cite(*m.span("currency")))
            break
    for key, pattern in (("record_date", RECORD), ("payment_date", PAYMENT)):
        hit = pattern.search(text)
        if hit and (date := normalize(key, hit[1])):
            fields[key] = _field(date, event.cite(*hit.span()))
            if re.search(r"\b(?:about|around|approximately)\b", hit[0], re.I):
                fields[key]["qualifier"] = "approximate"
    return fields


def _model_fields(roles, model):
    pieces = {key: part.text for key,part in roles.items()}
    pages = [f"{role}: {v}" for role,v in pieces.items()]
    # LocalFinGPT reserves 768 of its 4096 context tokens for generation.
    if hasattr(model, "model"):
        from json import dumps
        prompt = "Instruction: " + PROMPT + "\nInput: " + dumps(pages, ensure_ascii=False) + "\nAnswer: "
        if len(model.model.tokenize(prompt.encode())) + 768 > model.model.n_ctx():
            model.diagnostics["filing_context_overflow"] += 1
            return {}
    raw = model.generate_json(PROMPT, {"pages": pages}, SCHEMA)
    fields = {}
    for name in FIELD_NAMES:
        item = raw.get(name)
        if not isinstance(item, dict):
            continue
        value, quote = item.get("value"), item.get("quote")
        if not isinstance(quote,str) or not quote.strip() or not isinstance(value,str):
            continue
        part = next((p for p in roles.values() if quote in p.text), None)
        if part is None:
            continue
        if name == "distribution_type":
            value = value.lower() if value.lower() in {"final","interim","special"} else None
        elif name == "period":
            value = value if re.fullmatch(r"20\d{2}",value) and value in quote else None
        else:
            value = normalize(name,value)
        if value:
            start = part.text.index(quote)
            fields[name] = _field(value, part.cite(start, start+len(quote)))
            if name in {"payment_date", "record_date"}:
                context = part.text[max(0, start-20):start+len(quote)]
                if re.search(r"\b(?:about|around|approximately)\s+(?:[A-Za-z]+\s+)?\d{1,2}", context, re.I):
                    fields[name]["qualifier"] = "approximate"
    return fields


def extract_events(candidates, symbol, model=None):
    events, groups = [], defaultdict(list)
    for candidate in candidates:
        roles = _roles(candidate)
        fields = _rule_fields(roles) if model is None else _model_fields(roles,model)
        if not (("status" in fields or "amount" in fields) and
                ("distribution_type" in fields or "amount" in fields)):
            continue
        identity = tuple(fields.get(k,{}).get("value") for k in ("distribution_type", "period"))
        matches = groups[identity] if all(identity) else []
        compatible = [e for e in matches if all(
            e["fields"][k]["value"] == fields[k]["value"] for k in fields.keys() & e["fields"].keys())]
        if len(compatible) == 1:
            prior = compatible[0]
            prior["candidate_ids"].append(candidate["candidate_id"])
            for k, field in fields.items():
                if k not in prior["fields"]:
                    prior["fields"][k] = field
                else:
                    seen = prior["fields"][k]["citations"]
                    seen.extend(c for c in field["citations"] if c not in seen)
            continue
        uncertainty = []
        if not all(identity):
            uncertainty.append("Incomplete fiscal-period or distribution-type evidence; grouping unresolved")
        if matches:
            conflict = "Multiple mentions for this period and type; review conflicting or ambiguous linkage"
            uncertainty.append(conflict)
            for other in matches:
                if conflict not in other["uncertainties"]:
                    other["uncertainties"].append(conflict)
        entry = {"event_id": f"event-{len(events)+1:03d}", "symbol": symbol.upper(), "fields": fields,
                 "candidate_ids":[candidate["candidate_id"]], "uncertainties": uncertainty}
        events.append(entry)
        if all(identity):
            groups[identity].append(entry)
    return events
