import json
from collections import Counter

import pytest

from asof.cli import get_parser, run as run_cli
from asof.filing_events import discover, extract_events


def extract(*pages, model=None, fallback_only=False):
    candidates = discover(pages)
    events = extract_events(candidates, "TEST", model, fallback_only=fallback_only)
    for event in events:
        for field in event["fields"].values():
            for evidence in field["citations"]:
                assert evidence["quote"] in pages[evidence["page"] - 1]
    return events


def values(event):
    return {k: f["value"] for k, f in event["fields"].items()}


def test_scbk_style_multi_event_par_value_and_annual_total():
    intro = "Annual results for the year ended 31 December 2024."
    body = ("The Board will be recommending a final dividend of KShs 37.00 for every ordinary share of KShs 5.00. "
            "An interim dividend of KShs 8.00 was declared and paid. "
            "This brings the total dividend to KShs 45.00 per ordinary share.")
    events = extract(intro, body)
    assert [(values(e)["period"], values(e)["amount"], values(e)["status"]) for e in events] == [
        ("2024", "37", "proposed"), ("2024", "8", "paid")]
    assert {c["page"] for c in events[1]["fields"]["amount"]["citations"]} == {2}
    assert any("ordinary share" in c["quote"] for c in events[1]["fields"]["amount"]["citations"])


def test_verdict_is_scoped_to_its_own_resolution():
    text = ("Voting results      Verdict\n"
            "1   Appointment of directors                         PASSED\n"
            "2   Dividend                                         REJECTED\n"
            "a) To approve the final dividend of KES 3 per share for FY2024.\n"
            "3   Approve minutes                                  PASSED")
    event, = extract(text)
    assert values(event).get("status") != "approved"
    passed, = extract(text.replace("REJECTED", "PASSED"))
    assert values(passed)["status"] == "approved"
    assert "PASSED" in [c["quote"] for c in passed["fields"]["status"]["citations"]]


def test_repeated_mentions_prior_comparison_and_conflict():
    line = "The Board proposed a final dividend of KES 5 per share for FY2024, up from a final dividend of KES 4 per share."
    events = extract(line, "Highlights\n" + line, line.replace("KES 5", "KES 6"))
    assert [values(e)["amount"] for e in events] == ["5", "6"]
    assert len(events[0]["candidate_ids"]) == 2
    assert all(e["uncertainties"] for e in events)


@pytest.mark.parametrize("tail", ["to be paid on", "will be paid on", "if approved, will be paid on"])
def test_future_payment_is_not_a_completed_payment(tail):
    event, = extract("The Board recommends an interim dividend of KES 1.25 per share for FY2024, "
                     + tail + " 12 March 2025.")
    assert values(event)["status"] == "proposed"
    assert values(event)["payment_date"] == "2025-03-12"


def test_no_dividend_and_cancellation():
    assert extract("The directors do not recommend a final dividend for FY2025.") == []
    event, = extract("The plan to declare a final dividend for FY2024 was withdrawn.")
    assert values(event)["status"] == "cancelled"


def test_cross_page_continuation_dates_and_qualification():
    events = extract("A final dividend of KES 1.20 per share was declared for FY2023.",
                     "The dividend will be payable on or about 3 January 2024 to shareholders of record.")
    assert len(events) == 1
    field = events[0]["fields"]["payment_date"]
    assert field["value"] == "2024-01-03"
    assert field["qualifier"] == "approximate"
    assert field["citations"][0]["page"] == 2


def test_uncertainty_routed_fallback_skips_complete_rules_and_assists_missing():
    class CountingModel:
        diagnostics = Counter()

        def __init__(self):
            self.calls = 0

        def generate_json(self, prompt, doc, schema):
            self.calls += 1
            return {
                "period": {"value": "2024", "quote": "2024"},
                "status": {"value": "approved", "quote": "fabricated"},
            }

    # Complete core fields -> rules resolve everything, model is NOT called.
    m1 = CountingModel()
    e1, = extract("The Board proposed a final dividend of KES 5.75 per share for 2024.", model=m1, fallback_only=True)
    assert m1.calls == 0
    assert values(e1)["period"] == "2024"

    # Missing period -> model is invoked once to fill missing period, ignores fabricated quote.
    m2 = CountingModel()
    e2, = extract("The Board declared a final dividend of KES 4 per share (ref 2024).", model=m2, fallback_only=True)
    assert m2.calls == 1
    assert values(e2)["period"] == "2024"
    assert values(e2)["status"] == "declared"
    assert m2.diagnostics["fallback_candidates_routed"] == 1


def test_promote_filing_to_local_catalog_and_query_actions(tmp_path):
    pages = (
        "Annual results for the year ended 31 December 2024.",
        "The Board will be recommending a final dividend of KShs 37.00 for every ordinary share of KShs 5.00. "
        "An interim dividend of KShs 8.00 was declared and paid. "
        "This brings the total dividend to KShs 45.00 per ordinary share.",
    )
    events = extract_events(discover(pages), "SCBK")
    report_path = tmp_path / "scbk-report.json"
    report_path.write_text(json.dumps({
        "symbol": "SCBK",
        "source": "https://example.test/scbk-fy2024.pdf",
        "pdf_sha256": "deadbeef",
        "events": events,
    }))
    catalog_dir = tmp_path / "catalog"
    promoted = run_cli(get_parser().parse_args([
        "catalog-add", "--report", str(report_path), "--notice-date", "2025-03-19", "--catalog", str(catalog_dir),
    ]))
    assert promoted["total_notices"] == 2

    # Not visible on notice date itself (next Nairobi midnight rule).
    before = run_cli(get_parser().parse_args([
        "actions", "--as-of", "2025-03-19", "--symbol", "SCBK", "--catalog", str(catalog_dir),
    ]))
    assert before["rows"] == []

    # Visible on 2025-03-20 with both events intact.
    after = run_cli(get_parser().parse_args([
        "actions", "--as-of", "2025-03-20", "--symbol", "SCBK", "--catalog", str(catalog_dir),
    ]))
    by_event = {r["event"]: r["fields"] for r in after["rows"]}
    assert by_event["SCBK-2024-final"]["amount"]["value"] == "37"
    assert by_event["SCBK-2024-final"]["status"]["value"] == "proposed"
    assert by_event["SCBK-2024-interim"]["amount"]["value"] == "8"
    assert by_event["SCBK-2024-interim"]["status"]["value"] == "paid"
