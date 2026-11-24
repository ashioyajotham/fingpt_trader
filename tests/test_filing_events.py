from collections import Counter

import pytest

from fingpt_research.filing_events import discover, extract_events


def run(*pages, model=None):
    candidates = discover(pages)
    events = extract_events(candidates, "TEST", model)
    for event in events:
        for field in event["fields"].values():
            for evidence in field["citations"]:
                assert evidence["quote"] in pages[evidence["page"] - 1]
    return events


def values(event):
    return {k: f["value"] for k, f in event["fields"].items()}


def test_separate_current_instalments_context_unit_and_annual_total():
    intro = "Annual results for the year ended 31 December 2023."
    body = ("We recommend a final dividend of KES 3.25 for every ordinary share of KES 1.00. "
            "An interim dividend of KES 0.75 was declared and paid. "
            "This brings the total dividend to KES 4 per ordinary share.")
    events = run(intro, body)
    assert [(values(e)["period"], values(e)["amount"], values(e)["status"]) for e in events] == [
        ("2023", "3.25", "proposed"), ("2023", "0.75", "paid")]
    assert {c["page"] for c in events[1]["fields"]["amount"]["citations"]} == {2}
    assert any("ordinary share" in c["quote"] for c in events[1]["fields"]["amount"]["citations"])


def test_verdict_is_scoped_to_its_own_resolution():
    text = ("Voting results      Verdict\n"
            "1   Appointment of directors                         PASSED\n"
            "2   Dividend                                         REJECTED\n"
            "a) To approve the final dividend of KES 3 per share for FY2024.\n"
            "3   Approve minutes                                  PASSED")
    event, = run(text)
    assert values(event).get("status") != "approved"
    passed, = run(text.replace("REJECTED", "PASSED"))
    assert values(passed)["status"] == "approved"
    assert "PASSED" in [c["quote"] for c in passed["fields"]["status"]["citations"]]


def test_repeated_mentions_prior_comparison_and_conflict():
    line = "The Board proposed a final dividend of KES 5 per share for FY2024, up from a final dividend of KES 4 per share."
    events = run(line, "Highlights\n" + line, line.replace("KES 5", "KES 6"))
    assert [values(e)["amount"] for e in events] == ["5", "6"]
    assert len(events[0]["candidate_ids"]) == 2
    assert all(e["uncertainties"] for e in events)


@pytest.mark.parametrize("tail", ["to be paid on", "will be paid on", "if approved, will be paid on"])
def test_future_payment_is_not_a_completed_payment(tail):
    event, = run("The Board recommends an interim dividend of KES 1.25 per share for FY2024, "
                 + tail + " 12 March 2025.")
    assert values(event)["status"] == "proposed"
    assert values(event)["payment_date"] == "2025-03-12"


def test_no_dividend_and_cancellation():
    assert run("The directors do not recommend a final dividend for FY2025.") == []
    event, = run("The plan to declare a final dividend for FY2024 was withdrawn.")
    assert values(event)["status"] == "cancelled"


def test_cross_page_continuation_dates_and_qualification():
    events = run("A final dividend of KES 1.20 per share was declared for FY2023.",
                 "The dividend will be payable on or about 3 January 2024 to shareholders of record.")
    assert len(events) == 1
    field = events[0]["fields"]["payment_date"]
    assert field["value"] == "2024-01-03"
    assert field["qualifier"] == "approximate"
    assert field["citations"][0]["page"] == 2


def test_model_receives_discovered_source_and_rejects_fabricated_quote():
    class Model:
        diagnostics = Counter()

        def generate_json(self, prompt, doc, schema):
            assert all(isinstance(p, str) for p in doc["pages"])
            return {
                "amount": {"value": "4", "quote": "KES 4 per share"},
                "currency": {"value": "KES", "quote": "made up"},
                "distribution_type": {"value": "final", "quote": "final dividend"},
            }

    event, = run("A final dividend of KES 4 per share was declared.", model=Model())
    assert values(event) == {"amount": "4", "distribution_type": "final"}
