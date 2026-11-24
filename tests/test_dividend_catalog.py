from copy import deepcopy

import pytest

from fingpt_research.cli import get_parser, run
from fingpt_research.dividend_eval import evaluate, predict
from fingpt_research.dividends import actions, load_corpus
from tools.dividend_review import packet, reconcile


def query(*args):
    return run(get_parser().parse_args(["actions", *args]))["rows"]


def test_agenda_vote_and_offset_boundary():
    before = query("--as-of", "2026-06-24T20:59:59Z", "--symbol", "eqty")
    after = query("--as-of", "2026-06-24T21:00:00Z", "--symbol", "EQTY", "--status", "approved")
    find = lambda rows: next(r for r in rows if r["event"] == "EQTY-2025-final")
    assert find(before)["fields"]["status"]["value"] == "proposed"
    assert find(after)["fields"]["status"]["value"] == "approved"
    assert find(after)["fields"]["amount"]["value"] == "5.75"
    assert "payment_date" not in find(before)["fields"]
    assert query("--as-of", "2026-07-01", "--clock", "observed") == []
    assert query("--as-of", "2026-11-24", "--clock", "observed")
    with pytest.raises(ValueError):
        actions("2026-06-30T12:00:00")


def test_revised_terms_and_independent_interim_final_events():
    rows = query("--as-of", "2025-08-21", "--symbol", "EABL", "--history")
    changes = [r["fields"] for r in rows if r["event"] == "EABL-2025-interim"]
    assert [r["record_date"]["value"] for r in changes] == ["2025-02-21", "2025-02-16"]
    assert [r["payment_date"]["value"] for r in changes] == ["2025-04-30", "2025-04-28"]
    assert changes[-1]["amount"]["passage_id"] != changes[-1]["status"]["passage_id"]
    final = next(r for r in rows if r["event"] == "EABL-2025-final")
    assert final["fields"]["status"]["value"] == "proposed"
    later = query("--as-of", "2025-09-12", "--symbol", "EABL", "--status", "approved")
    assert any(r["event"] == "EABL-2025-final" for r in later)
    assert not any(r["event"] == "EABL-2025-interim" for r in later)


def test_frozen_splits_and_model_input_boundary():
    docs, membership, _, _ = load_corpus()
    names = {split: {d["symbol"] for d in docs if membership[d["id"]]["split"] == split}
             for split in ("development", "holdout", "prospective")}
    assert names["prospective"].isdisjoint(names["development"] | names["holdout"])
    class Dummy:
        def generate_json(self, instruction, doc, schema):
            assert set(doc) == {"pages"}
            return {"status": {"value": "approved", "quote": "invented"}}
    assert predict(Dummy(), docs[0]) == {}
    result = evaluate("prospective")
    assert result["passages"] == 7
    assert result["rules"]["totals"]["expected"] == 20
    assert result["reference_status"] == "adjudicated"
    assert load_corpus()[-1] != result["corpus_sha256"]
    rows = query("--as-of", "2025-03-20", "--symbol", "SCBK", "--status", "paid")
    assert next(r for r in rows if r["event"] == "SCBK-2024-interim")["fields"]["amount"]["value"] == "8"


def test_blind_reconciliation_requires_verbatim_sources():
    review = packet()
    assert all(not a["value"] for r in review["notices"] for a in r["fields"].values())
    review["reviewer"] = "test"
    row = review["notices"][0]
    row["reviewed"] = True
    assert reconcile(review)["differences"]
    invalid = deepcopy(review)
    invalid["notices"][0]["fields"]["status"] = {"value": "approved", "span": "fabricated"}
    with pytest.raises(ValueError, match="verbatim"):
        reconcile(invalid)
