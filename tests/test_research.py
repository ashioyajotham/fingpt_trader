import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from decimal import Decimal as D
from pathlib import Path
from unittest.mock import Mock, patch

from fingpt_research.cli import main
from fingpt_research.data import FIXTURES, instant, load_dataset
from fingpt_research.evidence import LocalFinGPT, extract_rules, score_extractor
from fingpt_research.experiments import compare, load_features, model_features, simulate
from fingpt_research.forward import ForwardBook
from fingpt_research.paper import Assumptions, PaperEngine, Signal
from fingpt_research.risk import drawdown, exposures


class ResearchRelease(unittest.TestCase):
    def setUp(self):
        self.ds = load_dataset(FIXTURES / "nse")
        self.cfg = Assumptions()

    def test_as_of_revisions_and_date_only(self):
        def visible(cutoff):
            return {d["id"]: d for d in self.ds.report(instant(cutoff))["documents"]}
        self.assertEqual(visible("2025-01-03T15:00:00+03:00"), {})
        first = visible("2025-01-06T15:00:00+03:00")
        self.assertEqual(set(first), {"statement"})
        self.assertEqual(first["statement"]["available_at"], instant("2025-01-06T09:30:00+03:00"))
        before = visible("2025-01-08T15:00:00+03:00")
        self.assertFalse(before["dividend-original"]["superseded"])
        self.assertNotIn("dividend-correction", before)
        after = visible("2025-01-09T15:00:00+03:00")
        self.assertTrue(after["dividend-original"]["superseded"])
        self.assertEqual(after["dividend-correction"]["dividend"]["amount"], "1.20")

    def test_offline_entrypoints_and_determinism(self):
        with patch("socket.socket", side_effect=AssertionError("network access")):
            for name in ("replay", "paper", "evidence", "compare"):
                outputs = []
                for _ in range(2):
                    stream = io.StringIO()
                    with redirect_stdout(stream):
                        main([name, "--dataset", "nse"])
                    outputs.append(stream.getvalue())
                self.assertEqual(*outputs)

    def test_dividends_fills_and_constraints(self):
        book = simulate(self.ds, "buy_hold", self.cfg)
        self.assertEqual(len(book.cash_events), 1)
        self.assertEqual(book.cash_events[0]["session"], "2025-01-16")
        amount = book.held["2025-01-10"] * D("1.20")
        self.assertGreater(amount, 0)
        self.assertEqual(book.cash_events[0]["amount"], amount)
        entries = {r["session"]: r for r in book.ledger}
        self.assertEqual(entries["2025-01-13"]["receivable"], amount)
        self.assertEqual(entries["2025-01-16"]["receivable"], 0)
        self.assertNotIn(book.fills[0]["session"], {"2025-01-02"})
        for fill in book.fills:
            self.assertLess(fill["decided_at"], fill["filled_at"])
            self.assertNotIn(fill["session"], {"2025-01-09", "2025-01-10"})
            self.assertGreater(fill["fee"], 0)
        limited = simulate(self.ds, "buy_hold", replace(self.cfg, max_exposure=D("0.5")))
        self.assertGreater(len(limited.fills), 0)
        self.assertLessEqual(limited.ledger[1]["gross_exposure"], D("0.5"))
        self.assertTrue(all(row["cash"] >= 0 and row["quantity"] >= 0 for row in book.ledger))

    def test_no_fill_when_volume_zero_or_suspended(self):
        bars = self.ds.bars
        engine = PaperEngine(self.ds.meta, "buy_hold", self.cfg)
        for bar in [bars[0], replace(bars[1], volume=D(0)), replace(bars[2], status="suspended")]:
            engine.observe(bar)
        self.assertEqual(engine.fills, [])
        engine.observe(bars[3])
        self.assertEqual(len(engine.fills), 1)

    def test_delayed_publication_cannot_fill_at_earlier_close(self):
        bars = self.ds.bars
        engine = PaperEngine(self.ds.meta, "buy_hold", self.cfg)
        engine.observe(replace(bars[0], available_at=bars[1].close_time))
        engine.observe(bars[1])
        self.assertEqual(engine.fills, [])

    def test_risk_examples(self):
        self.assertEqual(exposures([50, -50], 100), (1, 0))
        self.assertEqual([drawdown(v, 100) for v in (100, 80, 100)], [0, 0.2, 0])
        with self.assertRaises(ValueError):
            exposures([10], 0)

    def test_custom_signal_producer_cannot_see_future_features(self):
        latest = self.ds.bars[-1]
        feature = Signal(self.ds.meta["instrument"], "example", "1", latest.available_at, 1, D(1), ("news",))
        seen = []

        def researcher(instrument, bar, previous, available):
            seen.append(available)
            return None

        book = simulate(self.ds, researcher, self.cfg, [feature])
        self.assertTrue(all(not items for items in seen[:-1]))
        self.assertEqual(seen[-1], (feature,))
        self.assertEqual(book.fills, [])

    def test_forward_equals_replay_and_rejects_history_edits(self):
        first = replace(self.ds, bars=self.ds.bars[:4], documents=self.ds.documents[:2])
        book = ForwardBook(first, "buy_hold", self.cfg, [])
        self.assertTrue(book.advance(self.ds, []))
        self.assertFalse(book.advance(self.ds, []))
        self.assertEqual(book.engine.report(), simulate(self.ds, "buy_hold", self.cfg).report())
        with self.assertRaises(ValueError):
            book.advance(first, [])
        with self.assertRaises(ValueError):
            book.advance(replace(self.ds, documents=self.ds.documents + [
                {**self.ds.documents[0], "id": "backdated"}
            ]), [])

    def test_cited_extraction_and_abstention(self):
        labels = json.loads((FIXTURES / "nse" / "labels.json").read_text())
        result = score_extractor(self.ds.documents, labels, extract_rules)
        self.assertEqual((result["expected"], result["supported"]), (20, 20))
        doc = self.ds.documents[1]
        fake = object.__new__(LocalFinGPT)
        fake.passages = Mock(return_value=[(1, doc["pages"][0])])
        fake.generate_json = Mock(return_value={
            "amount": {"page": 1, "quote": "invented", "value": "4"},
            "currency": {"page": 1, "quote": "KES 1.00 per share", "value": "KES"}
        })
        self.assertEqual(set(fake.extract(doc)), {"currency"})
        abstain = score_extractor([doc], labels, lambda _: {})
        self.assertEqual(abstain["abstentions"], 7)
        self.assertIsNone(abstain["citation_precision"])

    def test_issuer_release_excerpt(self):
        doc = {"kind": "dividend_notice", "attribution": "issuer results",
               "source": "https://www.safaricom.co.ke/images/Downloads/FY25-Press-Release_May-9-2025.pdf",
               "pages": ["proposed a final dividend of KShs 0.65 per ordinary share"]}
        claims = extract_rules(doc)
        self.assertEqual({k: v["value"] for k, v in claims.items()},
                         {"status": "proposed", "currency": "KES", "amount": "0.65"})
        self.assertIn("KShs", claims["currency"]["quote"])

    def test_model_feature_timing_and_comparison(self):
        model = Mock(version="test-stub", provider="stub")
        model.name = "test"
        model.sentiment.return_value = -1
        rows = model_features(model, self.ds, instant("2024-01-01T00:00:00Z"), 60)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "features.json"
            path.write_text(json.dumps(rows, default=str))
            features = load_features(path, self.ds)
            result = compare(self.ds, self.cfg, features, "fixture")
            self.assertEqual(set(result["policies"]), {"cash", "buy_hold", "price", "price_ml"})
            self.assertEqual(result["policies"]["cash"]["net_return"], 0)
            self.assertEqual(len(result["policies"]["price"]["folds"]), 3)
            rows[0]["observed_at"] = "2024-01-01T00:00:00Z"
            path.write_text(json.dumps(rows, default=str))
            with self.assertRaises(ValueError):
                load_features(path, self.ds)
