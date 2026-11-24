import json

import pytest

from asof.cli import get_parser, run as run_cli
from asof.filing_statements import extract_financials
from asof.financials import promote_financials, query_financials


def test_financials_catalog_promote_and_nairobi_midnight_clock(tmp_path):
    pages = [
        "Standard Chartered Bank Kenya Limited results for the year ended 31 December 2024.",
        (
            "Financial performance summary\n"
            "                                31.12.2024   31.12.2023\n"
            "                                      KShs         KShs\n"
            "                                   million      million\n"
            "Total operating income              50,677       41,728\n"
            "Profit after taxation               20,060       13,835\n"
            "Earnings per share (KShs)             52.7         36.2\n"
        ),
    ]
    report = {
        "symbol": "SCBK",
        "source": "https://example.test/scbk-fy24.pdf",
        "pdf_sha256": "abc123",
        "financials": extract_financials(pages),
    }
    report_path = tmp_path / "scbk-fin.json"
    report_path.write_text(json.dumps(report))
    cat_dir = tmp_path / "catalog"

    promoted = run_cli(get_parser().parse_args([
        "financials-add", "--report", str(report_path), "--notice-date", "2025-03-19", "--catalog", str(cat_dir),
    ]))
    assert promoted["total_records"] == 1

    # Not visible on notice date itself under next-Nairobi-midnight disclosure clock.
    before = run_cli(get_parser().parse_args([
        "financials", "--as-of", "2025-03-19", "--catalog", str(cat_dir), "--symbol", "SCBK",
    ]))
    assert before["rows"] == []

    # Visible on 2025-03-20.
    after = run_cli(get_parser().parse_args([
        "financials", "--as-of", "2025-03-20", "--catalog", str(cat_dir), "--symbol", "SCBK",
    ]))
    assert len(after["rows"]) == 1
    row = after["rows"][0]
    assert row["metrics"]["revenue"]["value"] == "50677000000"
    assert row["metrics"]["eps"]["value"] == "52.7"

    # Later restatement on 2025-04-10 supersedes latest view while preserving --history.
    revised = {
        **report,
        "financials": {
            "eps": {**report["financials"]["eps"], "value": "52.8", "raw_value": "52.8"},
        },
    }
    promote_financials(revised, "2025-04-10", cat_dir)
    hist = query_financials("2025-04-15", root=cat_dir, symbol="SCBK", metric="eps", history=True)
    assert [r["metrics"]["eps"]["value"] for r in hist["rows"]] == ["52.7", "52.8"]
    latest = query_financials("2025-04-15", root=cat_dir, symbol="SCBK", metric="eps")
    assert latest["rows"][0]["metrics"]["eps"]["value"] == "52.8"

    with pytest.raises(ValueError, match="no extracted financial metrics"):
        promote_financials({"symbol": "SCBK", "source": "x", "financials": {}}, "2025-03-19", cat_dir)
