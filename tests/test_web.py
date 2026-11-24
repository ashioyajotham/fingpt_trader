import json
from http import HTTPStatus

import pytest

from asof.web import dispatch, dispatch_post


def _make_filing_pdf(path):
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                             NameObject("/Subtype"): NameObject("/Type1"),
                             NameObject("/BaseFont"): NameObject("/Helvetica")})

    def add_page(lines):
        page = writer.add_blank_page(width=612, height=792)
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})})
        cmds = [b"BT /F1 10 Tf"]
        y = 740
        for line in lines:
            esc = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            cmds.append(f"1 0 0 1 30 {y} Tm ({esc}) Tj".encode("latin1"))
            y -= 16
        cmds.append(b"ET")
        stream = DecodedStreamObject()
        stream.set_data(b"\n".join(cmds))
        page[NameObject("/Contents")] = writer._add_object(stream)

    add_page([
        "Standard Chartered results for the year ended 31 December 2024.",
        "The Board will be recommending a final dividend of KShs 37.00 for every ordinary share of KShs 5.00.",
        "An interim dividend of KShs 8.00 was declared and paid.",
        "This brings the total dividend to KShs 45.00 per ordinary share.",
    ])
    add_page([
        "Financial performance summary (KShs million)",
        "Item                             31.12.2024     31.12.2023",
        "Total operating income               50,677         41,728",
        "Profit after taxation                20,060         13,835",
        "Earnings per share (KShs)              52.7           36.2",
    ])
    writer.write(path)


def test_web_workstation_routes_offline():
    code, ctype, body = dispatch("/", {})
    assert code == HTTPStatus.OK
    assert "text/html" in ctype
    assert b"PAPER-ONLY" in body and b"Filing Workbench" in body

    code, _, body = dispatch("/favicon.ico", {})
    assert code == HTTPStatus.NO_CONTENT and body == b""

    code, _, body = dispatch("/api/health", {})
    assert code == HTTPStatus.OK
    assert json.loads(body) == {
        "app": "AsOf", "broker_connected": False, "catalog_writable": False, "mode": "paper-only",
    }

    code, _, body = dispatch("/api/actions", {"as_of": ["2026-06-25"], "symbol": ["EQTY"]})
    assert code == HTTPStatus.OK
    rows = json.loads(body)["rows"]
    assert any(r["event"] == "EQTY-2025-final" and r["fields"]["status"]["value"] == "approved" for r in rows)

    code, _, body = dispatch("/api/replay", {"dataset": ["nse"]})
    assert code == HTTPStatus.OK
    assert len(json.loads(body)["observations"]) == 13

    code, _, body = dispatch("/api/paper", {"dataset": ["nse"], "policy": ["buy_hold"]})
    assert code == HTTPStatus.OK
    paper = json.loads(body)
    assert "sharpe_ratio" in paper and "days_to_exit" in paper and "splits" in paper

    code, _, body = dispatch("/api/compare", {"dataset": ["nse"]})
    assert code == HTTPStatus.OK
    assert "policies" in json.loads(body)

    code, _, _ = dispatch("/api/unknown", {})
    assert code == HTTPStatus.NOT_FOUND

    code, _, body = dispatch("/api/actions", {"as_of": ["not-a-date"]})
    assert code == HTTPStatus.BAD_REQUEST
    assert "error" in json.loads(body)


def test_web_filing_workbench_extract_and_catalog_promotion(tmp_path):
    pytest.importorskip("pypdf")
    pytest.importorskip("pypdfium2")
    pdf_path = tmp_path / "scbk.pdf"
    _make_filing_pdf(pdf_path)

    code, _, body = dispatch_post(
        "/api/filing-extract",
        {"symbol": ["SCBK"], "source": ["https://example.test/scbk.pdf"]},
        pdf_path.read_bytes(),
    )
    assert code == HTTPStatus.OK
    report = json.loads(body)
    assert report["coverage"]["status"] == "complete"
    assert len(report["events"]) == 2
    assert report["financials"]["revenue"]["value"] == "50677000000"
    assert report["financials"]["eps"]["value"] == "52.7"

    # Promoting against bundled read-only catalog must fail cleanly.
    payload = json.dumps({"report": report, "notice_date": "2025-03-19"}).encode("utf-8")
    code, _, body = dispatch_post("/api/catalog-add", {}, payload)
    assert code == HTTPStatus.BAD_REQUEST
    assert "writable local catalog" in json.loads(body)["error"]

    # Promoting against a local --catalog dir succeeds and becomes queryable via /api/actions.
    local_cat = tmp_path / "local-catalog"
    code, _, body = dispatch_post("/api/catalog-add", {}, payload, catalog_root=local_cat)
    assert code == HTTPStatus.OK
    assert json.loads(body)["total_notices"] == 2

    code, _, body = dispatch("/api/actions", {"as_of": ["2025-03-20"], "symbol": ["SCBK"]}, catalog_root=local_cat)
    assert code == HTTPStatus.OK
    events = {r["event"] for r in json.loads(body)["rows"]}
    assert events == {"SCBK-2024-final", "SCBK-2024-interim"}
