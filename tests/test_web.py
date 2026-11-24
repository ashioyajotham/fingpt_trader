import json
from http import HTTPStatus

import pytest

from asof.web import dispatch, dispatch_post, serve


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
        "app": "AsOf", "broker_connected": False, "catalog_writable": False,
        "mode": "paper-only", "universe_configured": False,
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


def test_web_universe_financials_factors_and_loopback_guard(tmp_path):
    from asof.financials import promote_financials
    from asof.nse_pricelist import import_nse_universe

    csv_path = tmp_path / "uni.csv"
    csv_path.write_text(
        "Date,Code,Name,Day Low,Day High,Day Price,Previous,Volume\n"
        "2025-01-02,SCOM,Safaricom Plc,17.00,17.50,17.25,17.10,5000000\n"
        "2025-01-03,SCOM,Safaricom Plc,17.20,17.80,17.60,17.25,4100000\n"
        "2025-01-02,EQTY,Equity Group,47.00,48.50,48.00,47.50,1300000\n"
        "2025-01-03,EQTY,Equity Group,48.00,49.00,48.50,48.00,1200000\n"
        "2025-01-02,KCB,KCB Group,34.00,35.00,34.50,34.00,900000\n"
        "2025-01-03,KCB,KCB Group,34.50,35.50,35.00,34.50,850000\n"
    )
    uni_dir = tmp_path / "universe"
    import_nse_universe(csv_path, uni_dir)
    cat_dir = tmp_path / "catalog"
    promote_financials({
        "symbol": "SCOM", "source": "https://example.test/scom.pdf", "pdf_sha256": "abc",
        "financials": {"eps": {"value": "2.0", "raw_value": "2.0", "currency": "KES",
                               "unit": "KES/share", "scale": "1", "period": "2024",
                               "citations": [{"page": 1, "quote": "EPS 2.00"}]}},
    }, "2025-01-01", cat_dir)

    code, _, body = dispatch("/api/universe", {}, catalog_root=cat_dir, universe_root=uni_dir)
    assert code == HTTPStatus.OK
    uni = json.loads(body)
    assert uni["tickers_count"] == 3

    code, _, body = dispatch("/api/financials", {"as_of": ["2025-01-05"], "symbol": ["SCOM"]}, catalog_root=cat_dir)
    assert code == HTTPStatus.OK
    assert json.loads(body)["rows"][0]["metrics"]["eps"]["value"] == "2.0"

    code, _, body = dispatch("/api/factors", {"horizons": ["1"]}, catalog_root=cat_dir, universe_root=uni_dir)
    assert code == HTTPStatus.OK
    assert "liquidity_adv20" in json.loads(body)["factors"]

    code, _, body = dispatch(
        "/api/factor-backtest",
        {"factor_blend": ["liquidity_adv20:1.0"], "direction": ["top"], "top_n": ["2"],
         "rebalance": ["daily"], "max_sector_weight": ["0.6"]},
        catalog_root=cat_dir, universe_root=uni_dir,
    )
    assert code == HTTPStatus.OK
    fbt = json.loads(body)
    assert fbt["mode"] == "factor_backtest" and fbt["factor"] == "composite" and fbt["trades_count"] >= 1
    assert "yearly" in fbt and "sector_exposure" in fbt

    # Non-loopback host binding must be rejected unless allow_remote=True.
    with pytest.raises(ValueError, match="non-loopback"):
        serve(host="0.0.0.0", port=0)
