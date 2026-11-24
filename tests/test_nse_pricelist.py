import pytest

from fingpt_research.cli import get_parser, run as run_cli
from fingpt_research.data import load_dataset
from fingpt_research.nse_pricelist import extract_session_date, import_nse_pricelist, parse_nse_pricelist_text


def _write_text_pdf(path, lines):
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                             NameObject("/Subtype"): NameObject("/Type1"),
                             NameObject("/BaseFont"): NameObject("/Helvetica")})
    page = writer.add_blank_page(width=612, height=792)
    page[NameObject("/Resources")] = DictionaryObject({
        NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})})
    commands = [b"BT /F1 10 Tf"]
    y = 740
    for line in lines:
        escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        commands.append(f"1 0 0 1 30 {y} Tm ({escaped}) Tj".encode("latin1"))
        y -= 16
    commands.append(b"ET")
    stream = DecodedStreamObject()
    stream.set_data(b"\n".join(commands))
    page[NameObject("/Contents")] = writer._add_object(stream)
    writer.write(path)


def test_parse_nse_pricelist_text_handles_traded_and_untraded_rows():
    pages = [
        "NAIROBI SECURITIES EXCHANGE DAILY EQUITY PRICE LIST\nDate: 02-SEP-26\n"
        "SECURITY                  ISIN          HIGH   LOW    VWAP   PREV   VOLUME\n"
        "Safaricom Plc             KE1000001402  39.00  38.10  38.65  38.50  6,097,383\n"
        "Standard Chartered Bank   KE0000000448  -      -      -      215.00 -\n"
    ]
    assert extract_session_date(pages) == "2026-09-02"
    scom = parse_nse_pricelist_text(pages, "SCOM")
    assert (scom["session"], scom["high"], scom["low"], scom["close"], scom["volume"]) == (
        "2026-09-02", "39.00", "38.10", "38.65", "6097383")
    assert scom["open"] == ""
    assert scom["price_type"] == "vwap"

    scbk = parse_nse_pricelist_text(pages, "SCBK")
    assert (scbk["close"], scbk["high"], scbk["low"], scbk["volume"]) == ("215.00", "215.00", "215.00", "0")


def test_nse_import_cli_builds_valid_multi_day_dataset_for_replay(tmp_path):
    pytest.importorskip("pypdf")
    pytest.importorskip("pypdfium2")
    pdfs = tmp_path / "lists"
    pdfs.mkdir()
    _write_text_pdf(pdfs / "2026-09-02.pdf", [
        "NAIROBI SECURITIES EXCHANGE DAILY EQUITY PRICE LIST",
        "Date: 02-SEP-26",
        "Safaricom Plc KE1000001402 39.00 38.10 38.65 38.50 6,097,383",
    ])
    _write_text_pdf(pdfs / "2026-09-03.pdf", [
        "NAIROBI SECURITIES EXCHANGE DAILY EQUITY PRICE LIST",
        "Date: 03-SEP-26",
        "Safaricom Plc KE1000001402 39.50 38.60 39.10 38.65 4,120,000",
    ])
    out = tmp_path / "scom-dataset"
    summary = run_cli(get_parser().parse_args([
        "nse-import", "--pdf", str(pdfs), "--symbol", "SCOM", "--out", str(out),
    ]))
    assert summary["sessions_total"] == 2
    assert summary["instrument"] == "XNAI:KE1000001402"

    # Re-importing the same directory is idempotent and passes strict load_dataset + replay.
    summary2 = import_nse_pricelist(pdfs, "SCOM", out, ingested_at="2026-09-04T00:00:00Z")
    assert summary2["sessions_total"] == 2
    dataset = load_dataset(out)
    assert [b.session for b in dataset.bars] == ["2026-09-02", "2026-09-03"]
    replay = run_cli(get_parser().parse_args(["replay", "--dataset", str(out)]))
    assert len(replay["observations"]) == 2


def test_nse_import_rejects_image_only_scanned_pdf(tmp_path):
    pytest.importorskip("pypdf")
    pytest.importorskip("pypdfium2")
    from pypdf import PdfWriter

    scanned = tmp_path / "02-SEP-26.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.write(scanned)
    with pytest.raises(ValueError, match="no selectable text layer"):
        import_nse_pricelist(scanned, "SCOM", tmp_path / "out")
