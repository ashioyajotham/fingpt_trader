import hashlib

import pytest

from fingpt_research.filing_pdf import read_pdf


def _write_pdf(path, page_specs):
    """page_specs: list of (text_str, raster_rect_or_None) where rect is (w, h, x, y)."""
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject, NumberObject

    writer = PdfWriter()
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                             NameObject("/Subtype"): NameObject("/Type1"),
                             NameObject("/BaseFont"): NameObject("/Helvetica")})
    image = DecodedStreamObject()
    image.set_data(b"\x00\xff\xff\x00")
    image.update({NameObject("/Type"): NameObject("/XObject"),
                  NameObject("/Subtype"): NameObject("/Image"),
                  NameObject("/Width"): NumberObject(2), NameObject("/Height"): NumberObject(2),
                  NameObject("/ColorSpace"): NameObject("/DeviceGray"),
                  NameObject("/BitsPerComponent"): NumberObject(8)})
    for text, raster in page_specs:
        page = writer.add_blank_page(width=600, height=800)
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)}),
            NameObject("/XObject"): DictionaryObject({NameObject("/Im"): writer._add_object(image)})})
        content = DecodedStreamObject()
        commands = b""
        if text:
            lines = text if isinstance(text, list) else [text]
            y = 750
            for line in lines:
                escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
                commands += f"BT /F1 12 Tf 40 {y} Td ({escaped}) Tj ET\n".encode("latin1")
                y -= 18
        if raster:
            w, h, x, y = raster
            commands += f"q {w} 0 0 {h} {x} {y} cm /Im Do Q\n".encode("ascii")
        content.set_data(commands)
        page[NameObject("/Contents")] = writer._add_object(content)
    writer.write(path)


def test_coverage_retains_text_on_mixed_pages_and_physical_page_numbers(tmp_path):
    pytest.importorskip("pypdf")
    pytest.importorskip("pypdfium2")
    path = tmp_path / "filing.pdf"
    _write_pdf(path, [
        ("Company results", None),
        ("", (550, 650, 10, 10)),
        ("Company results", (550, 650, 10, 10)),
    ])
    pages, metadata = read_pdf(path)
    assert [s.strip() for s in pages] == ["Company results", "", "Company results"]
    assert metadata["pdf_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    coverage = metadata["coverage"]
    assert (coverage["status"], coverage["page_count"]) == ("partial", 3)
    assert [(p["page"], p["status"]) for p in coverage["pages"]] == [
        (1, "readable"), (2, "unreadable"), (3, "partial")]


def test_text_rich_page_with_moderate_chart_stays_readable(tmp_path):
    pytest.importorskip("pypdf")
    pytest.importorskip("pypdfium2")
    path = tmp_path / "press_release.pdf"
    long_text = [
        "Standard Chartered Bank Kenya Limited announces audited full year results",
        "for the year ended 31 December 2024 with profit after tax rising strongly",
        "and a recommended final dividend of KES 37 per share for shareholders.",
    ]
    # 200x200 decorative chart (~8.3% of 600x800 page) on a text-rich page should not force partial.
    _write_pdf(path, [(long_text, (200, 200, 200, 300))])
    _, metadata = read_pdf(path)
    assert metadata["coverage"]["status"] == "complete"
    assert metadata["coverage"]["pages"][0]["status"] == "readable"


def test_text_only_filing_and_invalid_input(tmp_path):
    pytest.importorskip("pypdf")
    pytest.importorskip("pypdfium2")
    path = tmp_path / "filing.pdf"
    _write_pdf(path, [("Company results", None)])
    pages, metadata = read_pdf(path)
    assert pages[0].strip() == "Company results"
    assert metadata["coverage"]["status"] == "complete"
    path.write_bytes(b"not a PDF")
    with pytest.raises(ValueError, match="could not open"):
        read_pdf(path)
