import hashlib

import pytest

from fingpt_research.filing_pdf import read_pdf


def _write_pdf(path, page_types):
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
    for text, raster in page_types:
        page = writer.add_blank_page(width=600, height=800)
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)}),
            NameObject("/XObject"): DictionaryObject({NameObject("/Im"): writer._add_object(image)})})
        content = DecodedStreamObject()
        commands = b"BT /F1 16 Tf 40 750 Td (Company results) Tj ET\n" if text else b""
        if raster:
            commands += b"q 550 0 0 650 10 10 cm /Im Do Q\n"
        content.set_data(commands)
        page[NameObject("/Contents")] = writer._add_object(content)
    writer.write(path)


def test_coverage_retains_text_on_mixed_pages_and_physical_page_numbers(tmp_path):
    pytest.importorskip("pypdf")
    pytest.importorskip("pypdfium2")
    path = tmp_path / "filing.pdf"
    _write_pdf(path, [(True, False), (False, True), (True, True)])
    pages, metadata = read_pdf(path)
    assert [s.strip() for s in pages] == ["Company results", "", "Company results"]
    assert metadata["pdf_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    coverage = metadata["coverage"]
    assert (coverage["status"], coverage["page_count"]) == ("partial", 3)
    assert [(p["page"], p["status"]) for p in coverage["pages"]] == [
        (1, "readable"), (2, "unreadable"), (3, "partial")]


def test_text_only_filing_and_invalid_input(tmp_path):
    pytest.importorskip("pypdf")
    pytest.importorskip("pypdfium2")
    path = tmp_path / "filing.pdf"
    _write_pdf(path, [(True, False)])
    pages, metadata = read_pdf(path)
    assert pages[0].strip() == "Company results"
    assert metadata["coverage"]["status"] == "complete"
    path.write_bytes(b"not a PDF")
    with pytest.raises(ValueError, match="could not open"):
        read_pdf(path)
