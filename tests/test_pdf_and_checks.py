import io

import pytest
from conftest import SAMPLES

from assistant.checks import check_limits, check_registers
from assistant.errors import PdfError
from assistant.pdf import read_pdf
from assistant.schemas import Limits, RegisterMap


def test_sample_pdf_reads_with_table_rows_kept():
    pages = read_pdf((SAMPLES / "DX-IMU6_datasheet.pdf").read_bytes())
    assert len(pages) == 8
    assert "0x6B | PWR_MGMT_1 | RW | 0x40" in pages[4].text


@pytest.mark.parametrize("data", [b"", b"hello, not a pdf"])
def test_bad_files(data):
    with pytest.raises(PdfError):
        read_pdf(data)


def test_scanned_pdf_without_text():
    from reportlab.pdfgen import canvas
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.rect(100, 100, 300, 300, fill=1)           # an image-like page with no text
    c.showPage()
    c.save()
    with pytest.raises(PdfError, match="scanned"):
        read_pdf(buf.getvalue())


def rmap(*regs):
    return RegisterMap(device="D", registers=[dict(name=n, address=a, access=acc, reset="0x00", fields=f) for n, a, acc, f in regs])


def test_register_checks():
    issues = check_registers(rmap(("A", "0x10", "RW", [{"name": "X", "bits": "7:4"}, {"name": "Y", "bits": "4"}]),
                                  ("B", "0x10", "R", [{"name": "Z", "bits": "0:2"}])))
    text = " ".join(issues)
    assert "also used by A" in text and "overlap" in text and "low:high" in text


def test_clean_register_map_passes():
    assert check_registers(rmap(("A", "0x10", "RW", [{"name": "X", "bits": "7:4"}, {"name": "Y", "bits": "3:0"}]))) == []


def test_limit_checks():
    lims = Limits(limits=[{"parameter": "VDD", "min": 3.6, "typ": 3.3, "max": 1.8, "unit": "V"},
                          {"parameter": "I", "typ": 3, "unit": "furlongs"}])
    text = " ".join(check_limits(lims))
    assert "order" in text and "unusual unit" in text


def test_register_name_must_be_c_identifier():
    with pytest.raises(ValueError):
        RegisterMap(device="D", registers=[{"name": "1BAD", "address": "0x01", "access": "R", "reset": "0x00"}])
