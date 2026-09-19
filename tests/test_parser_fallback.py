"""Tests for parse_pdf fallback: corrupt PDF degrades to abstract-only, no exception."""

import pytest


class TestParserFallback:
    def test_corrupt_pdf_returns_degraded_abstract_only(self, isolated_settings, tmp_path):
        """A corrupt/garbage PDF file must return degraded_abstract_only with full_text seeded from abstract, no exception."""
        from agent.services.pdf_parser import parse_pdf

        corrupt_pdf = tmp_path / "corrupt.pdf"
        corrupt_pdf.write_bytes(b"This is not a valid PDF file. Just garbage data that cannot be parsed by PyMuPDF or pdfplumber.")

        abstract_text = "This is the abstract fallback text for a paper about test topics."
        sections, full_text, parse_mode = parse_pdf(str(corrupt_pdf), "2401.12345", abstract_text)

        assert parse_mode == "degraded_abstract_only"
        assert full_text == abstract_text
        assert len(sections) > 0
        assert sections[0]["title"] == "Abstract"
        assert sections[0]["text"] == abstract_text

    def test_corrupt_pdf_no_exception_raised(self, isolated_settings, tmp_path):
        """parse_pdf must not raise any exception on a corrupt PDF."""
        from agent.services.pdf_parser import parse_pdf

        corrupt_pdf = tmp_path / "garbage.pdf"
        corrupt_pdf.write_bytes(bytes(range(256)) * 10)

        abstract_text = "Fallback abstract."
        try:
            result = parse_pdf(str(corrupt_pdf), "2401.12345", abstract_text)
            assert isinstance(result, tuple)
            assert len(result) == 3
            assert result[2] == "degraded_abstract_only"
        except Exception as e:
            pytest.fail(f"parse_pdf raised an exception on corrupt PDF: {e}")
