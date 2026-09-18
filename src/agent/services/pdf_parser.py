"""PDF parsing with PyMuPDF primary and pdfplumber fallback."""

import logging
import re
from pathlib import Path
from typing import Any, Literal
import difflib

import pymupdf as fitz  # PyMuPDF
import pdfplumber

from agent.config import get_settings
from agent.models import ParseMode

logger = logging.getLogger(__name__)

# Section heading patterns (case-insensitive)
SECTION_PATTERNS = [
    r"^\s*(\d+\.?\d*)?\s*Abstract\b",
    r"^\s*(\d+\.?\d*)?\s*Introduction\b",
    r"^\s*(\d+\.?\d*)?\s*Related Work\b",
    r"^\s*(\d+\.?\d*)?\s*Background\b",
    r"^\s*(\d+\.?\d*)?\s*Method\b",
    r"^\s*(\d+\.?\d*)?\s*Methodology\b",
    r"^\s*(\d+\.?\d*)?\s*Approach\b",
    r"^\s*(\d+\.?\d*)?\s*Experiments?\b",
    r"^\s*(\d+\.?\d*)?\s*Results?\b",
    r"^\s*(\d+\.?\d*)?\s*Evaluation\b",
    r"^\s*(\d+\.?\d*)?\s*Discussion\b",
    r"^\s*(\d+\.?\d*)?\s*Limitations?\b",
    r"^\s*(\d+\.?\d*)?\s*Conclusion\b",
    r"^\s*(\d+\.?\d*)?\s*References\b",
    r"^\s*(\d+\.?\d*)?\s*Appendix\b",
]

SECTION_REGEX = re.compile("|".join(SECTION_PATTERNS), re.IGNORECASE | re.MULTILINE)


def parse_pdf(
    pdf_path: str,
    arxiv_id: str,
    abstract_fallback: str,
) -> tuple[list[dict], str, ParseMode]:
    """Parse PDF with PyMuPDF, fallback to pdfplumber, degrade to abstract-only.

    Returns (sections, full_text, parse_mode).
    sections: [{"title": str, "text": str, "page_start": int}]
    Never raises on a bad PDF — degrades to parse_mode="degraded_abstract_only".
    """
    pdf_path_obj = Path(pdf_path)

    # Try PyMuPDF first
    sections, full_text, parse_mode, warnings = _parse_with_pymupdf(pdf_path_obj, abstract_fallback)
    if parse_mode == "full":
        return sections, full_text, parse_mode

    # Fallback to pdfplumber
    logger.warning("PyMuPDF parsing degraded, trying pdfplumber fallback")
    sections, full_text, parse_mode, warnings = _parse_with_pdfplumber(pdf_path_obj, abstract_fallback)
    if parse_mode == "full":
        return sections, full_text, parse_mode

    # Both failed - degrade to abstract-only
    logger.warning("Both parsers failed, degrading to abstract-only mode")
    sections = [{"title": "Abstract", "text": abstract_fallback, "page_start": 0}]
    return sections, abstract_fallback, "degraded_abstract_only"

def _dedupe_repeated_blocks(text: str, min_block_chars: int = 60) -> str:
    """Strip verbatim-repeated blocks caused by PDF producers that emit the
    same content twice (common in LaTeX theorem/box environments in
    two-column layouts). Keeps the first occurrence, drops later repeats."""
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) < min_block_chars * 2:
        return text

    matcher = difflib.SequenceMatcher(None, normalized, normalized, autojunk=False)
    dupes = [m for m in matcher.get_matching_blocks()
             if m.size >= min_block_chars and m.a != m.b]
    if not dupes:
        return text

    remove_ranges = []
    for m in dupes:
        first, second = sorted([m.a, m.b])
        remove_ranges.append((second, second + m.size))
    remove_ranges.sort()

    merged = []
    for start, end in remove_ranges:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    out, last = [], 0
    for start, end in merged:
        out.append(normalized[last:start])
        last = end
    out.append(normalized[last:])
    return " ".join(p for p in out if p).strip()

def _parse_with_pymupdf(pdf_path: Path, abstract_fallback: str) -> tuple[list[dict], str, ParseMode, list[str]]:
    """Parse PDF using PyMuPDF."""
    warnings = []
    doc = None

    try:
        doc = fitz.open(pdf_path)
        pages_text = []
        total_chars = 0
        total_alpha = 0
        total_chars_all = 0

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = _dedupe_repeated_blocks(page.get_text("text", sort=True))
            pages_text.append(text)

            # Quality metrics
            chars = len(text)
            alpha_chars = sum(1 for c in text if c.isalpha())
            total_chars += chars
            total_alpha += alpha_chars
            total_chars_all += chars

        full_text = "\n".join(pages_text)

        # Quality gate
        num_pages = len(doc)
        chars_per_page = total_chars / num_pages if num_pages > 0 else 0
        alpha_ratio = total_alpha / total_chars_all if total_chars_all > 0 else 0

        if chars_per_page < 120 or alpha_ratio < 0.6:
            warnings.append(
                f"Low text quality: chars_per_page={chars_per_page:.1f}, "
                f"alpha_ratio={alpha_ratio:.2f}"
            )
            return [], "", "failed", warnings

        # Extract sections
        sections = _extract_sections_pymupdf(doc, full_text)

        # Truncate References section
        for section in sections:
            if section["title"].lower() == "references":
                section["text"] = section["text"][:4000]

        return sections, full_text, "full", warnings

    except Exception as e:
        logger.error(f"PyMuPDF parsing failed: {e}")
        return [], "", "failed", [f"PyMuPDF error: {e}"]

    finally:
        if doc:
            doc.close()


def _parse_with_pdfplumber(pdf_path: Path, abstract_fallback: str) -> tuple[list[dict], str, ParseMode, list[str]]:
    """Parse PDF using pdfplumber as fallback."""
    warnings = []

    try:
        pages_text = []
        total_chars = 0
        total_alpha = 0
        total_chars_all = 0

        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = _dedupe_repeated_blocks(page.extract_text() or "")
                pages_text.append(text)

                chars = len(text)
                alpha_chars = sum(1 for c in text if c.isalpha())
                total_chars += chars
                total_alpha += alpha_chars
                total_chars_all += chars

        full_text = "\n".join(pages_text)

        # Quality gate
        num_pages = len(pages_text)
        chars_per_page = total_chars / num_pages if num_pages > 0 else 0
        alpha_ratio = total_alpha / total_chars_all if total_chars_all > 0 else 0

        if chars_per_page < 120 or alpha_ratio < 0.6:
            warnings.append(
                f"Low text quality (pdfplumber): chars_per_page={chars_per_page:.1f}, "
                f"alpha_ratio={alpha_ratio:.2f}"
            )
            return [], "", "failed", warnings

        # Extract sections using regex (no font info in pdfplumber)
        sections = _extract_sections_regex(full_text)

        # Truncate References section
        for section in sections:
            if section["title"].lower() == "references":
                section["text"] = section["text"][:4000]

        return sections, full_text, "full", warnings

    except Exception as e:
        logger.error(f"pdfplumber parsing failed: {e}")
        return [], "", "failed", [f"pdfplumber error: {e}"]


def _extract_sections_pymupdf(doc: fitz.Document, full_text: str) -> list[dict[str, Any]]:
    """Extract sections using PyMuPDF with font-size heuristic."""
    # First try regex-based extraction
    sections = _extract_sections_regex(full_text)

    # If we got good sections, enhance with page info from PyMuPDF
    if len(sections) > 1:
        sections = _add_page_numbers_pymupdf(doc, sections)
        return sections

    # Fallback: try font-size based detection
    return _extract_sections_font_size(doc)


def _extract_sections_regex(full_text: str) -> list[dict[str, Any]]:
    """Extract sections using regex pattern matching."""
    matches = list(SECTION_REGEX.finditer(full_text))

    if not matches:
        return [{"title": "Body", "text": full_text.strip(), "page_start": 0}]

    sections = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        title = match.group(0).strip()
        text = full_text[start:end].strip()

        # Clean up title (remove numbering)
        title = re.sub(r"^\s*\d+\.?\d*\s*", "", title).strip()

        sections.append({"title": title, "text": text, "page_start": 0})

    return sections


def _add_page_numbers_pymupdf(doc: fitz.Document, sections: list[dict]) -> list[dict]:
    """Add page_start to sections by searching for section titles in pages."""
    for section in sections:
        title = section["title"]
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            if title.lower() in text.lower():
                section["page_start"] = page_num + 1  # 1-indexed
                break
    return sections


def _extract_sections_font_size(doc: fitz.Document) -> list[dict[str, Any]]:
    """Extract sections using font-size heuristic from PyMuPDF dict output."""
    sections = []
    current_section = {"title": "Body", "text": "", "page_start": 1}
    current_title = "Body"

    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if "lines" not in block:
                continue

            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if not text:
                        continue

                    font_size = span["size"]
                    is_heading = font_size > 12 and len(text) < 100

                    if is_heading and SECTION_REGEX.search(text):
                        # Save previous section
                        if current_section["text"].strip():
                            sections.append(current_section)

                        # Start new section
                        current_title = text
                        current_section = {
                            "title": current_title,
                            "text": "",
                            "page_start": page_num + 1,
                        }
                    else:
                        current_section["text"] += text + " "

    # Add final section
    if current_section["text"].strip():
        sections.append(current_section)

    return sections if sections else [{"title": "Body", "text": "", "page_start": 1}]


def download_pdf(arxiv_id: str, pdf_url: str) -> str:
    """Download PDF to cache directory.

    Returns the local path to the downloaded PDF.
    """
    import httpx

    settings = get_settings()
    cache_dir = settings.pdf_cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize arxiv_id for filename
    safe_id = arxiv_id.replace("/", "_").replace(".", "_")
    pdf_path = cache_dir / f"{safe_id}.pdf"

    if pdf_path.exists():
        logger.info(f"PDF already cached: {pdf_path}")
        return str(pdf_path)

    logger.info(f"Downloading PDF from {pdf_url}")
    try:
        # Convert HttpUrl to string if needed
        pdf_url_str = str(pdf_url)
        with httpx.stream("GET", pdf_url_str, timeout=60.0, follow_redirects=True) as response:
            response.raise_for_status()

            # Check content length
            content_length = response.headers.get("content-length")
            if content_length:
                size_mb = int(content_length) / (1024 * 1024)
                if size_mb > 100:
                    logger.warning(f"PDF size {size_mb:.1f} MB exceeds 100 MB limit")
                elif size_mb > 40:
                    logger.warning(f"PDF size {size_mb:.1f} MB exceeds 40 MB warning threshold")

            with open(pdf_path, "wb") as f:
                for chunk in response.iter_bytes(chunk_size=8192):
                    f.write(chunk)

        logger.info(f"PDF downloaded to {pdf_path}")
        return str(pdf_path)

    except Exception as e:
        logger.error(f"PDF download failed: {e}")
        raise