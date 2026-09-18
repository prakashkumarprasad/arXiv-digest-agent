"""Fetch and parse nodes: fetch_pdf and parse."""

import logging
from typing import Any

from agent.config import get_settings
from agent.services.pdf_parser import download_pdf, parse_pdf
from agent.state import AgentState

logger = logging.getLogger(__name__)


def fetch_pdf(state: AgentState) -> dict[str, Any]:
    """Download PDF for the selected paper.

    Implements the fetch_pdf node from the graph.
    Caches at .data/pdfs/{arxiv_id}.pdf; skips if present.
    60s timeout, 2 retries, size guard (>40MB warn, >100MB degrade).
    """
    paper = state.get("paper")
    if not paper:
        return {"pdf_path": None, "errors": [{"code": "NO_PAPER", "node": "fetch_pdf", "detail": "No paper selected", "recoverable": False}]}

    arxiv_id = paper.get("arxiv_id")
    pdf_url = paper.get("pdf_url")

    if not arxiv_id or not pdf_url:
        return {"pdf_path": None, "errors": [{"code": "MISSING_PDF_INFO", "node": "fetch_pdf", "detail": "Paper missing arxiv_id or pdf_url", "recoverable": False}]}

    # Check if we've retried too many times
    retries = state.get("retries", {})
    fetch_attempts = retries.get("fetch_pdf", 0)

    if fetch_attempts >= 2:
        return {"pdf_path": None, "errors": [{"code": "PDF_FETCH_EXHAUSTED", "node": "fetch_pdf", "detail": "Max retries exceeded for PDF download", "recoverable": False}]}

    try:
        pdf_path = download_pdf(arxiv_id, pdf_url)

        # Check file size
        import os
        size_mb = os.path.getsize(pdf_path) / (1024 * 1024)

        warnings = []
        if size_mb > 100:
            warnings.append(f"PDF size {size_mb:.1f} MB exceeds 100 MB limit - may degrade parsing")
        elif size_mb > 40:
            warnings.append(f"PDF size {size_mb:.1f} MB exceeds 40 MB warning threshold")

        return {
            "pdf_path": pdf_path,
            "warnings": warnings,
        }

    except Exception as e:
        logger.error(f"PDF download failed: {e}")
        # Increment retry counter
        new_retries = dict(retries)
        new_retries["fetch_pdf"] = fetch_attempts + 1

        error = {
            "code": "PDF_DOWNLOAD_FAILED",
            "node": "fetch_pdf",
            "detail": str(e),
            "recoverable": True,
        }
        return {"pdf_path": None, "errors": [error], "retries": new_retries}


def parse(state: AgentState) -> dict[str, Any]:
    """Parse the downloaded PDF.

    Implements the parse node from the graph.
    Quality gate: chars_per_page < 120 OR alpha_ratio < 0.6 -> degraded mode.
    """
    pdf_path = state.get("pdf_path")
    paper = state.get("paper")

    if not pdf_path:
        return {"parse_mode": "failed", "errors": [{"code": "NO_PDF_PATH", "node": "parse", "detail": "No PDF path provided", "recoverable": False}]}

    arxiv_id = paper.get("arxiv_id", "") if paper else ""
    arxiv_abstract = paper.get("summary", "") if paper else ""

    try:
        sections, full_text, parse_mode = parse_pdf(pdf_path, arxiv_id, arxiv_abstract)

        return {
            "sections": sections,
            "full_text": full_text,
            "parse_mode": parse_mode,
            "warnings": [],
        }

    except Exception as e:
        logger.error(f"PDF parsing failed: {e}")
        error = {
            "code": "PDF_PARSE_FAILED",
            "node": "parse",
            "detail": str(e),
            "recoverable": False,
        }
        return {"parse_mode": "failed", "errors": [error]}


def degrade_mode(state: AgentState) -> dict[str, Any]:
    """Handle degraded parsing mode (abstract-only).

    This node is entered when parse quality is insufficient.
    It ensures the state has at least the abstract for downstream processing.
    """
    paper = state.get("paper")
    arxiv_abstract = paper.get("summary", "") if paper else ""

    if not arxiv_abstract:
        return {"parse_mode": "failed", "errors": [{"code": "NO_ABSTRACT", "node": "degrade_mode", "detail": "No abstract available for degraded mode", "recoverable": False}]}

    warnings = state.get("warnings", [])
    warnings.append("Using abstract-only mode due to PDF parsing failure")

    return {
        "sections": [{"title": "Abstract", "text": arxiv_abstract, "page_start": 0}],
        "full_text": arxiv_abstract,
        "parse_mode": "degraded_abstract_only",
        "warnings": warnings,
    }