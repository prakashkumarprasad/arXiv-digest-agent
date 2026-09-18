"""Query understanding node: detect arXiv ID or normalize topic to search query."""

import logging
import re
from typing import Any

from agent.services.arxiv_client import get_arxiv_client
from agent.services.llm import complete_json
from agent.services.prompts import SYSTEM_QUERY_NORMALIZE, query_normalize_prompt
from agent.state import AgentState
from agent.models import ParseMode

logger = logging.getLogger(__name__)

# Regex patterns for arXiv ID detection
ARXIV_ID_PATTERN = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")
OLD_ARXIV_PATTERN = re.compile(r"([a-z\-]+(\.[A-Z]{2})?/\d{7})")


def query_understanding(state: AgentState) -> dict[str, Any]:
    """Determine user intent and extract/normalize query.

    Implements the query_understanding node from the graph.
    - Regex first (deterministic, zero cost) for arXiv ID detection
    - If no match → intent="topic_search"; ask LLM to normalize
    - If LLM fails → fall back to raw string
    """
    raw_input = state.get("raw_input", "").strip()
    if not raw_input:
        return {
            "intent": "unclear",
            "errors": [{"code": "EMPTY_INPUT", "node": "query_understanding", "detail": "No input provided", "recoverable": False}],
        }

    # Try to extract arXiv ID from input (also matches inside URLs)
    arxiv_id = _extract_arxiv_id(raw_input)
    if arxiv_id:
        logger.info(f"Detected arXiv ID: {arxiv_id}")
        return {"intent": "paper_lookup", "arxiv_id": arxiv_id}

    # No arXiv ID found - treat as topic search
    logger.info("No arXiv ID detected, treating as topic search")
    search_query = _normalize_query_with_llm(raw_input)

    return {"intent": "topic_search", "search_query": search_query}


def _extract_arxiv_id(text: str) -> str | None:
    """Extract arXiv ID from text using regex patterns."""
    # Try new format: YYMM.NNNNN(vN)?
    match = ARXIV_ID_PATTERN.search(text)
    if match:
        return match.group(1)  # Return without version suffix

    # Try old format: category/NNNNNNN
    match = OLD_ARXIV_PATTERN.search(text)
    if match:
        return match.group(1)

    return None


def _normalize_query_with_llm(raw_input: str) -> str:
    """Use LLM to normalize user query into arXiv-friendly search query.

    Falls back to raw string if LLM fails.
    """
    try:
        # Define schema for the normalized query
        from pydantic import BaseModel

        class NormalizedQuery(BaseModel):
            query: str
            categories: list[str] = []

        result = complete_json(
            NormalizedQuery,
            SYSTEM_QUERY_NORMALIZE,
            query_normalize_prompt(raw_input),
        )
        # Build arXiv query from extracted terms and categories
        return _build_arxiv_query(result.query, result.categories)
    except Exception as e:
        logger.warning(f"LLM query normalization failed: {e}, falling back to raw input")
        # Fallback: use raw input as-is, but try to build a reasonable query
        return _build_arxiv_query(raw_input, None)


def _build_arxiv_query(query: str, categories: list[str] | None) -> str:
    """Build arXiv search query from terms and optional categories."""
    # Split query into terms and wrap in all:"..."
    STOPWORDS = {"the", "and", "for", "with", "from", "into", "onto", "via"}
    terms = [t.strip() for t in query.split() if len(t.strip()) > 2 and t.strip().lower() not in STOPWORDS]
    if not terms:
        return query

    # Build query with AND between terms
    query_parts = [f'all:"{term}"' for term in terms]
    arxiv_query = " AND ".join(query_parts)

    if categories:
        cat_filters = " OR ".join([f"cat:{cat}" for cat in categories])
        arxiv_query = f"({arxiv_query}) AND ({cat_filters})"

    return arxiv_query