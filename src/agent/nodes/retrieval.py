"""Retrieval nodes: search_arxiv, broaden_query, and fetch_metadata."""

import logging
import re
from typing import Any

import arxiv
from agent.config import get_settings
from agent.services.arxiv_client import get_arxiv_client
from agent.state import AgentState

logger = logging.getLogger(__name__)


def search_arxiv(state: AgentState) -> dict[str, Any]:
    """Search arXiv for papers matching the query.

    Implements the search_arxiv node from the graph.
    """
    search_query = state.get("search_query")
    if not search_query:
        logger.warning("search_arxiv called without search_query")
        return {"candidates": [], "errors": [{"code": "MISSING_QUERY", "node": "search_arxiv", "detail": "No search query provided", "recoverable": False}]}

    client = get_arxiv_client()

    try:
        candidates = client.search(search_query)
        candidate_dicts = [c.model_dump(mode="json") if hasattr(c, "model_dump") else c for c in candidates]
        logger.info(f"Found {len(candidate_dicts)} candidates for query: {search_query}")
        return {"candidates": candidate_dicts}

    except Exception as e:
        logger.error(f"arXiv search failed: {e}")
        error = {
            "code": "ARXIV_SEARCH_FAILED",
            "node": "search_arxiv",
            "detail": str(e),
            "recoverable": True,
        }
        return {"candidates": [], "errors": [error]}


def broaden_query(state: AgentState) -> dict[str, Any]:
    """Broaden the search query when zero results are returned.

    Implements the broaden_query node - attempts to relax the query
    up to 2 times before giving up.
    """
    retries = state.get("retries", {})
    broaden_attempts = retries.get("broaden_query", 0)

    if broaden_attempts >= 2:
        # Exhausted broadening attempts
        original_query = state.get("search_query", "")
        error = {
            "code": "ZERO_RESULTS_EXHAUSTED",
            "node": "broaden_query",
            "detail": f"No results after 2 broadening attempts. Original query: {original_query}",
            "recoverable": False,
        }
        return {
            "candidates": [],
            "errors": [error],
            "search_query": None,  # Signal termination
        }

    original_query = state.get("search_query", "")
    client = get_arxiv_client()

    if broaden_attempts == 0:
        # Attempt 1: drop quotes and cat: filter, AND -> OR
        new_query = _broaden_attempt_1(original_query)
        logger.info(f"Broadening query (attempt 1): {original_query} -> {new_query}")

    else:
        # Attempt 2: keep only two highest-IDF terms, widen sort_by=SubmittedDate
        new_query = _broaden_attempt_2(original_query)
        logger.info(f"Broadening query (attempt 2): {original_query} -> {new_query}")

    # Increment retry counter
    new_retries = dict(retries)
    new_retries["broaden_query"] = broaden_attempts + 1

    # Search with broadened query
    try:
        candidates = client.search(new_query, sort_by=arxiv.SortCriterion.SubmittedDate)
        candidate_dicts = [c.model_dump(mode="json") if hasattr(c, "model_dump") else c for c in candidates]
        if candidate_dicts:
            return {
                "candidates": candidate_dicts,
                "search_query": new_query,
                "retries": new_retries,
                "warnings": [f"Query broadened (attempt {broaden_attempts + 1}): {new_query}"],
            }
    except Exception as e:
        logger.error(f"Broadened search failed: {e}")

    # No results, continue to next attempt or exhaust
    return {
        "candidates": [],
        "search_query": new_query,
        "retries": new_retries,
        "warnings": [f"Broadening attempt {broaden_attempts + 1} yielded no results"],
    }


def _broaden_attempt_1(query: str) -> str:
    """First broadening: drop quotes and cat: filter, AND -> OR."""
    # Remove cat: filters
    query = re.sub(r"\s+cat:\w+", "", query)
    # Replace AND with OR
    query = query.replace(" AND ", " OR ")
    # Remove quotes
    query = query.replace('"', "")
    return query.strip()


def _broaden_attempt_2(query: str) -> str:
    """Second broadening: keep only two highest-IDF terms."""
    # Simple heuristic: take first two meaningful terms
    terms = query.split()
    # Filter out common stopwords and operators
    stopwords = {"the", "a", "an", "and", "or", "of", "in", "on", "for", "to", "with", "by", "is", "are", "was", "were"}
    meaningful_terms = [t for t in terms if t.lower() not in stopwords and len(t) > 2]

    # Take first two
    if len(meaningful_terms) >= 2:
        return f"all:{meaningful_terms[0]} OR all:{meaningful_terms[1]}"
    elif meaningful_terms:
        return f"all:{meaningful_terms[0]}"
    else:
        return query


def fetch_metadata(state: AgentState) -> dict[str, Any]:
    """Fetch metadata for a specific arXiv ID (paper_lookup intent)."""
    arxiv_id = state.get("arxiv_id")
    if not arxiv_id:
        return {"paper": None, "errors": [{"code": "MISSING_ARXIV_ID", "node": "fetch_metadata", "detail": "No arXiv ID provided", "recoverable": False}]}

    client = get_arxiv_client()

    try:
        paper = client.fetch_metadata(arxiv_id)
        if paper:
            # Convert Pydantic model to dict for state serialization
            paper_dict = paper.model_dump(mode="json") if hasattr(paper, 'model_dump') else paper
            return {"paper": paper_dict, "candidates": [paper_dict]}
        else:
            return {"paper": None, "candidates": [], "errors": [{"code": "ARXIV_ID_NOT_FOUND", "node": "fetch_metadata", "detail": f"Paper {arxiv_id} not found", "recoverable": False}]}
    except Exception as e:
        logger.error(f"Metadata fetch failed: {e}")
        error = {
            "code": "ARXIV_FETCH_FAILED",
            "node": "fetch_metadata",
            "detail": str(e),
            "recoverable": True,
        }
        return {"paper": None, "candidates": [], "errors": [error]}


