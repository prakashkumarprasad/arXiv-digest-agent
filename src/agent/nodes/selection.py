"""Selection node: rank and select paper from candidates."""

import logging
from typing import Any

from agent.config import get_settings
from agent.services.llm import complete_json
from agent.services.prompts import SYSTEM_SELECT_PAPER, select_paper_prompt
from agent.state import AgentState

logger = logging.getLogger(__name__)


class PaperScore:
    """Score for a single paper."""

    def __init__(self, index: int, score: int, reason: str):
        self.index = index
        self.score = score
        self.reason = reason


def select_paper(state: AgentState) -> dict[str, Any]:
    """Rank candidates and select the best paper.

    Implements the select_paper node from the graph.
    - Score = 0.6 * llm_relevance + 0.25 * recency_decay + 0.15 * has_full_text
    - llm_relevance: batched LLM call for top 10
    - Interactive mode: Rich table, --auto picks rank 1
    - Stores selection_reason in state
    """
    candidates = state.get("candidates", [])
    if not candidates:
        return {
            "paper": None,
            "errors": [{"code": "NO_CANDIDATES", "node": "select_paper", "detail": "No candidates to select from", "recoverable": False}],
        }

    # Limit to top 10 for LLM scoring
    top_candidates = candidates[:10]

    # Get LLM relevance scores
    llm_scores = _get_llm_relevance(top_candidates, state.get("search_query", ""))

    # Calculate composite scores
    scored_papers = []
    for i, candidate in enumerate(top_candidates):
        llm_score = llm_scores.get(i, 0)
        recency_score = _calculate_recency_score(candidate)
        full_text_score = _calculate_full_text_score(candidate)

        composite = 0.6 * llm_score + 0.25 * recency_score + 0.15 * full_text_score
        scored_papers.append((composite, i, candidate, llm_score, recency_score, full_text_score))

    # Sort by composite score descending
    scored_papers.sort(key=lambda x: x[0], reverse=True)

    # Auto mode: pick rank 1
    # In a real CLI, we'd check for --auto flag. For now, assume auto.
    selected_idx = scored_papers[0][1]
    selected_paper = scored_papers[0][2]
    selection_reason = _format_selection_reason(scored_papers, selected_idx)

    logger.info(f"Selected paper: {selected_paper.get('arxiv_id')} (score: {scored_papers[0][0]:.2f})")

    return {
        "paper": selected_paper,
        "selection_reason": selection_reason,
    }


def _get_llm_relevance(candidates: list[dict], query: str) -> dict[int, float]:
    """Get LLM relevance scores for top 10 candidates."""
    try:
        from pydantic import BaseModel

        from pydantic import RootModel

        class RelevanceScore(BaseModel):
            index: int
            score: int
            reason: str

        class RelevanceScores(RootModel[list[RelevanceScore]]):
            pass

        result = complete_json(
            RelevanceScores,
            SYSTEM_SELECT_PAPER,
            select_paper_prompt(query, candidates),
        )

        scores = {}
        for item in result.root:
            scores[item.index] = float(item.score)
        return scores
    except Exception as e:
        logger.warning(f"LLM relevance scoring failed: {e}, using default scores")
        # Default: all get 5/10
        return {i: 5.0 for i in range(len(candidates))}


def _calculate_recency_score(candidate: dict) -> float:
    """Calculate recency score (0-10) based on publication date."""
    from datetime import datetime

    published_str = candidate.get("published", "")
    if not published_str:
        return 5.0

    try:
        published = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
        now = datetime.now(published.tzinfo)
        days_old = (now - published).days

        # Exponential decay: papers within 30 days get ~10, 1 year gets ~5, 5+ years gets ~1
        if days_old <= 30:
            return 10.0
        elif days_old <= 365:
            return 7.0
        elif days_old <= 365 * 3:
            return 4.0
        else:
            return 1.0
    except Exception:
        return 5.0


def _calculate_full_text_score(candidate: dict) -> float:
    """Calculate full text availability score (0-10)."""
    # Check if we can likely get full text (has PDF URL)
    if candidate.get("pdf_url"):
        return 10.0
    return 5.0


def _format_selection_reason(scored_papers: list[tuple], selected_idx: int) -> str:
    """Format human-readable selection reason."""
    if not scored_papers:
        return "No candidates available"

    top = scored_papers[0]
    reason = f"Selected rank 1 (composite score: {top[0]:.2f}) - {top[2].get('title', 'Unknown')[:80]}"
    reason += f" | LLM relevance: {top[3]:.1f}/10, Recency: {top[4]:.1f}/10, Full-text: {top[5]:.1f}/10"
    return reason