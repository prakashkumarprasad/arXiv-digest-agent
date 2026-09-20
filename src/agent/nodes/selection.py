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


# Minimum LLM relevance (0-10) a candidate needs to be digested. The composite score adds
# recency and full-text points that any recent paper earns (4 of 10 for free), so it cannot
# tell "relevant" from "newest": the floor is applied to the LLM relevance alone, before ranking.
MIN_LLM_RELEVANCE = 4.0


def select_paper(state: AgentState) -> dict[str, Any]:
    """Rank candidates and select the best relevant paper.

    - Score = 0.6 * llm_relevance + 0.25 * recency_decay + 0.15 * has_full_text
    - llm_relevance: batched LLM call for the top 10, judged against the user's own request
    - Candidates below MIN_LLM_RELEVANCE are dropped before ranking. If none remain, the node
      returns paper=None and a NO_RELEVANT_PAPER error, and the graph ends.
    - If LLM scoring itself fails, the floor is skipped (there is nothing to judge with).
    - Auto mode: picks rank 1.
    """
    candidates = state.get("candidates", [])
    if not candidates:
        return {
            "paper": None,
            "errors": [{"code": "NO_CANDIDATES", "node": "select_paper", "detail": "No candidates to select from", "recoverable": False}],
        }

    # Limit to top 10 for LLM scoring
    top_candidates = candidates[:10]
    request = state.get("raw_input") or state.get("search_query") or ""

    llm_scores = _get_llm_relevance(top_candidates, request)
    scoring_failed = llm_scores is None
    if scoring_failed:
        llm_scores = {i: 5.0 for i in range(len(top_candidates))}

    # Calculate composite scores
    scored_papers = []
    for i, candidate in enumerate(top_candidates):
        llm_score = llm_scores.get(i, 0)
        recency_score = _calculate_recency_score(candidate)
        full_text_score = _calculate_full_text_score(candidate)

        composite = 0.6 * llm_score + 0.25 * recency_score + 0.15 * full_text_score
        scored_papers.append((composite, i, candidate, llm_score, recency_score, full_text_score))

    if not scoring_failed:
        relevant = [p for p in scored_papers if p[3] >= MIN_LLM_RELEVANCE]
        if not relevant:
            best = max(scored_papers, key=lambda p: p[3])
            closest = str(best[2].get("title", "Unknown"))[:80]
            return {
                "paper": None,
                "selection_reason": f"No candidate reached the minimum relevance of {MIN_LLM_RELEVANCE:.0f}/10 (best: {best[3]:.0f}/10).",
                "errors": [{
                    "code": "NO_RELEVANT_PAPER",
                    "node": "select_paper",
                    "detail": (
                        f"Found {len(candidates)} paper(s), but none looks relevant to your request "
                        f"(best relevance {best[3]:.0f}/10, minimum {MIN_LLM_RELEVANCE:.0f}/10; "
                        f"closest match: '{closest}'). "
                        "Try more specific keywords, or give an arXiv ID directly."
                    ),
                    "recoverable": False,
                }],
            }
        scored_papers = relevant

    # Sort by composite score descending
    scored_papers.sort(key=lambda x: x[0], reverse=True)

    # Auto mode: pick rank 1
    selected_idx = scored_papers[0][1]
    selected_paper = scored_papers[0][2]
    selection_reason = _format_selection_reason(scored_papers, selected_idx)

    logger.info(f"Selected paper: {selected_paper.get('arxiv_id')} (score: {scored_papers[0][0]:.2f})")

    return {
        "paper": selected_paper,
        "selection_reason": selection_reason,
    }


def _get_llm_relevance(candidates: list[dict], query: str) -> dict[int, float] | None:
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
        logger.warning(f"LLM relevance scoring failed: {e}; the relevance floor is skipped")
        return None


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