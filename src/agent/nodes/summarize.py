"""Summarize node: map-reduce briefing generation with limitations guard."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import RootModel, ValidationError

from agent.config import get_settings
from agent.models import Briefing, KeyResult, ParseMode
from agent.services.llm import complete_json, complete_text
from agent.services.prompts import (
    SYSTEM_LIMITATIONS_PASS,
    SYSTEM_SUMMARIZE_MAP,
    SYSTEM_SUMMARIZE_REDUCE,
    limitations_pass_prompt,
    summarize_map_prompt,
    summarize_reduce_prompt,
)
from agent.services.vectorstore import query_chunks
from agent.state import AgentState

logger = logging.getLogger(__name__)

# Pydantic models for list validation
class BulletList(RootModel):
    root: list[str]

# Sections considered most informative for the map phase
INFORMATIVE_SECTIONS = {
    "abstract",
    "introduction",
    "method",
    "methodology",
    "approach",
    "experiments",
    "results",
    "evaluation",
    "discussion",
    "conclusion",
    "limitations",
}


def summarize(state: AgentState) -> dict[str, Any]:
    """Generate structured briefing via map-reduce over paper sections.

    Implements the summarize node from the graph.
    - Map: per informative section → 3-5 factual bullets with section name
    - Reduce: all bullets → Briefing Pydantic model as JSON
    - Limitations guard: second pass if needed
    - JSON repair loop with fallback
    """
    sections = state.get("sections", [])
    paper = state.get("paper")
    collection = state.get("collection")
    parse_mode = state.get("parse_mode", "full")
    n_chunks = state.get("n_chunks", 0)
    warnings = state.get("warnings", [])

    if not sections:
        return {"briefing": None, "errors": [{"code": "NO_SECTIONS", "node": "summarize", "detail": "No sections to summarize", "recoverable": False}]}

    if not paper:
        return {"briefing": None, "errors": [{"code": "NO_PAPER", "node": "summarize", "detail": "No paper in state", "recoverable": False}]}

    # Handle both dict and Pydantic model
    if hasattr(paper, 'model_dump'):
        paper_dict = paper.model_dump(mode="json")
    else:
        paper_dict = paper

    try:
        # Map phase: extract bullets from informative sections
        all_bullets = _map_phase(sections)

        if not all_bullets:
            return {"briefing": None, "errors": [{"code": "NO_BULLETS", "node": "summarize", "detail": "No bullets extracted from sections", "recoverable": False}]}

        # Reduce phase: synthesize into Briefing
        briefing = _reduce_phase(paper_dict, all_bullets)

        # Limitations guard
        briefing = _limitations_guard(briefing, collection, paper_dict.get("arxiv_id", ""))

         # Prepare meta
        meta = {
                "model": get_settings().llm_provider,
                "parse_mode": parse_mode,
                "n_chunks": n_chunks,
                "warnings": warnings,
                "generated_at": datetime.utcnow().isoformat() + "Z",
                }
        briefing.meta = meta

        # Write briefing to file
        _write_briefing(briefing, paper_dict.get("arxiv_id", ""))

        # Print Markdown via Rich
        _print_briefing_markdown(briefing)

        # Use mode='json' to ensure HttpUrl and other types are serialized to strings
        return {"briefing": briefing.model_dump(mode="json")}

    except Exception as e:
        logger.error(f"Summarize failed: {e}")
        error = {
            "code": "SUMMARIZE_FAILED",
            "node": "summarize",
            "detail": str(e),
            "recoverable": False,
        }
        return {"briefing": None, "errors": [error]}


def _map_phase(sections: list[dict]) -> list[str]:
    """Extract 3-5 factual bullets from each informative section."""
    all_bullets = []

    for section in sections:
        title = section.get("title", "").strip()
        text = section.get("text", "").strip()

        if not text:
            continue

        # Check if section is informative (case-insensitive)
        title_lower = title.lower()
        if not any(info in title_lower for info in INFORMATIVE_SECTIONS):
            continue

        try:
            prompt = summarize_map_prompt(title, text)
            bullet_list = complete_json(
                BulletList,
                SYSTEM_SUMMARIZE_MAP,
                prompt,
            )
            all_bullets.extend(bullet_list.root)
        except Exception as e:
            logger.warning(f"Map phase failed for section '{title}': {e}")
            # Continue with other sections

    return all_bullets


def _reduce_phase(paper_meta: dict[str, Any], all_bullets: list[str]) -> Briefing:
    """Synthesize bullets into Briefing model with JSON repair loop."""
    prompt = summarize_reduce_prompt(paper_meta, all_bullets)

    # First attempt
    try:
        briefing = complete_json(Briefing, SYSTEM_SUMMARIZE_REDUCE, prompt)
        return briefing
    except (ValidationError, json.JSONDecodeError) as e:
        logger.warning(f"First briefing attempt failed validation: {e}")
        # Retry with error feedback
        retry_prompt = f"{prompt}\n\nPrevious response failed validation:\n{e}\n\nPlease return ONLY valid JSON matching the Briefing schema."
        try:
            briefing = complete_json(Briefing, SYSTEM_SUMMARIZE_REDUCE, retry_prompt)
            return briefing
        except (ValidationError, json.JSONDecodeError) as e2:
            logger.error(f"Second briefing attempt also failed: {e2}")
            # Fallback to markdown briefing
            return _fallback_briefing(paper_meta, all_bullets, str(e2))


def _limitations_guard(briefing: Briefing, collection: str | None, arxiv_id: str) -> Briefing:
    """Ensure limitations field is non-empty and meaningful."""
    limitations = briefing.limitations

    # Check if limitations are empty or contain filler
    filler_terms = {"none stated", "n/a", "none", "no limitations", "not mentioned"}
    is_filler = not limitations or any(
        any(filler in lim.lower() for filler in filler_terms)
        for lim in limitations
    )

    if not is_filler:
        return briefing

    # Second pass: retrieve chunks for limitations query
    if not collection or not arxiv_id:
        # No collection to query, add inferred limitation
        briefing.limitations = ["Not explicitly stated by the authors; reviewer-inferred: No limitations section found in paper."]
        return briefing

    try:
        chunks = query_chunks(collection, "limitations failure cases threats to validity future work", n_results=10)
        if not chunks:
            briefing.limitations = ["Not explicitly stated by the authors; reviewer-inferred: No limitations section found in paper."]
            return briefing

        prompt = limitations_pass_prompt(chunks)
        found_limitations = complete_json(BulletList, SYSTEM_LIMITATIONS_PASS, prompt)

        if found_limitations:
            briefing.limitations = found_limitations.root
        else:
            briefing.limitations = ["Not explicitly stated by the authors; reviewer-inferred: No limitations section found in paper."]

    except Exception as e:
        logger.warning(f"Limitations pass failed: {e}")
        briefing.limitations = ["Not explicitly stated by the authors; reviewer-inferred: No limitations section found in paper."]

    return briefing


def _fallback_briefing(paper_meta: dict[str, Any], all_bullets: list[str], error: str) -> Briefing:
    """Create a minimal valid Briefing when JSON validation fails repeatedly."""
    # Extract minimal fields from paper_meta
    arxiv_id = paper_meta.get("arxiv_id", "unknown")
    title = paper_meta.get("title", "Unknown Title")
    authors = paper_meta.get("authors", [])
    published = paper_meta.get("published", datetime.utcnow().isoformat() + "Z")
    categories = paper_meta.get("categories", [])
    url = str(paper_meta.get("url", f"https://arxiv.org/abs/{arxiv_id}"))
    pdf_url = str(paper_meta.get("pdf_url", f"https://arxiv.org/pdf/{arxiv_id}"))

    # Create minimal briefing from bullets
    key_results = []
    for bullet in all_bullets[:5]:
        # Try to parse section prefix
        parts = bullet.split("] ", 1)
        if len(parts) == 2 and parts[0].startswith("["):
            section = parts[0][1:]
            claim = parts[1]
        else:
            section = "Unknown"
            claim = bullet
        key_results.append(KeyResult(claim=claim, evidence="From paper", source_section=section))

    return Briefing(
        arxiv_id=arxiv_id,
        title=title,
        authors=authors,
        published=published,
        categories=categories,
        url=url,
        pdf_url=pdf_url,
        why_it_matters="Briefing generated in fallback mode due to validation errors.",
        problem_statement="See paper for problem statement.",
        method=[b for b in all_bullets if "method" in b.lower()][:5] or ["See paper"],
        key_results=key_results,
        limitations=["Not explicitly stated by the authors; reviewer-inferred: Briefing generated in fallback mode."],
        followup_questions=["What are the key contributions?", "What experiments were run?", "What are the limitations?"],
        meta={"fallback_error": error, "fallback_mode": True},
    )


def _write_briefing(briefing: Briefing, arxiv_id: str) -> None:
    """Write briefing JSON to examples/briefing_<id>.json."""
    safe_id = arxiv_id.replace("/", "_").replace(".", "_")
    output_dir = Path("examples")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"briefing_{safe_id}.json"

    try:
        with open(output_path, "w") as f:
            json.dump(briefing.model_dump(mode="json"), f, indent=2)
        logger.info(f"Briefing written to {output_path}")
    except Exception as e:
        logger.warning(f"Failed to write briefing file: {e}")


def _print_briefing_markdown(briefing: Briefing) -> None:
    """Print briefing as formatted Markdown via Rich."""
    try:
        from rich.console import Console
        from rich.markdown import Markdown

        console = Console()

        md = f"""# {briefing.title}

**Authors:** {', '.join(briefing.authors)}  
**Published:** {briefing.published}  
**Categories:** {', '.join(briefing.categories)}  
**arXiv:** [{briefing.arxiv_id}]({briefing.url}) | [PDF]({briefing.pdf_url})

---

## Why It Matters

{briefing.why_it_matters}

## Problem Statement

{briefing.problem_statement}

## Method

{chr(10).join(f"- {m}" for m in briefing.method)}

## Key Results

{chr(10).join(f"- **{kr.claim}** (Evidence: {kr.evidence}, Section: {kr.source_section})" for kr in briefing.key_results)}

## Limitations

{chr(10).join(f"- {l}" for l in briefing.limitations)}

## Follow-up Questions

{chr(10).join(f"{i+1}. {q}" for i, q in enumerate(briefing.followup_questions))}

---

*Generated by arXiv Digest Agent*
"""
        console.print(Markdown(md))
    except Exception as e:
        logger.warning(f"Failed to print Markdown: {e}")