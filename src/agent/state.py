"""Agent state definition for the LangGraph workflow."""

from typing import TypedDict, Literal, NotRequired
from typing import Annotated
import operator

from agent.models import AgentError


class AgentState(TypedDict, total=False):
    """Shared state for the agent graph.

    All fields are optional (total=False) to allow partial updates
    from individual nodes. Nodes return only the keys they modify.
    """

    # --- input ---
    raw_input: str
    intent: Literal["paper_lookup", "topic_search", "unclear"]
    arxiv_id: str | None
    search_query: str | None

    # --- retrieval ---
    candidates: list[dict]
    selection_reason: str | None
    paper: dict | None

    # --- parsing ---
    pdf_path: str | None
    sections: list[dict]
    full_text: str | None
    parse_mode: Literal["full", "degraded_abstract_only", "failed"]

    # --- indexing ---
    collection: str | None
    n_chunks: int

    # --- output ---
    briefing: dict | None

    # --- QA ---
    question: str | None
    retrieved: list[dict]

    # --- control ---
    errors: Annotated[list[dict], operator.add]
    warnings: Annotated[list[str], operator.add]
    messages: Annotated[list[dict], operator.add]
    retries: dict[str, int]