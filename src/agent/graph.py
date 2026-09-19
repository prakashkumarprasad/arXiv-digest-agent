"""LangGraph workflow graph for the arXiv Digest Agent."""

from typing import Any
from contextlib import contextmanager

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from agent.state import AgentState
from agent.config import get_settings

# Import all nodes
from agent.nodes.query_understanding import query_understanding
from agent.nodes.retrieval import search_arxiv, broaden_query, fetch_metadata
from agent.nodes.fetch_parse import fetch_pdf, parse, degrade_mode
from agent.nodes.indexing import chunk_embed
from agent.nodes.selection import select_paper
from agent.nodes.summarize import summarize
# QA node will be added in later stages
# from agent.nodes.qa import qa_node


# Global checkpointer instance (for SqliteSaver)
_sqlite_checkpointer = None


@contextmanager
def _sqlite_checkpointer_context():
    """Context manager for SqliteSaver checkpointer."""
    global _sqlite_checkpointer
    if _sqlite_checkpointer is None:
        settings = get_settings()
        _sqlite_checkpointer = SqliteSaver.from_conn_string(str(settings.sqlite_db_path))
        _sqlite_checkpointer = _sqlite_checkpointer.__enter__()
    yield _sqlite_checkpointer


def _route_intent(state: AgentState) -> str:
    """Route based on intent: paper_lookup or topic_search."""
    intent = state.get("intent", "unclear")
    if intent == "paper_lookup":
        return "fetch_metadata"
    elif intent == "topic_search":
        return "search_arxiv"
    return "search_arxiv"  # fallback


def _route_candidate_count(state: AgentState) -> str:
    """Route based on number of candidates: 0 -> broaden_query, 1+ -> select_paper."""
    candidates = state.get("candidates", [])
    if len(candidates) == 0:
        return "broaden_query"
    return "select_paper"


def _route_parse_quality(state: AgentState) -> str:
    """Route based on parse quality: full -> chunk_embed, degraded/failed -> degrade_mode."""
    parse_mode = state.get("parse_mode", "failed")
    if parse_mode == "full":
        return "chunk_embed"
    return "degrade_mode"


def _should_continue_broaden(state: AgentState) -> str:
    """Decide whether to continue broadening or terminate."""
    retries = state.get("retries", {})
    broaden_attempts = retries.get("broaden_query", 0)
    search_query = state.get("search_query")
    candidates = state.get("candidates", [])

    if candidates:
        return "select_paper"  # broaden_query already found results, don't re-search
    if broaden_attempts >= 2 and not search_query:
        return "end"  # exhausted
    if search_query:
        return "search_arxiv"  # still zero results, try again with broadened query
    return "end"


def build_graph(checkpointer=None) -> StateGraph:
    """Build and compile the LangGraph workflow.

    Args:
        checkpointer: Optional checkpointer. If None, uses InMemorySaver for testing.
                     Pass a SqliteSaver context manager for persistence.

    Returns a compiled graph.
    thread_id = arxiv_id so QA re-attaches to previous session without re-parsing.
    """
    if checkpointer is None:
        checkpointer = InMemorySaver()

    workflow = StateGraph(AgentState)

    # Add all nodes
    workflow.add_node("query_understanding", query_understanding)
    workflow.add_node("fetch_metadata", fetch_metadata)
    workflow.add_node("search_arxiv", search_arxiv)
    workflow.add_node("broaden_query", broaden_query)
    workflow.add_node("select_paper", select_paper)
    workflow.add_node("fetch_pdf", fetch_pdf)
    workflow.add_node("parse", parse)
    workflow.add_node("degrade_mode", degrade_mode)
    workflow.add_node("chunk_embed", chunk_embed)
    workflow.add_node("summarize", summarize)
    # workflow.add_node("qa_node", qa_node)  # Stage 7

    # Set entry point
    workflow.set_entry_point("query_understanding")

    # Intent routing
    workflow.add_conditional_edges(
        "query_understanding",
        _route_intent,
        {
            "fetch_metadata": "fetch_metadata",
            "search_arxiv": "search_arxiv",
        },
    )

    # fetch_metadata goes to fetch_pdf
    workflow.add_edge("fetch_metadata", "fetch_pdf")

    # Candidate count routing
    workflow.add_conditional_edges(
        "search_arxiv",
        _route_candidate_count,
        {
            "broaden_query": "broaden_query",
            "select_paper": "select_paper",
        },
    )

    # broaden_query either goes back to search_arxiv or ends
    workflow.add_conditional_edges(
        "broaden_query",
        _should_continue_broaden,
        {
            "search_arxiv": "search_arxiv",
            "select_paper": "select_paper",
            "end": END,
        },
    )

    # select_paper -> fetch_pdf
    workflow.add_edge("select_paper", "fetch_pdf")

    # fetch_pdf -> parse
    workflow.add_edge("fetch_pdf", "parse")

    # Parse quality routing
    workflow.add_conditional_edges(
        "parse",
        _route_parse_quality,
        {
            "chunk_embed": "chunk_embed",
            "degrade_mode": "degrade_mode",
        },
    )

    # degrade_mode -> chunk_embed
    workflow.add_edge("degrade_mode", "chunk_embed")

    # chunk_embed -> summarize
    workflow.add_edge("chunk_embed", "summarize")

    # summarize -> END
    workflow.add_edge("summarize", END)

    return workflow.compile(checkpointer=checkpointer)


def get_graph() -> StateGraph:
    """Get the compiled graph (for inspection/drawing) using InMemorySaver."""
    return build_graph()


def get_persistent_graph():
    """Get the compiled graph with SqliteSaver checkpointer for persistent sessions.

    Must be used within a context manager:
        with get_persistent_graph() as graph:
            result = graph.invoke(...)
    """
    return _sqlite_checkpointer_context()