"""End-to-end test of required failure case #1: a topic search with zero results.

Runs the REAL compiled graph (real routing, real search_arxiv / broaden_query
nodes) with only the LLM call and the arXiv network call stubbed. Every node
after search/selection is replaced by one that fails if it is ever reached.
"""

import uuid
from types import SimpleNamespace

from langgraph.checkpoint.memory import InMemorySaver

from agent.graph import build_graph
from agent.services.arxiv_client import ArxivClient

DOWNSTREAM_NODES = [
    "fetch_metadata", "select_paper", "fetch_pdf", "parse",
    "degrade_mode", "chunk_embed", "summarize", "qa_node",
]


def _forbidden(name):
    def node(state):
        raise AssertionError(f"{name} must not run on the zero-result path")
    return node


def test_zero_results_graph_ends_cleanly(monkeypatch):
    search_calls = []

    def fake_search(self, query, *args, **kwargs):
        search_calls.append(query)
        return []

    monkeypatch.setattr(ArxivClient, "search", fake_search)
    monkeypatch.setattr(
        "agent.nodes.query_understanding.complete_json",
        lambda *a, **k: SimpleNamespace(query="quantum flux capacitor theory", categories=[]),
    )
    # graph.py binds node functions when build_graph() runs, so patch BEFORE building.
    for name in DOWNSTREAM_NODES:
        monkeypatch.setattr(f"agent.graph.{name}", _forbidden(name))

    graph = build_graph(checkpointer=InMemorySaver())
    config = {
        "configurable": {"thread_id": f"zero-results-{uuid.uuid4()}"},
        "recursion_limit": 30,  # a runaway broaden loop fails fast instead of hanging
    }

    result = graph.invoke(
        {"raw_input": "quantum flux capacitor theory", "question": None},
        config=config,
    )

    # Ended without raising, and never reached a downstream node.
    assert not result.get("paper")
    assert not result.get("pdf_path")
    assert result.get("candidates") == []

    # Broadening really ran through the graph: exactly two attempts.
    assert result["retries"]["broaden_query"] == 2
    assert result.get("search_query") is None  # termination signal

    # search_arxiv x3, plus one search inside each of the 2 broaden attempts.
    # The exhausted broaden makes no search call.
    assert len(search_calls) == 5
    assert '"' in search_calls[0] and " AND " in search_calls[0]       # original query
    assert '"' not in search_calls[1] and " AND " not in search_calls[1]  # broadened

    # Clean, non-traceback error entry.
    exhausted = [e for e in result.get("errors", []) if e["code"] == "ZERO_RESULTS_EXHAUSTED"]
    assert len(exhausted) == 1
    assert "No results after 2 broadening attempts" in exhausted[0]["detail"]
    assert "Traceback" not in exhausted[0]["detail"]

    # The message lists every query tried and says what to do next.
    detail = exhausted[0]["detail"]
    assert "Queries tried:" in detail
    assert search_calls[0] in detail  # the original query
    assert search_calls[1] in detail  # broadened attempt 1
    assert search_calls[3] in detail  # broadened attempt 2
    assert "arXiv ID" in detail
