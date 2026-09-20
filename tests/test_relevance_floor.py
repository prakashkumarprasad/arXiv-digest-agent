"""select_paper must reject irrelevant matches, and the graph must stop instead of digesting them."""

import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from langgraph.checkpoint.memory import InMemorySaver

from agent.graph import build_graph
from agent.nodes.selection import select_paper
from agent.services.arxiv_client import ArxivClient


def _candidate(title, published="2024-01-01T00:00:00Z"):
    return {"arxiv_id": "x", "title": title, "published": published, "pdf_url": "http://example.com/x.pdf"}


def _scores(monkeypatch, scores):
    monkeypatch.setattr("agent.nodes.selection._get_llm_relevance", lambda candidates, query: scores)


def test_no_relevant_candidate_returns_no_paper_and_actionable_error(monkeypatch):
    _scores(monkeypatch, {0: 1.0, 1: 2.0})
    state = {"raw_input": "my topic", "candidates": [_candidate("First"), _candidate("Second")]}

    result = select_paper(state)

    assert result["paper"] is None
    error = result["errors"][0]
    assert error["code"] == "NO_RELEVANT_PAPER"
    assert "Second" in error["detail"]  # the closest match is named
    assert "arXiv ID" in error["detail"]  # and the user is told what to do next


def test_relevant_candidate_is_selected(monkeypatch):
    _scores(monkeypatch, {0: 9.0, 1: 2.0})
    state = {"raw_input": "my topic", "candidates": [_candidate("Good"), _candidate("Bad")]}

    result = select_paper(state)

    assert result["paper"]["title"] == "Good"
    assert "errors" not in result


def test_floor_is_applied_before_ranking(monkeypatch):
    # Without the floor the brand-new paper wins on recency (composite ~6.3 vs ~4.2).
    fresh = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    _scores(monkeypatch, {0: 4.0, 1: 3.9})
    state = {
        "raw_input": "my topic",
        "candidates": [_candidate("Old but relevant", "2010-01-01T00:00:00Z"), _candidate("New but off-topic", fresh)],
    }

    result = select_paper(state)

    assert result["paper"]["title"] == "Old but relevant"


def test_scoring_failure_does_not_reject(monkeypatch):
    _scores(monkeypatch, None)  # the LLM call failed: there is nothing to judge relevance with
    state = {"raw_input": "my topic", "candidates": [_candidate("Only one")]}

    result = select_paper(state)

    assert result["paper"]["title"] == "Only one"


def test_route_after_selection():
    from agent.graph import _route_after_selection

    assert _route_after_selection({"paper": {"arxiv_id": "x"}}) == "fetch_pdf"
    assert _route_after_selection({"paper": None}) == "end"
    assert _route_after_selection({}) == "end"


def _forbidden(name):
    def node(state):
        raise AssertionError(f"{name} must not run when no paper is relevant")
    return node


def test_irrelevant_match_ends_the_graph_before_fetch_pdf(monkeypatch):
    def fake_search(self, query, *args, **kwargs):
        return [{
            "arxiv_id": "2609.00001",
            "title": "Coding agent harness study",
            "published": "2026-09-17T00:00:00Z",
            "pdf_url": "https://arxiv.org/pdf/2609.00001",
        }]

    monkeypatch.setattr(ArxivClient, "search", fake_search)
    monkeypatch.setattr(
        "agent.nodes.query_understanding.complete_json",
        lambda *a, **k: SimpleNamespace(query="obscure topic words", categories=[]),
    )
    _scores(monkeypatch, {0: 1.0})
    for name in ["fetch_metadata", "fetch_pdf", "parse", "degrade_mode", "chunk_embed", "summarize", "qa_node"]:
        monkeypatch.setattr(f"agent.graph.{name}", _forbidden(name))

    graph = build_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": f"irrelevant-{uuid.uuid4()}"}, "recursion_limit": 30}

    result = graph.invoke({"raw_input": "obscure topic words", "question": None}, config=config)

    assert not result.get("paper")
    assert [e["code"] for e in result.get("errors", [])] == ["NO_RELEVANT_PAPER"]