"""The exhausted zero-result message must list every query tried and say what to do next."""

from agent.nodes.retrieval import broaden_query
from agent.services.arxiv_client import ArxivClient
from agent.state import AgentState


def test_exhausted_message_lists_every_query_and_suggests_next_step():
    tried = ["all:a AND all:b", "all:a OR all:b", "all:c OR all:d"]
    state = {"search_query": tried[-1], "retries": {"broaden_query": 2}, "queries_tried": tried, "candidates": []}

    result = broaden_query(state)

    error = result["errors"][0]
    assert error["code"] == "ZERO_RESULTS_EXHAUSTED"
    assert "No results after 2 broadening attempts" in error["detail"]
    for query in tried:
        assert query in error["detail"]
    assert "arXiv ID" in error["detail"]
    assert result["search_query"] is None


def test_failed_attempts_record_the_queries_tried(monkeypatch):
    monkeypatch.setattr(ArxivClient, "search", lambda self, *a, **k: [])
    original = '(all:"alpha" AND all:"beta") AND (cat:cs.CL)'

    first = broaden_query({"search_query": original, "retries": {}, "candidates": []})
    assert first["queries_tried"] == [original, "all:alpha OR all:beta"]
    assert "all:alpha OR all:beta" in first["warnings"][0]

    second = broaden_query({**first, "candidates": []})
    assert second["queries_tried"] == first["queries_tried"] + [second["search_query"]]


def test_queries_tried_is_declared_in_agent_state():
    # LangGraph silently drops keys that are not declared in the state schema.
    assert "queries_tried" in AgentState.__annotations__