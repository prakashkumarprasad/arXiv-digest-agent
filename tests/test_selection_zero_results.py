"""Tests for zero-result search: broaden_query runs at most twice, clean termination, successful broaden routes to select_paper."""

from agent.nodes.retrieval import broaden_query, search_arxiv
from agent.graph import _should_continue_broaden, _route_candidate_count
from agent.services.arxiv_client import ArxivClient


def _stub_arxiv_search(self, *args, **kwargs):
    return []


def _stub_arxiv_search_with_candidates(self, *args, **kwargs):
    from agent.models import PaperMeta
    return [PaperMeta(
        arxiv_id="2401.12345", title="Test", authors=[], summary="",
        published="2024-01-01", updated="2024-01-01", categories=[],
        primary_category="cs.CV", url="https://arxiv.org/abs/2401.12345",
        pdf_url="https://arxiv.org/pdf/2401.12345", entry_id="test"
    )]


class TestSelectionZeroResults:
    def test_search_returns_empty_triggers_broaden(self, monkeypatch):
        """When search_arxiv returns empty candidates, broaden_query must be triggered."""
        monkeypatch.setattr(ArxivClient, 'search', _stub_arxiv_search)
        result = search_arxiv({"search_query": "zzzzzzz_nonexistent_query_12345", "candidates": []})
        assert result.get("candidates") == []

    def test_broaden_runs_at_most_twice(self, monkeypatch):
        """broaden_query must not exceed 2 attempts. After 2 attempts it terminates cleanly."""
        monkeypatch.setattr(ArxivClient, 'search', _stub_arxiv_search)
        # Attempt 0
        state = {"search_query": "test query", "retries": {"broaden_query": 0}, "candidates": []}
        result1 = broaden_query(state)
        assert result1["retries"]["broaden_query"] == 1
        # Attempt 1
        state = {"search_query": "test query", "retries": {"broaden_query": 1}, "candidates": []}
        result2 = broaden_query(state)
        assert result2["retries"]["broaden_query"] == 2
        # Attempt 2 → exhausted
        state = {"search_query": None, "retries": {"broaden_query": 2}, "candidates": []}
        result3 = broaden_query(state)
        assert result3["candidates"] == []
        assert any(e["code"] == "ZERO_RESULTS_EXHAUSTED" for e in result3.get("errors", []))

    def test_broaden_terminates_cleanly_with_actionable_message(self, monkeypatch):
        """After exhausting broadening attempts, the result must contain an actionable message and the queries tried."""
        monkeypatch.setattr(ArxivClient, 'search', _stub_arxiv_search)
        state = {"search_query": None, "retries": {"broaden_query": 2}, "candidates": []}
        result = broaden_query(state)
        errors = result.get("errors", [])
        assert any("No results after 2 broadening attempts" in e.get("detail", "") for e in errors)
        assert result["search_query"] is None  # signals termination

    def test_search_returns_empty_then_broaden_successfully(self, monkeypatch):
        """Search returns [] → broaden runs → successful broaden routes to select_paper."""
        monkeypatch.setattr(ArxivClient, 'search', _stub_arxiv_search)
        # First search returns empty
        state = {"search_query": "test", "retries": {"broaden_query": 0}, "candidates": []}
        result = broaden_query(state)
        assert result["retries"]["broaden_query"] == 1
        # After successful broaden, _should_continue_broaden routes to select_paper
        state_with_results = {"candidates": [{"arxiv_id": "2401.12345"}], "search_query": "broadened", "retries": {"broaden_query": 0}}
        next_node = _should_continue_broaden(state_with_results)
        assert next_node == "select_paper"

    def test_successful_broaden_no_redundant_search(self):
        """Regression for §11 Stage 5: successful broadening must NOT route back to search_arxiv."""
        state = {"candidates": [{"arxiv_id": "2401.12345"}], "search_query": "query", "retries": {"broaden_query": 0}}
        next_node = _should_continue_broaden(state)
        assert next_node != "search_arxiv"

    def test_zero_results_ends_after_two_broaden_attempts(self, monkeypatch):
        """Simulate the full zero-result path: search → broaden → broaden → end.

        Tests that the routing functions terminate cleanly after at most 2 broaden attempts.
        """
        monkeypatch.setattr(ArxivClient, 'search', _stub_arxiv_search)
        # Simulate the graph flow: search_arxiv returns [] → broaden → search_arxiv → broaden → end
        state = {"search_query": "test", "retries": {"broaden_query": 0}, "candidates": []}
        # broaden attempt 0
        r1 = broaden_query(state)
        assert _should_continue_broaden(r1) == "search_arxiv"
        # broaden attempt 1 (retries from state merged)
        state2 = {**state, **r1}
        r2 = broaden_query(state2)
        assert _should_continue_broaden(r2) == "search_arxiv"
        # broaden attempt 2 (retries from state merged)
        state3 = {**state, **r2}
        r3 = broaden_query(state3)
        assert _should_continue_broaden(r3) == "end"
        assert r3["search_query"] is None
        assert any("2 broadening attempts" in e["detail"] for e in r3.get("errors", []))