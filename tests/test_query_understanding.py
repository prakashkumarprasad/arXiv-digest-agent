"""Tests for query_understanding node: arXiv ID detection, URL parsing, topic search, LLM fallback."""

from types import SimpleNamespace

from agent.nodes.query_understanding import query_understanding


def _llm_down(*a, **k):
    raise Exception("LLM down")


class TestQueryUnderstanding:
    def test_new_style_arxiv_id(self):
        """New-format ID '2401.12345' → paper_lookup with correct arxiv_id."""
        result = query_understanding({"raw_input": "2401.12345"})
        assert result["intent"] == "paper_lookup"
        assert result["arxiv_id"] == "2401.12345"

    def test_arxiv_id_with_v_suffix_stripped(self):
        """ID with version suffix '2401.12345v2' → arxiv_id without suffix."""
        result = query_understanding({"raw_input": "2401.12345v2"})
        assert result["intent"] == "paper_lookup"
        assert result["arxiv_id"] == "2401.12345"

    def test_arxiv_org_abs_url(self):
        """arxiv.org/abs URL → extracts arxiv_id."""
        result = query_understanding({"raw_input": "https://arxiv.org/abs/2401.12345"})
        assert result["intent"] == "paper_lookup"
        assert result["arxiv_id"] == "2401.12345"

    def test_arxiv_org_pdf_url(self):
        """arxiv.org/pdf URL → extracts arxiv_id."""
        result = query_understanding({"raw_input": "https://arxiv.org/pdf/2401.12345"})
        assert result["intent"] == "paper_lookup"
        assert result["arxiv_id"] == "2401.12345"

    def test_old_style_id(self):
        """Old-style ID 'cs.CL/1234567' → paper_lookup with correct id."""
        result = query_understanding({"raw_input": "cs.CL/1234567"})
        assert result["intent"] == "paper_lookup"
        assert result["arxiv_id"] == "cs.CL/1234567"

    def test_free_text_topic_search(self, monkeypatch):
        """Free text without an arXiv ID → topic_search, LLM call is stubbed."""
        monkeypatch.setattr("agent.nodes.query_understanding.complete_json",
                            lambda *a, **k: SimpleNamespace(query="topic", categories=[]))
        result = query_understanding({"raw_input": "recent work on KV-cache compression"})
        assert result["intent"] == "topic_search"
        assert "search_query" in result

    def test_llm_failure_falls_back_to_raw_string(self, monkeypatch):
        """When complete_json fails, falls back to raw string as search_query, no exception."""
        monkeypatch.setattr("agent.nodes.query_understanding.complete_json", _llm_down)
        result = query_understanding({"raw_input": "some topic about transformers"})
        assert result["intent"] == "topic_search"
        assert "search_query" in result
        assert isinstance(result["search_query"], str)

    def test_empty_input_returns_unclear(self):
        """Empty input → unclear intent with error."""
        result = query_understanding({"raw_input": ""})
        assert result["intent"] == "unclear"
        assert len(result.get("errors", [])) > 0