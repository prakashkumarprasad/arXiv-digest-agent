"""Tests for QA abstain gate: distance threshold, LLM call suppression, no-valid-citation abstain, _needs_rewrite."""

import pytest

from agent.nodes.qa import _needs_rewrite, ABSTAIN_MESSAGE


class TestNeedsRewrite:
    def test_standalone_question_verbatim(self, monkeypatch):
        """A standalone question >5 words with no reference word is used verbatim; rewrite LLM NOT called."""
        from unittest.mock import MagicMock
        monkeypatch.setattr("agent.nodes.qa.complete_text", MagicMock(side_effect=Exception("should not be called")))
        question = "What method does this paper propose"
        result = _needs_rewrite(question)
        assert result is False

    def test_short_follow_up_is_rewritten(self):
        """A short question (<=5 words) is treated as needing rewrite."""
        result = _needs_rewrite("Does it scale?")
        assert result is True

    def test_pronoun_follow_up_is_rewritten(self):
        """A follow-up containing a reference word is rewritten."""
        result = _needs_rewrite("How does it compare")
        assert result is True


class TestQaAbstain:
    def test_abstain_when_distance_above_threshold(self, monkeypatch):
        """best distance above abstain_max_distance -> canned message, citations empty, LLM NOT called."""
        from unittest.mock import MagicMock
        from agent.nodes.qa import qa_node

        # Return chunks with distance > 0.45 (above default abstain_max_distance)
        mock_chunks = [
            {"chunk_id": "S1", "text": "text", "section": "intro", "page_start": 1, "distance": 0.9},
        ]
        monkeypatch.setattr("agent.nodes.qa.query_chunks", lambda *a, **k: mock_chunks)
        llm_mock = MagicMock(side_effect=Exception("LLM should not be called"))
        monkeypatch.setattr("agent.nodes.qa.complete_json", llm_mock)
        monkeypatch.setattr("agent.nodes.qa.complete_text", llm_mock)

        state = {
            "question": "What is quantum gravity?",
            "collection": "test_collection",
            "messages": [],
        }
        result = qa_node(state)

        assert len(result["messages"]) == 2
        assert result["messages"][1]["content"] == ABSTAIN_MESSAGE
        assert result["messages"][1]["citations"] == []
        assert llm_mock.called is False

    def test_llm_called_when_distance_in_range(self, monkeypatch):
        """best distance within range -> the LLM IS called."""
        from unittest.mock import MagicMock
        from agent.nodes.qa import qa_node

        # Return chunks with distance < 0.45
        mock_chunks = [
            {"chunk_id": "S1", "text": "relevant text about quantum gravity", "section": "intro", "page_start": 1, "distance": 0.1},
        ]
        monkeypatch.setattr("agent.nodes.qa.query_chunks", lambda *a, **k: mock_chunks)
        llm_mock = MagicMock(return_value=MagicMock(
            model_dump=lambda: {"answer": "Quantum gravity is a theory.", "citations": [], "grounded": True}
        ))
        monkeypatch.setattr("agent.nodes.qa.complete_json", llm_mock)

        state = {
            "question": "What is quantum gravity?",
            "collection": "test_collection",
            "messages": [],
        }
        result = qa_node(state)
        assert llm_mock.called is True

    def test_no_valid_citations_returns_canned_message(self, monkeypatch):
        """An LLM answer with no valid citations -> the canned abstain message."""
        from unittest.mock import MagicMock
        from agent.nodes.qa import qa_node

        mock_chunks = [
            {"chunk_id": "S1", "text": "relevant text", "section": "intro", "page_start": 1, "distance": 0.1},
        ]
        monkeypatch.setattr("agent.nodes.qa.query_chunks", lambda *a, **k: mock_chunks)
        # LLM returns citations with unknown chunk_ids (not in S1..S6)
        llm_mock = MagicMock(return_value=MagicMock(
            model_dump=lambda: {"answer": "some answer", "citations": [{"chunk_id": "S99", "section": "unknown", "text": "x"}], "grounded": True}
        ))
        monkeypatch.setattr("agent.nodes.qa.complete_json", llm_mock)

        state = {
            "question": "What is quantum gravity?",
            "collection": "test_collection",
            "messages": [],
        }
        result = qa_node(state)

        assert len(result["messages"]) == 2
        assert result["messages"][1]["content"] == ABSTAIN_MESSAGE
        assert result["messages"][1]["citations"] == []
