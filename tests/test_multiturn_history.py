"""Tests for multiturn QA history: three turns give message counts 2, 4, 6; _start returns {}."""

from agent.graph import build_graph, _start
from agent.nodes.qa import qa_node


def _stub_query_chunks(*a, **k):
    return [
        {"chunk_id": "S1", "text": "relevant text", "section": "intro", "page_start": 1, "distance": 0.05},
    ]


def _stub_complete_json(*a, **k):
    from unittest.mock import MagicMock
    return MagicMock(model_dump=lambda: {"answer": "Test answer.", "citations": [{"chunk_id": "S1", "section": "intro", "text": "text"}], "grounded": True})


def _stub_complete_text(*a, **k):
    return "rewritten question"


class TestMultiturnHistory:
    def test_three_turns_message_counts(self, monkeypatch):
        """Three turns through the compiled graph give message counts 2, 4, 6 with each Q/A exactly once."""
        monkeypatch.setattr("agent.nodes.qa.query_chunks", _stub_query_chunks)
        monkeypatch.setattr("agent.nodes.qa.complete_json", _stub_complete_json)
        monkeypatch.setattr("agent.nodes.qa.complete_text", _stub_complete_text)

        graph = build_graph()
        thread_id = "test_multiturn_001"
        config = {"configurable": {"thread_id": thread_id}}

        # Turn 1: only pass question and collection; messages are in the checkpoint
        result1 = graph.invoke({"question": "What is quantum gravity?", "collection": "test_collection"}, config=config)
        n_msgs1 = len(result1.get("messages", []))
        assert n_msgs1 == 2, f"Turn 1: expected 2 messages, got {n_msgs1}"

        # Turn 2: only pass new question; checkpointer maintains history
        result2 = graph.invoke({"question": "What datasets did they use?", "collection": "test_collection"}, config=config)
        n_msgs2 = len(result2.get("messages", []))
        assert n_msgs2 == 4, f"Turn 2: expected 4 messages, got {n_msgs2}"

        # Turn 3
        result3 = graph.invoke({"question": "What are the limitations?", "collection": "test_collection"}, config=config)
        n_msgs3 = len(result3.get("messages", []))
        assert n_msgs3 == 6, f"Turn 3: expected 6 messages, got {n_msgs3}"

    def test_start_returns_empty_dict(self):
        """The start node returns {} (regression for §11 Stage 7)."""
        result = _start({})
        assert result == {}

    def test_each_turn_has_exactly_one_qa(self, monkeypatch):
        """Each turn has exactly one user question and one assistant answer."""
        monkeypatch.setattr("agent.nodes.qa.query_chunks", _stub_query_chunks)
        monkeypatch.setattr("agent.nodes.qa.complete_json", _stub_complete_json)
        monkeypatch.setattr("agent.nodes.qa.complete_text", _stub_complete_text)

        graph = build_graph()
        thread_id = "test_multiturn_002"
        config = {"configurable": {"thread_id": thread_id}}

        messages = []
        for i, q in enumerate(["Q1?", "Q2?", "Q3?"]):
            result = graph.invoke({"question": q, "collection": "test_collection"}, config=config)
            state_msgs = result.get("messages", [])
            # Each turn adds exactly 2 messages (1 user + 1 assistant)
            assert len(state_msgs) == (i + 1) * 2, f"Turn {i+1}: expected {(i+1)*2} messages, got {len(state_msgs)}"
            # Last message is assistant answer
            assert state_msgs[-1]["role"] == "assistant", f"Turn {i+1}: last message should be assistant"
            assert state_msgs[-1]["content"] == "Test answer."