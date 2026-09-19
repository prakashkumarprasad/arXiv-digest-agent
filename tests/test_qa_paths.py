"""Behavior tests for qa_node: return-path message handling and [Sn] mapping."""

from types import SimpleNamespace

import pytest

from agent.nodes.qa import ABSTAIN_MESSAGE, qa_node
from agent.services.prompts import qa_prompt as real_qa_prompt

QUESTION = "What does the paper say about tokamak plasma confinement physics?"

CHUNKS = [
    {"chunk_id": "p:0", "text": "alpha bravo charlie delta", "section": "intro", "page_start": 1, "distance": 0.10},
    {"chunk_id": "p:1", "text": "echo foxtrot golf hotel", "section": "method", "page_start": 2, "distance": 0.15},
    {"chunk_id": "p:2", "text": "india juliet kilo lima", "section": "results", "page_start": 3, "distance": 0.20},
    {"chunk_id": "p:3", "text": "mike november oscar papa", "section": "conclusion", "page_start": 4, "distance": 0.25},
]

FAR_CHUNKS = [{**c, "distance": 0.90} for c in CHUNKS]

HISTORY = [
    {"role": "user", "content": "Q1?"},
    {"role": "assistant", "content": "A1.", "citations": []},
    {"role": "user", "content": "Q2?"},
    {"role": "assistant", "content": "A2.", "citations": []},
]


class _FakeLLMResponse:
    def __init__(self, data):
        self._data = data

    def model_dump(self):
        return self._data


def _setup(monkeypatch, chunks, llm_data=None):
    """Stub settings, vector store and LLM in the qa module. Returns a dict that
    records the context blocks the prompt was built from."""
    seen = {}

    def recording_prompt(question, context_blocks):
        seen["blocks"] = context_blocks
        return real_qa_prompt(question, context_blocks)

    def fake_complete_json(schema, system_prompt, user_prompt, *args, **kwargs):
        if llm_data is None:
            raise AssertionError("LLM should not be called on this path")
        return _FakeLLMResponse(llm_data)

    monkeypatch.setattr("agent.nodes.qa.get_settings", lambda: SimpleNamespace(abstain_max_distance=0.45))
    monkeypatch.setattr("agent.nodes.qa.query_chunks", lambda *a, **k: [dict(c) for c in chunks])
    monkeypatch.setattr("agent.nodes.qa.qa_prompt", recording_prompt)
    monkeypatch.setattr("agent.nodes.qa.complete_json", fake_complete_json)
    return seen


def test_sn_label_maps_to_chunk_shown_in_that_prompt_block(monkeypatch):
    """[S1]/[S2] in the answer must resolve to the chunks printed as blocks 1/2 in the prompt."""
    llm_data = {
        "answer": "It works [S1][S2].",
        "citations": [{"chunk_id": "S1", "section": "x"}, {"chunk_id": "S2", "section": "x"}],
        "grounded": True,
    }
    seen = _setup(monkeypatch, CHUNKS, llm_data)

    result = qa_node({"question": QUESTION, "collection": "c", "messages": []})

    by_text = {c["text"]: c for c in CHUNKS}
    expected = [by_text[seen["blocks"][0]["text"]], by_text[seen["blocks"][1]["text"]]]
    citations = result["messages"][-1]["citations"]

    assert [c["chunk_id"] for c in citations] == [c["chunk_id"] for c in expected]
    assert [c["section"] for c in citations] == [c["section"] for c in expected]
    assert [c["snippet"] for c in citations] == [c["text"] for c in expected]


@pytest.mark.parametrize(
    "chunks,llm_data,expect_abstain",
    [
        ([], None, True),
        (FAR_CHUNKS, None, True),
        (
            CHUNKS,
            {"answer": "Made up [S9].", "citations": [{"chunk_id": "S9", "section": "x"}], "grounded": True},
            True,
        ),
        (
            CHUNKS,
            {"answer": "Real [S1].", "citations": [{"chunk_id": "S1", "section": "x"}], "grounded": True},
            False,
        ),
    ],
    ids=["no_chunks", "distance_abstain", "no_valid_citation", "success"],
)
def test_every_return_path_returns_only_new_messages(monkeypatch, chunks, llm_data, expect_abstain):
    """messages uses operator.add, so each path must return ONLY this turn's Q/A pair."""
    _setup(monkeypatch, chunks, llm_data)

    result = qa_node({"question": QUESTION, "collection": "c", "messages": list(HISTORY)})

    msgs = result["messages"]
    assert len(msgs) == 2
    assert msgs[0] == {"role": "user", "content": QUESTION}
    assert msgs[1]["role"] == "assistant"
    if expect_abstain:
        assert msgs[1] == {"role": "assistant", "content": ABSTAIN_MESSAGE, "citations": []}
    else:
        assert msgs[1]["content"] == "Real [S1]."
        assert len(msgs[1]["citations"]) == 1