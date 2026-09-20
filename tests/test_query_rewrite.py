"""Follow-up query rewrite: it must stay short and must never break retrieval.

A real run showed the model copying a whole list of method names from the history into
the query, which pulled a results table to the top of retrieval and produced a wrong answer.
"""

from agent.nodes import qa as qa_module
from agent.nodes.qa import _rewrite_query
from agent.services.prompts import SYSTEM_QUERY_REWRITE, query_rewrite_prompt

QUESTION = "Which of them suppresses outliers best?"
HISTORY = [
    {"role": "user", "content": "Which beamformers did they compare?"},
    {"role": "assistant", "content": "They compared Wiener, Wiener-DL, Capon and Kernel.", "citations": []},
]


def _stub_rewrite(monkeypatch, result):
    """Replace the rewrite LLM call. Returns the list of prompts it was called with."""
    calls = []

    def fake(system_prompt, user_prompt):
        calls.append(user_prompt)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(qa_module, "complete_text", fake)
    return calls


def test_short_rewrite_is_used(monkeypatch):
    _stub_rewrite(monkeypatch, "  Which of the compared methods suppresses outliers best?\n")
    assert _rewrite_query(QUESTION, HISTORY) == "Which of the compared methods suppresses outliers best?"


def test_overlong_rewrite_falls_back_to_the_original_question(monkeypatch):
    _stub_rewrite(monkeypatch, " ".join(["beamformer"] * 31))
    assert _rewrite_query(QUESTION, HISTORY) == QUESTION


def test_rewrite_at_the_limit_is_kept(monkeypatch):
    text = " ".join(["word"] * 30)
    _stub_rewrite(monkeypatch, text)
    assert _rewrite_query(QUESTION, HISTORY) == text


def test_empty_rewrite_and_llm_errors_fall_back(monkeypatch):
    _stub_rewrite(monkeypatch, "   ")
    assert _rewrite_query(QUESTION, HISTORY) == QUESTION
    _stub_rewrite(monkeypatch, RuntimeError("boom"))
    assert _rewrite_query(QUESTION, HISTORY) == QUESTION


def test_no_llm_call_without_history_or_for_standalone_questions(monkeypatch):
    calls = _stub_rewrite(monkeypatch, "should not be used")
    assert _rewrite_query(QUESTION, []) == QUESTION
    standalone = "What does this paper say about the 2026 World Cup?"
    assert _rewrite_query(standalone, HISTORY) == standalone
    assert calls == []  # an off-topic question is never rewritten, so the abstain gate still sees it as asked


def test_prompts_discourage_copying_lists_from_history():
    assert "Do NOT copy" in SYSTEM_QUERY_REWRITE
    long_answer = "x" * 1000
    prompt = query_rewrite_prompt(QUESTION, [{"role": "assistant", "content": long_answer}])
    assert long_answer not in prompt
    assert len(prompt) < 700