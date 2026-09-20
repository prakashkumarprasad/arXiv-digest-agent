"""Unit tests for the query-broadening helpers (pure functions, no network)."""

from agent.nodes.retrieval import _broaden_attempt_1, _broaden_attempt_2

ORIGINAL = '(all:"zzqxv" AND all:"qwzxq" AND all:"flibberzzq") AND (cat:cs.CL OR cat:cs.LG)'


def test_attempt_1_drops_the_category_clause_entirely():
    result = _broaden_attempt_1(ORIGINAL)
    assert "cat:" not in result
    assert "cs.CL" not in result and ".LG" not in result


def test_attempt_1_relaxes_and_to_or_and_drops_quotes():
    assert _broaden_attempt_1(ORIGINAL) == "all:zzqxv OR all:qwzxq OR all:flibberzzq"


def test_attempt_1_without_categories():
    assert _broaden_attempt_1('all:"alpha" AND all:"beta"') == "all:alpha OR all:beta"


def test_attempt_1_single_category():
    query = '(all:"alpha" AND all:"beta") AND (cat:cs.CL)'
    assert _broaden_attempt_1(query) == "all:alpha OR all:beta"


def test_attempt_2_output_is_well_formed():
    query = _broaden_attempt_2(_broaden_attempt_1(ORIGINAL))
    assert "all:all:" not in query
    assert query.count("(") == query.count(")")
    assert query == "all:zzqxv OR all:flibberzzq"


def test_attempt_2_keeps_the_two_longest_terms():
    assert _broaden_attempt_2("all:aaa OR all:bbbbbb OR all:cccc") == "all:bbbbbb OR all:cccc"


def test_attempt_2_skips_stopwords():
    query = "all:the OR all:transformer OR all:attention"
    assert _broaden_attempt_2(query) == "all:transformer OR all:attention"


def test_attempt_2_single_term():
    assert _broaden_attempt_2("all:alpha") == "all:alpha"


def test_attempt_2_plain_words():
    assert _broaden_attempt_2("test query") == "all:test OR all:query"