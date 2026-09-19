"""S7 acceptance check: message counts, abstain behavior, distance calibration.

Run from the repo root:
    python check_abstain.py                # default paper 2401.12345
    python check_abstain.py 1706.03762     # a second paper, to validate the threshold
"""
import sys

from agent.config import get_settings
from agent.graph import build_graph
from agent.nodes.qa import ABSTAIN_MESSAGE
from agent.services.vectorstore import query_chunks

ARXIV_ID = sys.argv[1] if len(sys.argv) > 1 else "2401.12345"

# Multi-turn check: 2 in-paper questions, then 1 off-topic (must abstain).
TURNS = [
    "What method does this paper propose?",
    "What datasets or experiments did they use?",
    "What does this say about the 2026 World Cup?",
]

# Calibration sets (raw best distance is printed for each).
IN_PAPER = [
    "What method does this paper propose?",
    "What datasets or experiments did they use?",
    "What are the limitations of the proposed method?",
    "How does the Wiener beamformer compare to the proposed one?",
    "What is the uncertainty set used?",
    "What noise model do the experiments use?",
]
OFF_TOPIC = [
    "What does this say about the 2026 World Cup?",
    "Who won the last FIFA World Cup?",
    "How do I bake sourdough bread?",
    "What is the capital of France?",
]

# The last three IN_PAPER questions are specific to 2401.12345, so only the
# generic ones are used for other papers.
if ARXIV_ID != "2401.12345":
    IN_PAPER = IN_PAPER[:3]
if ARXIV_ID == "1706.03762":
    IN_PAPER += [
        "What is multi-head attention?",
        "What BLEU score was reported on the WMT 2014 English-to-German task?",
    ]


def main() -> None:
    max_distance = get_settings().abstain_max_distance
    g = build_graph()
    thread = {"configurable": {"thread_id": f"qa_check_{ARXIV_ID}"}}
    g.invoke({"raw_input": ARXIV_ID}, config=thread)

    print("\n" + "=" * 60)
    print(f"MULTI-TURN CHECK  (paper {ARXIV_ID})")
    print("=" * 60)
    ok = True
    last = {}
    for i, q in enumerate(TURNS, 1):
        r = g.invoke({"question": q}, config=thread)
        msgs = r["messages"]
        last = msgs[-1]
        count_ok = len(msgs) == 2 * i
        ok &= count_ok
        print(f"\nturn {i}: count={len(msgs)} (expected {2 * i}) {'OK' if count_ok else 'FAIL'}")
        print("  answer   :", last["content"][:120])
        print("  citations:", len(last.get("citations", [])))

    abstain_ok = last.get("content") == ABSTAIN_MESSAGE and not last.get("citations")
    ok &= abstain_ok
    print(f"\nturn 3 exact abstain message: {'OK' if abstain_ok else 'FAIL'}")

    print("\n" + "=" * 60)
    print(f"DISTANCE CALIBRATION (current ABSTAIN_MAX_DISTANCE={max_distance})")
    print("=" * 60)
    coll = g.get_state(thread).values["collection"]

    def best(q: str) -> float:
        return query_chunks(coll, q, n_results=1)[0]["distance"]

    in_vals = []
    print("\nin-paper (should be <= threshold):")
    for q in IN_PAPER:
        d = best(q)
        in_vals.append(d)
        flag = "" if d <= max_distance else "  <-- WOULD WRONGLY ABSTAIN"
        print(f"  {d:.3f}  {q}{flag}")

    off_vals = []
    print("\noff-topic (should be > threshold):")
    for q in OFF_TOPIC:
        d = best(q)
        off_vals.append(d)
        flag = "" if d > max_distance else "  <-- WOULD WRONGLY ANSWER"
        print(f"  {d:.3f}  {q}{flag}")

    hi_in, lo_off = max(in_vals), min(off_vals)
    print(f"\nlargest in-paper distance : {hi_in:.3f}")
    print(f"smallest off-topic distance: {lo_off:.3f}")
    if hi_in < lo_off:
        print(f"clean gap -> suggested ABSTAIN_MAX_DISTANCE ~ {(hi_in + lo_off) / 2:.3f}")
    else:
        print("RANGES OVERLAP -> an embedding-only gate is too coarse; rely on the")
        print("no-valid-citations abstain path as the safety net.")

    print("\n" + "=" * 60)
    print("RESULT:", "PASS" if ok else "FAIL")
    print("=" * 60)


if __name__ == "__main__":
    main()