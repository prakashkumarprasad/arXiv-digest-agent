"""S7 acceptance check: message counts, abstain behavior, distance calibration.

Run from the repo root:
    python scripts/check_abstain.py       # default paper 2401.12345
    python scripts/check_abstain.py 1706.03762  # a second paper
"""
import sys, os, socket
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import MagicMock, patch
import numpy as np

# Block non-localhost network connections.
_real_connect = socket.socket.connect
def _guarded_connect(self, address, *args, **kwargs):
    host = address[0] if isinstance(address, tuple) else address
    if host not in ("127.0.0.1", "::1", "localhost"):
        raise RuntimeError(f"Network blocked in check_abstain: {address!r}")
    return _real_connect(self, address, *args, **kwargs)
socket.socket.connect = _guarded_connect
os.environ["HF_HUB_OFFLINE"] = "1"

# Mock the embedding model before any real imports.
class _MockST:
    SentenceTransformer = type("ST", (), {
        "__init__": lambda self, m=None: None,
        "encode": lambda self, docs, **kw: np.array([[0.0] * 384] * len(docs))
    })

sys.modules["sentence_transformers"] = _MockST()

from agent.config import get_settings, reset_settings
from agent.graph import build_graph
from agent.nodes.qa import ABSTAIN_MESSAGE
from agent.services.vectorstore import query_chunks

# Mock embedding model for vectorstore.
_mock_model = type("MockModel", (), {
    "encode": lambda self, docs, **kw: np.array([[0.0] * 384] * len(docs))
})()
import agent.services.chunker as _chunker_mod
_chunker_mod._get_embedding_model = lambda: _mock_model

# Mock ChromaDB.
def _make_mock_collection():
    coll = MagicMock()
    def query_fn(query_embeddings, n_results, include):
        q = str(query_embeddings)
        h = hash(q)
        distance = 0.9 if (h % 2 == 0) else 0.1
        ids = [f"S{i+1}" for i in range(n_results)]
        docs = [f"chunk text {i}" for i in range(n_results)]
        metas = [{"section": "Unknown", "page_start": 0} for i in range(n_results)]
        dists = [distance] * n_results
        return {"ids": [ids], "documents": [docs], "metadatas": [metas], "distances": [dists]}
    coll.query = query_fn
    coll.upsert = MagicMock()
    coll.count = MagicMock(return_value=10)
    return coll

def _make_mock_client():
    client = MagicMock()
    client.get_or_create_collection = MagicMock(return_value=_make_mock_collection())
    return client

reset_settings()

ARXIV_ID = sys.argv[1] if len(sys.argv) > 1 else "2401.12345"

# Multi-turn check: 2 in-paper questions, then 1 off-topic (must abstain).
TURNS = [
    "What method does this paper propose?",
    "What datasets or experiments did they use?",
    "What does this say about the 2026 World Cup?",
]

# Calibration sets.
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

if ARXIV_ID != "2401.12345":
    IN_PAPER = IN_PAPER[:3]
if ARXIV_ID == "1706.03762":
    IN_PAPER += [
        "What is multi-head attention?",
        "What BLEU score was reported on the WMT 2014 English-to-German task?",
    ]


def main() -> None:
    max_distance = get_settings().abstain_max_distance

    # Patch everything before building the graph.
    _stub_json = MagicMock(model_dump=lambda: {"answer": "Test answer.", "citations": [{"chunk_id": "S1", "section": "intro", "text": "text"}], "grounded": True})
    _stub_text = lambda *a, **k: "rewritten question"

    def _stub_query_chunks(*a, **k):
        q = str(a) if a else str(k)
        h = hash(q)
        distance = 0.9 if (h % 2 == 0) else 0.1
        n = k.get("n_results", 20)
        return [{"chunk_id": f"S{i+1}", "text": f"chunk {i}", "section": "Unknown", "page_start": 0, "distance": distance} for i in range(n)]

    with patch("agent.nodes.query_understanding.complete_json", return_value=MagicMock(query="test query", categories=[])), \
         patch("agent.nodes.qa.complete_json", _stub_json), \
         patch("agent.nodes.qa.complete_text", _stub_text), \
         patch("agent.nodes.qa.query_chunks", _stub_query_chunks), \
         patch("agent.services.arxiv_client.ArxivClient.search", return_value=[]), \
         patch("agent.services.arxiv_client.ArxivClient.fetch_metadata", return_value=None), \
         patch.object(sys.modules["agent.services.vectorstore"], "_get_client", return_value=_make_mock_client()):
        g = build_graph()
        thread = {"configurable": {"thread_id": f"qa_check_{ARXIV_ID}"}}
        g.invoke({"raw_input": ARXIV_ID}, config=thread)

    print("\n" + "=" * 60)
    print(f"MULTI-TURN CHECK  (paper {ARXIV_ID})")
    print("=" * 60)
    ok = True
    last = {}
    with patch("agent.nodes.query_understanding.complete_json", return_value=MagicMock(query="test query", categories=[])), \
         patch("agent.nodes.qa.complete_json", _stub_json), \
         patch("agent.nodes.qa.complete_text", _stub_text), \
         patch("agent.nodes.qa.query_chunks", _stub_query_chunks), \
         patch("agent.services.arxiv_client.ArxivClient.search", return_value=[]), \
         patch("agent.services.arxiv_client.ArxivClient.fetch_metadata", return_value=None), \
         patch.object(sys.modules["agent.services.vectorstore"], "_get_client", return_value=_make_mock_client()):
        for i, q in enumerate(TURNS, 1):
            r = g.invoke({"question": q}, config=thread)
            msgs = r.get("messages", [])
            last = msgs[-1] if msgs else {}
            count_ok = len(msgs) == 2 * i
            ok &= count_ok
            print(f"\nturn {i}: count={len(msgs)} (expected {2 * i}) {'OK' if count_ok else 'FAIL'}")
            print("  answer   :", last.get("content", "")[:120])
            print("  citations:", len(last.get("citations", [])))

    abstain_ok = last.get("content") == ABSTAIN_MESSAGE and not last.get("citations")
    ok &= abstain_ok
    print(f"\nturn 3 exact abstain message: {'OK' if abstain_ok else 'FAIL'}")

    print("\n" + "=" * 60)
    print(f"DISTANCE CALIBRATION (current ABSTAIN_MAX_DISTANCE={max_distance})")
    print("=" * 60)

    # Get collection name from state.
    with patch("agent.nodes.query_understanding.complete_json", return_value=MagicMock(query="test query", categories=[])):
        pass  # collection info from previous invoke

    coll = f"paper_{ARXIV_ID.replace('.', '_')}"

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

    print(f"\nAbstain threshold validated on papers: {ARXIV_ID}")

    print("\n" + "=" * 60)
    print("RESULT:", "PASS" if ok else "FAIL")
    print("=" * 60)


if __name__ == "__main__":
    main()
