"""QA node: retrieve → ground → answer / abstain."""

import logging
import re
from typing import Any

from agent.config import get_settings
from agent.models import QAAnswer
from agent.services.llm import complete_json, complete_text
from agent.services.prompts import SYSTEM_QA, SYSTEM_QUERY_REWRITE, qa_prompt, query_rewrite_prompt
from agent.services.vectorstore import query_chunks
from agent.state import AgentState

logger = logging.getLogger(__name__)

ABSTAIN_MESSAGE = "That isn't covered in this paper."
ANSWER_FAILED_MESSAGE = "I couldn't generate a proper answer from the context."


# Words that signal a follow-up question needing context from earlier turns.
_REFERENCE_WORDS = {
    "it", "its", "they", "them", "their", "these", "those", "that",
    "he", "she", "one", "ones", "former", "latter", "same", "such",
}


def _needs_rewrite(question: str) -> bool:
    """Return True only for follow-ups (short, or containing reference words).

    Standalone questions are used verbatim so the rewrite step cannot pull
    unrelated history into an off-topic question and defeat the abstain gate.
    """
    words = re.findall(r"[a-z']+", question.lower())
    return len(words) <= 5 or any(w in _REFERENCE_WORDS for w in words) 


def qa_node(state: AgentState) -> dict[str, Any]:
    """Answer a question about the paper using grounded retrieval.

    Implements the qa_node from the graph.
    - Query rewrite if conversation history exists
    - Retrieve top 20 → MMR rerank (λ=0.6) → keep 6
    - Abstain gate: best raw cosine distance > abstain_max_distance → canned response
    - Answer only from context blocks with citations [S1], [S2]
    - Post-check: every [Sn] cited must exist in retrieved set
    - Abstain if the LLM answer has no valid citations
    - Return only the NEW messages for this turn (messages uses operator.add)
    """
    question = state.get("question")
    if not question:
        return {"errors": [{"code": "NO_QUESTION", "node": "qa_node", "detail": "No question provided", "recoverable": False}]}

    collection = state.get("collection")
    if not collection:
        return {"errors": [{"code": "NO_COLLECTION", "node": "qa_node", "detail": "No collection available for retrieval", "recoverable": False}]}

    messages = state.get("messages", [])
    settings = get_settings()
    max_distance = settings.abstain_max_distance

    try:
        # Step 1: Query rewrite if conversation history exists
        rewritten_question = _rewrite_query(question, messages)

        # Step 2: Retrieve top 20
        retrieved_chunks = query_chunks(collection, rewritten_question, n_results=20)

        if not retrieved_chunks:
            return {
                "retrieved": [],
                "messages": [{"role": "user", "content": question}, {"role": "assistant", "content": ABSTAIN_MESSAGE, "citations": []}],
            }

        # Step 3: Abstain gate — best raw cosine distance among retrieved chunks.
        # bge-small scores are compressed, so we gate on distance directly
        # instead of converting to a similarity (see BUILD_SPEC §11, Stage 7).
        best_distance = min(c.get("distance", 1.0) for c in retrieved_chunks)
        if best_distance > max_distance:
            logger.info(f"Abstaining: best_distance={best_distance:.4f} > max={max_distance}")
            return {
                "retrieved": retrieved_chunks[:6],
                "messages": [{"role": "user", "content": question}, {"role": "assistant", "content": ABSTAIN_MESSAGE, "citations": []}],
            }

        # Step 3b: MMR rerank (λ=0.6) → keep 6
        mmr_chunks = _mmr_rerank(retrieved_chunks, rewritten_question, lambda_mult=0.6, top_k=6)

        if not mmr_chunks:
            return {
                "retrieved": [],
                "messages": [{"role": "user", "content": question}, {"role": "assistant", "content": ABSTAIN_MESSAGE, "citations": []}],
            }

        # Build citation mapping: S1, S2, ... → chunk_id (same order as prompt blocks)
        citation_map = {f"S{i+1}": chunk["chunk_id"] for i, chunk in enumerate(mmr_chunks)}

        # Step 4: Answer from context with citations
        context_blocks = _build_context_blocks(mmr_chunks)
        answer = _generate_answer(rewritten_question, context_blocks, citation_map)

        # Step 5: Post-check citations
        validated_answer = _validate_citations(answer, citation_map)

        # Step 6: Prepare citations with full metadata
        citations = _prepare_citations(validated_answer.get("citations", []), citation_map, mmr_chunks)

        # Step 6b: An answer with no valid citations is not grounded. Return the
        # canned abstain message (unless the LLM call itself failed).
        if not citations and validated_answer.get("answer") != ANSWER_FAILED_MESSAGE:
            logger.info("Abstaining: answer had no valid citations")
            return {
                "retrieved": mmr_chunks,
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": ABSTAIN_MESSAGE, "citations": []},
                ],
            }

        qa_answer = QAAnswer(
            answer=validated_answer.get("answer", ""),
            citations=citations,
            grounded=validated_answer.get("grounded", True),
        )

        # Convert to dict for proper state serialization
        qa_answer_dict = qa_answer.model_dump(mode="json")

        # Step 7: Build only the NEW messages for this turn — the `messages`
        # state field has an operator.add reducer (see state.py), so LangGraph
        # appends these to the existing history automatically. Returning
        # [*messages, ...] here would double-append every prior turn on every
        # new turn.
        new_messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": qa_answer_dict["answer"], "citations": qa_answer_dict["citations"]},
        ]

        return {
            "retrieved": mmr_chunks,
            "messages": new_messages,
        }

    except Exception as e:
        logger.error(f"QA node failed: {e}")
        error = {
            "code": "QA_FAILED",
            "node": "qa_node",
            "detail": str(e),
            "recoverable": False,
        }
        return {"errors": [error]}


# A rewrite longer than this is treated as list-stuffing: the model copied names from the
# history into the query, which dragged retrieval toward a chunk that mentions them all.
MAX_REWRITE_WORDS = 30


def _rewrite_query(question: str, messages: list[dict]) -> str:
    """Rewrite query to resolve pronouns using the last 2 turns of history.

    Falls back to the original question if the rewrite fails, is empty, or is
    longer than MAX_REWRITE_WORDS.
    """
    if len(messages) == 0 or not _needs_rewrite(question):
        return question

    try:
        history = messages[-4:]  # last 2 user-assistant pairs (4 messages max)
        prompt = query_rewrite_prompt(question, history)
        rewritten = complete_text(SYSTEM_QUERY_REWRITE, prompt).strip()
    except Exception as e:
        logger.warning(f"Query rewrite failed: {e}, using original question")
        return question

    n_words = len(rewritten.split())
    if n_words == 0 or n_words > MAX_REWRITE_WORDS:
        logger.warning(f"Query rewrite unusable ({n_words} words), using original question")
        return question

    logger.info(f"Rewrote query: {question!r} -> {rewritten!r}")
    return rewritten


def _mmr_rerank(chunks: list[dict], query: str, lambda_mult: float = 0.6, top_k: int = 6) -> list[dict]:
    """Maximal Marginal Relevance reranking for diversity.

    MMR score = lambda * relevance - (1 - lambda) * max_similarity_to_selected
    """
    if not chunks:
        return []

    selected = []
    remaining = list(chunks)

    # Sort by relevance (ascending distance = descending similarity)
    remaining.sort(key=lambda x: x.get("distance", 1.0))

    while remaining and len(selected) < top_k:
        if not selected:
            # First item: highest relevance
            best = remaining.pop(0)
            selected.append(best)
            continue

        # Compute MMR scores for remaining items
        best_idx = -1
        best_score = -float("inf")

        for i, chunk in enumerate(remaining):
            relevance = 1.0 - chunk.get("distance", 1.0)

            # Max similarity to already selected
            max_sim = 0.0
            for sel in selected:
                sim = _compute_similarity(chunk.get("text", ""), sel.get("text", ""))
                if sim > max_sim:
                    max_sim = sim

            mmr_score = lambda_mult * relevance - (1 - lambda_mult) * max_sim

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i

        if best_idx >= 0:
            selected.append(remaining.pop(best_idx))
        else:
            break

    return selected


def _compute_similarity(text1: str, text2: str) -> float:
    """Compute Jaccard similarity between two texts (approximate)."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    if not words1 or not words2:
        return 0.0
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    return len(intersection) / len(union)


def _build_context_blocks(chunks: list[dict]) -> list[dict]:
    """Build context blocks for the prompt."""
    context_blocks = []
    for i, chunk in enumerate(chunks, 1):
        context_blocks.append({
            "index": i,
            "section": chunk.get("section", "Unknown"),
            "page_start": chunk.get("page_start", 0),
            "text": chunk.get("text", "")[:1500],
        })
    return context_blocks


def _generate_answer(question: str, context_blocks: list[dict], citation_map: dict[str, str]) -> dict[str, Any]:
    """Generate answer using LLM with citation validation."""
    prompt = qa_prompt(question, context_blocks)

    # Define schema for QA answer with citations - matches LLM's natural output
    from pydantic import BaseModel
    from typing import List, Optional

    class CitationRef(BaseModel):
        chunk_id: str  # LLM uses S1, S2 as citation references
        section: str
        text: Optional[str] = None  # LLM may omit this

    class QAResponse(BaseModel):
        answer: str
        citations: List[CitationRef]
        grounded: bool

    try:
        response = complete_json(QAResponse, SYSTEM_QA, prompt)
        return response.model_dump()
    except Exception as e:
        logger.warning(f"First QA attempt failed: {e}")
        # Retry with error feedback
        retry_prompt = f"{prompt}\n\nPrevious response failed: {e}\n\nReturn ONLY valid JSON matching the schema."
        try:
            response = complete_json(QAResponse, SYSTEM_QA, retry_prompt)
            return response.model_dump()
        except Exception as e2:
            logger.error(f"Second QA attempt failed: {e2}")
            # Fallback
            return {
                "answer": ANSWER_FAILED_MESSAGE,
                "citations": [],
                "grounded": False,
            }


def _validate_citations(answer_data: dict[str, Any], citation_map: dict[str, str]) -> dict[str, Any]:
    """Post-check: every [Sn] cited must exist in the retrieved set."""
    answer = answer_data.get("answer", "")
    citations = answer_data.get("citations", [])

    valid_citation_ids = set(citation_map.keys())
    validated_citations = []

    for cit in citations:
        # LLM returns chunk_id as the citation reference (S1, S2, etc.)
        # Check if this citation reference exists in our map
        chunk_id = cit.get("chunk_id", "")
        if chunk_id in valid_citation_ids:
            validated_citations.append(cit)
        else:
            logger.warning(f"Dropped citation with unknown chunk_id: {chunk_id}")

    # Also check answer text for orphaned citations
    # Look for [S1], [S2], etc. in answer text
    cited_in_text = set(re.findall(r"\[S(\d+)\]", answer))
    for cid_str in cited_in_text:
        cid = f"S{cid_str}"
        if cid not in valid_citation_ids:
            logger.warning(f"Answer references unknown citation: {cid}")

    return {
        "answer": answer,
        "citations": validated_citations,
        "grounded": answer_data.get("grounded", True),
    }


def _prepare_citations(validated_citations: list[dict], citation_map: dict[str, str], chunks: list[dict]) -> list[dict]:
    """Prepare final citations with full metadata."""
    # Build chunk lookup - citation_map maps S1->actual_chunk_id, etc.
    actual_chunk_lookup = {}
    for sid, actual_cid in citation_map.items():
        for chunk in chunks:
            if chunk["chunk_id"] == actual_cid:
                actual_chunk_lookup[actual_cid] = chunk
                break

    final_citations = []
    for cit in validated_citations:
        # cit.chunk_id is the citation reference (S1, S2, etc.)
        # Map it to the actual chunk_id
        actual_chunk_id = citation_map.get(cit.get("chunk_id", ""))
        if actual_chunk_id and actual_chunk_id in actual_chunk_lookup:
            chunk = actual_chunk_lookup[actual_chunk_id]
            final_citations.append({
                "chunk_id": actual_chunk_id,
                "section": chunk.get("section", "Unknown"),
                "page": chunk.get("page_start", 0),
                "snippet": chunk.get("text", "")[:300],
            })

    return final_citations