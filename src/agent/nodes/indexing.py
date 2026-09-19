"""Indexing node: chunk + embed + upsert to ChromaDB."""

import logging
from typing import Any

from agent.config import get_settings
from agent.services.chunker import chunk_sections
from agent.services.vectorstore import upsert_chunks, get_collection_count
from agent.state import AgentState

logger = logging.getLogger(__name__)


def chunk_embed(state: AgentState) -> dict[str, Any]:
    """Chunk sections, embed, and upsert to ChromaDB.

    Implements the chunk_embed node from the graph.
    - Section-aware recursive chunking (never cross section boundary)
    - Target 400 tokens, 80 overlap
    - Deterministic IDs for idempotent upsert
    - Skip if collection already has chunks for this paper
    """
    sections = state.get("sections", [])
    paper = state.get("paper")
    parse_mode = state.get("parse_mode", "full")

    if not sections:
        return {"n_chunks": 0, "collection": None, "warnings": ["No sections to index"]}

    if not paper:
        return {"n_chunks": 0, "collection": None, "errors": [{"code": "NO_PAPER", "node": "chunk_embed", "detail": "No paper in state", "recoverable": False}]}

    # Handle both dict and Pydantic model
    if hasattr(paper, 'model_dump'):
        paper = paper.model_dump(mode="json")

    arxiv_id = paper.get("arxiv_id")
    if not arxiv_id:
        return {"n_chunks": 0, "collection": None, "errors": [{"code": "NO_ARXIV_ID", "node": "chunk_embed", "detail": "Paper missing arxiv_id", "recoverable": False}]}

    # Sanitize arxiv_id for collection name
    safe_id = arxiv_id.replace("/", "_").replace(".", "_")
    collection_name = f"paper_{safe_id}"

    # Check if already indexed (cheap resume)
    existing_count = get_collection_count(collection_name)
    if existing_count > 0:
        logger.info(f"Collection {collection_name} already has {existing_count} chunks, skipping indexing")
        return {"collection": collection_name, "n_chunks": existing_count}

    # Chunk sections
    settings = get_settings()
    chunks = chunk_sections(
        sections,
        arxiv_id,
        target_tokens=settings.chunk_target_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
    )

    if not chunks:
        return {"collection": collection_name, "n_chunks": 0, "warnings": ["No chunks generated from sections"]}

    # Upsert to ChromaDB
    try:
        n_upserted = upsert_chunks(collection_name, chunks)
        logger.info(f"Indexed {n_upserted} chunks for {arxiv_id} in collection {collection_name}")
        return {"collection": collection_name, "n_chunks": n_upserted}

    except Exception as e:
        logger.error(f"Chunk embedding/upsert failed: {e}")
        error = {
            "code": "INDEXING_FAILED",
            "node": "chunk_embed",
            "detail": str(e),
            "recoverable": False,
        }
        return {"collection": collection_name, "n_chunks": 0, "errors": [error]}