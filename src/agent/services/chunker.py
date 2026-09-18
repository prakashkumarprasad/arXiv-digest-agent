"""Section-aware recursive chunking for arXiv papers."""

import re
from typing import Any

from sentence_transformers import SentenceTransformer

from agent.config import get_settings
from agent.models import Chunk

# Global model cache (lazy initialization)
_embedding_model: SentenceTransformer | None = None


def _get_embedding_model() -> SentenceTransformer:
    """Get or create the embedding model (lazy initialization)."""
    global _embedding_model
    if _embedding_model is None:
        settings = get_settings()
        _embedding_model = SentenceTransformer(settings.embedding_model)
    return _embedding_model


def _count_tokens(text: str) -> int:
    """Approximate token count (roughly 4 chars per token for English)."""
    return len(text) // 4


def _split_text_into_chunks(text: str, target_tokens: int, overlap_tokens: int) -> list[str]:
    """Split text into overlapping chunks using character-based sliding window.

    This avoids sentence-splitting issues with academic text (initials, citations, etc.).
    Strategy:
    1. Use sliding window over characters: window ~target_tokens*4 chars, stride ~(target-overlap)*4 chars
    2. Adjust window boundaries to nearest sentence/paragraph boundary when possible
    3. Guarantee no gaps and proper overlap
    """
    if _count_tokens(text) <= target_tokens:
        return [text] if text.strip() else []

    target_chars = target_tokens * 4
    overlap_chars = overlap_tokens * 4
    stride_chars = target_chars - overlap_chars

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + target_chars, text_len)

        # Try to extend to a sentence boundary (period, newline) within reasonable range
        if end < text_len:
            # Look for sentence end within +200 chars
            search_end = min(end + 200, text_len)
            # Find last sentence-ending punctuation followed by space/newline
            match = None
            for m in re.finditer(r'[.!?](?:\s|$)', text[end:search_end]):
                match = m
            if match:
                end = end + match.end()
            else:
                # Fall back to paragraph boundary
                para_match = None
                for m in re.finditer(r'\n\n', text[end:search_end]):
                    para_match = m
                if para_match:
                    end = end + para_match.end()

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_len:
            break

        # Next start = current end - overlap
        start = end - overlap_chars
        # Ensure we make progress
        if start <= chunks[-1].__len__() and len(chunks) > 1:
            start = end

    # Now apply overlap: each chunk after first gets prepended with tail of previous
    if len(chunks) <= 1:
        return chunks

    overlap_words = max(1, overlap_tokens * 4 // 5)
    overlapped = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_words = chunks[i - 1].split()
        overlap_text = " ".join(prev_words[-overlap_words:]) if len(prev_words) > overlap_words else chunks[i - 1]
        overlapped.append(overlap_text + " " + chunks[i])

    return overlapped


def chunk_sections(
    sections: list[dict],
    arxiv_id: str,
    target_tokens: int = 400,
    overlap_tokens: int = 80,
) -> list[Chunk]:
    """Chunk sections into token-bounded pieces without crossing section boundaries.

    Args:
        sections: List of dicts with keys 'title', 'text', 'page_start'
        arxiv_id: Paper identifier for chunk IDs
        target_tokens: Target tokens per chunk (default 400)
        overlap_tokens: Overlap tokens between chunks (default 80)

    Returns:
        List of Chunk objects with deterministic IDs for idempotent upsert.
    """
    chunks = []
    chunk_index = 0
    char_offset = 0

    for section in sections:
        section_title = section.get("title", "Unknown")
        section_text = section.get("text", "")
        page_start = section.get("page_start", 0)

        if not section_text.strip():
            continue

        # Split section text into chunks
        section_chunks = _split_text_into_chunks(section_text, target_tokens, overlap_tokens)

        for chunk_text in section_chunks:
            chunk_id = f"{arxiv_id}:{chunk_index}"

            chunk = Chunk(
                id=chunk_id,
                arxiv_id=arxiv_id,
                text=chunk_text,
                section=section_title,
                chunk_index=chunk_index,
                page_start=page_start,
                char_start=char_offset,
            )
            chunks.append(chunk)
            chunk_index += 1
            char_offset += len(chunk_text)

    return chunks