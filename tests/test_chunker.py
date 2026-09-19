"""Tests for chunk_sections: no section boundary crossing, overlap, deterministic IDs, re-chunking."""

from agent.services.chunker import chunk_sections


class TestChunker:
    def test_no_chunk_crosses_section_boundary(self):
        """Chunks from one section must not contain text from another section's title."""
        sections = [
            {"title": "Abstract", "text": "This is the abstract about the paper methodology.", "page_start": 0},
            {"title": "Introduction", "text": "The introduction discusses the background and motivation.", "page_start": 1},
        ]
        chunks = chunk_sections(sections, arxiv_id="2401.12345", target_tokens=400, overlap_tokens=80)
        for chunk in chunks:
            assert chunk.section in ("Abstract", "Introduction")
            # The chunk text should not start with another section's title
            for other_title in ("Abstract", "Introduction"):
                if other_title != chunk.section:
                    assert not chunk.text.startswith(other_title)

    def test_consecutive_chunks_inside_section_overlap(self):
        """Multiple chunks from the same section should have overlapping text."""
        long_text = ("The quick brown fox jumps over the lazy dog. " * 50)
        sections = [{"title": "Body", "text": long_text, "page_start": 0}]
        chunks = chunk_sections(sections, arxiv_id="2401.12345", target_tokens=400, overlap_tokens=80)
        assert len(chunks) > 1
        # Check overlap: end of chunk[i] should appear at start of chunk[i+1]
        for i in range(len(chunks) - 1):
            prev_end_words = chunks[i].text.split()[-20:]
            next_start_words = chunks[i + 1].text.split()[:20]
            overlap = set(prev_end_words) & set(next_start_words)
            assert len(overlap) > 0, f"Chunks {i} and {i+1} should overlap"

    def test_ids_are_arxiv_id_index(self):
        """Chunk IDs must be in the format 'arxiv_id:index'."""
        sections = [{"title": "Abstract", "text": "Some text here. " * 20, "page_start": 0}]
        chunks = chunk_sections(sections, arxiv_id="2401.12345", target_tokens=400, overlap_tokens=80)
        for i, chunk in enumerate(chunks):
            assert chunk.id == f"2401.12345:{i}"

    def test_re_chunking_gives_identical_ids(self):
        """Running chunk_sections twice on the same input must produce the same IDs."""
        sections = [{"title": "Abstract", "text": "Consistent text for re-chunking test. " * 30, "page_start": 0}]
        chunks1 = chunk_sections(sections, arxiv_id="2401.12345", target_tokens=400, overlap_tokens=80)
        chunks2 = chunk_sections(sections, arxiv_id="2401.12345", target_tokens=400, overlap_tokens=80)
        ids1 = [c.id for c in chunks1]
        ids2 = [c.id for c in chunks2]
        assert ids1 == ids2

    def test_empty_section_skipped(self):
        """Empty section text should be skipped, not produce chunks."""
        sections = [
            {"title": "Abstract", "text": "", "page_start": 0},
            {"title": "Introduction", "text": "Some content.", "page_start": 1},
        ]
        chunks = chunk_sections(sections, arxiv_id="2401.12345", target_tokens=400, overlap_tokens=80)
        for chunk in chunks:
            assert chunk.section == "Introduction"
