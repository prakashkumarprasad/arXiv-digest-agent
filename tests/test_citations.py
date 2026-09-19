"""Tests for [Sn] -> chunk_id citation mapping: order preserved, unknown Sn dropped, final citations have chunk_id/section/page/snippet."""

from agent.nodes.qa import _prepare_citations, _validate_citations, _build_context_blocks


class TestCitationMapping:
    def test_sn_to_chunk_id_keeps_prompt_order(self):
        """The [Sn] -> chunk_id mapping preserves the prompt's context order."""
        mmr_chunks = [
            {"chunk_id": "arxiv:0", "section": "Abstract", "page_start": 1, "text": "abstract text"},
            {"chunk_id": "arxiv:1", "section": "Method", "page_start": 3, "text": "method text"},
            {"chunk_id": "arxiv:2", "section": "Results", "page_start": 5, "text": "results text"},
        ]
        citation_map = {f"S{i+1}": chunk["chunk_id"] for i, chunk in enumerate(mmr_chunks)}
        assert citation_map == {"S1": "arxiv:0", "S2": "arxiv:1", "S3": "arxiv:2"}

    def test_unknown_sn_is_dropped(self):
        """An unknown Sn (not in citation_map) is dropped during validation."""
        citation_map = {"S1": "arxiv:0", "S2": "arxiv:1"}
        answer_data = {
            "answer": "Some answer",
            "citations": [
                {"chunk_id": "S1", "section": "Abstract", "text": "text1"},
                {"chunk_id": "S99", "section": "Unknown", "text": "text2"},
            ],
            "grounded": True,
        }
        validated = _validate_citations(answer_data, citation_map)
        assert len(validated["citations"]) == 1
        assert validated["citations"][0]["chunk_id"] == "S1"

    def test_final_citations_have_required_fields(self):
        """Final citation objects contain chunk_id, section, page, snippet from retrieved chunks."""
        citation_map = {"S1": "arxiv:0", "S2": "arxiv:1"}
        mmr_chunks = [
            {"chunk_id": "arxiv:0", "section": "Abstract", "page_start": 1, "text": "abstract text here"},
            {"chunk_id": "arxiv:1", "section": "Method", "page_start": 3, "text": "method details"},
        ]
        validated_citations = [
            {"chunk_id": "S1", "section": "Abstract", "text": "abstract text"},
            {"chunk_id": "S2", "section": "Method", "text": "method details"},
        ]
        citations = _prepare_citations(validated_citations, citation_map, mmr_chunks)

        assert len(citations) == 2
        for cit in citations:
            assert "chunk_id" in cit
            assert "section" in cit
            assert "page" in cit
            assert "snippet" in cit
        # chunk_id is the actual chunk_id, not the S1/S2 label
        assert citations[0]["chunk_id"] == "arxiv:0"
        assert citations[1]["chunk_id"] == "arxiv:1"
        # page is from the retrieved chunk
        assert citations[0]["page"] == 1
        assert citations[1]["page"] == 3
        # snippet is truncated from the chunk text
        assert citations[0]["snippet"] == "abstract text here"[:300]

    def test_build_context_blocks_preserves_order(self):
        """_build_context_blocks preserves the order of mmr_chunks for prompt context."""
        mmr_chunks = [
            {"chunk_id": "arxiv:0", "section": "Abstract", "page_start": 1, "text": "text"},
            {"chunk_id": "arxiv:1", "section": "Method", "page_start": 3, "text": "text"},
        ]
        blocks = _build_context_blocks(mmr_chunks)
        assert len(blocks) == 2
        assert blocks[0]["section"] == "Abstract"
        assert blocks[1]["section"] == "Method"