"""arXiv client wrapper with retry logic and search utilities."""

import logging
import time
from typing import Any
from agent.models import PaperMeta
import re

import arxiv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from agent.config import get_settings

logger = logging.getLogger(__name__)


class ArxivClient:
    """Wrapper around arxiv package with retry logic and search helpers."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._client = arxiv.Client(
            page_size=self._settings.arxiv_max_results,
            delay_seconds=3.0,
            num_retries=0,  # We handle retries ourselves
        )

    def search(
        self,
        query: str,
        max_results: int | None = None,
        sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance,
    ) -> list[dict]:
        """Search arXiv and return list of paper metadata dicts."""
        max_results = max_results or self._settings.arxiv_max_results

        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=sort_by,
        )

        results = self._execute_with_retry(lambda: list(self._client.results(search)))
        return [self._paper_to_dict(paper) for paper in results]

    def fetch_metadata(self, arxiv_id: str) -> dict | None:
        """Fetch metadata for a specific arXiv ID."""
        search = arxiv.Search(id_list=[arxiv_id], max_results=1)
        results = self._execute_with_retry(lambda: list(self._client.results(search)))
        if results:
            return self._paper_to_dict(results[0])
        return None

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((arxiv.ArxivError, ConnectionError, TimeoutError)),
        reraise=True,
    )
    def _execute_with_retry(self, func):
        """Execute a function with exponential backoff retry."""
        return func()
    
    @staticmethod
    def _extract_arxiv_id(entry_id: str) -> str:
        """Extract the bare arXiv ID, stripping any trailing version suffix."""
        raw = entry_id.split("/")[-1]
        return re.sub(r"v\d+$", "", raw)
    
    def _paper_to_dict(self, paper: arxiv.Result) -> PaperMeta:
        return PaperMeta(
            arxiv_id=self._extract_arxiv_id(paper.entry_id),
            title=paper.title,
            authors=[author.name for author in paper.authors],
            summary=paper.summary,
            published=paper.published.isoformat() if paper.published else "",
            updated=paper.updated.isoformat() if paper.updated else "",
            categories=paper.categories,
            primary_category=paper.primary_category,
            pdf_url=paper.pdf_url,
            entry_id=paper.entry_id,
            journal_ref=paper.journal_ref,
            doi=paper.doi,
            comment=paper.comment,
        )

    def build_search_query(
        self,
        terms: list[str],
        categories: list[str] | None = None,
    ) -> str:
        """Build an arXiv search query from terms and optional categories."""
        if not terms:
            return ""

        # Join terms with AND, each wrapped in all:"..."
        query_parts = [f'all:"{term}"' for term in terms]
        query = " AND ".join(query_parts)

        if categories:
            cat_filters = " OR ".join([f"cat:{cat}" for cat in categories])
            query = f"({query}) AND ({cat_filters})"

        return query


def get_arxiv_client() -> ArxivClient:
    """Get or create the arXiv client (lazy initialization)."""
    return ArxivClient()

