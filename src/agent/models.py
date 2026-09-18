"""Pydantic models for the arXiv Digest Agent."""

from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, Field, HttpUrl


class PaperMeta(BaseModel):
    """Metadata for an arXiv paper."""

    arxiv_id: str
    title: str
    authors: list[str]
    summary: str
    published: datetime
    updated: datetime
    categories: list[str]
    primary_category: str
    pdf_url: HttpUrl
    entry_id: HttpUrl
    journal_ref: str | None = None
    doi: str | None = None
    comment: str | None = None


class Chunk(BaseModel):
    """A text chunk with metadata for vector storage."""

    id: str
    arxiv_id: str
    text: str
    section: str
    chunk_index: int
    page_start: int
    char_start: int
    embedding: list[float] | None = None


class KeyResult(BaseModel):
    """A key result from the paper with evidence."""

    claim: str
    evidence: str
    source_section: str


class Briefing(BaseModel):
    """Structured briefing for a paper."""

    arxiv_id: str
    title: str
    authors: list[str]
    published: str
    categories: list[str]
    url: HttpUrl
    pdf_url: HttpUrl
    why_it_matters: str
    problem_statement: str
    method: list[str]
    key_results: list[KeyResult]
    limitations: list[str]
    followup_questions: list[str]
    meta: dict[str, Any] = Field(default_factory=dict)


class QAAnswer(BaseModel):
    """Answer to a question with citations."""

    answer: str
    citations: list[dict[str, Any]]
    grounded: bool


class AgentError(BaseModel):
    """Structured error for state tracking."""

    code: str
    node: str
    detail: str
    recoverable: bool = True


# Type aliases for state
Intent = Literal["paper_lookup", "topic_search", "unclear"]
ParseMode = Literal["full", "degraded_abstract_only", "failed"]