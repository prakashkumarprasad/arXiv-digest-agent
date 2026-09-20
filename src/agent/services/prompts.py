"""All prompt templates for the agent."""

import json
from typing import Any


# System prompts
SYSTEM_SUMMARIZE_MAP = """You are an expert research assistant. Your task is to extract 3-5 factual bullet points from a single section of an academic paper.

Rules:
- Each bullet must be a concrete, verifiable fact from the text
- Include the section name as a prefix: "[Section Name] ..."
- Focus on: methods, results, numbers, claims, findings
- Do NOT include generic fluff or meta-commentary
- Do NOT hallucinate - only use information in the provided text
- Keep bullets concise (1-2 sentences each)

Return as JSON array of strings: ["[Section] bullet 1", "[Section] bullet 2", ...]"""

SYSTEM_SUMMARIZE_REDUCE = """You are an expert research assistant. Your task is to synthesize bullet points from multiple sections into a structured briefing.

Rules:
- Group related bullets logically
- Preserve all factual claims with their section attribution
- Write clear, complete sentences
- The output MUST conform exactly to the Briefing schema provided
- For limitations: if the paper doesn't explicitly state limitations, write "Not explicitly stated by the authors; reviewer-inferred: [one reasonable limitation]"
- Followup questions should be specific and research-oriented (3-5 questions)

Return valid JSON matching the Briefing schema."""

SYSTEM_QA = """You are a precise QA assistant that answers questions ONLY from the provided context.

Rules:
- Answer ONLY from the numbered context blocks [S1], [S2], etc.
- Cite sources as [S1], [S2] after each claim
- If context is insufficient, say "The provided context does not contain information about this."
- NEVER use outside knowledge
- Be concise but complete
- If you cite a source, the citation number MUST exist in the provided context

Return ONLY valid JSON with EXACTLY this structure (no extra fields, no markdown):

{
  "answer": "your answer text here",
  "citations": [
    {"chunk_id": "S1", "section": "section name", "text": "relevant snippet from context"},
    {"chunk_id": "S2", "section": "section name", "text": "relevant snippet from context"}
  ],
  "grounded": true

}

Rules for citations:
- chunk_id: use the citation reference like "S1", "S2", etc. from the context blocks
- section: the section name from the context block
- text: a relevant snippet from the context block (max 200 chars)
- Only include citations for claims you make in your answer
- grounded: true if you can answer from context, false if context is insufficient"""

SYSTEM_QUERY_REWRITE = """Rewrite the user's question to be self-contained by resolving pronouns and references using the conversation history.

Rules:
- Replace pronouns and references ("it", "this", "that", "they", "them", "the method") with a SHORT noun phrase for the topic, e.g. "the compared methods" or "the proposed approach"
- Do NOT copy lists, names or numbers from the history into the question; refer to a list by its category instead
- Keep the core question intent, in under 25 words
- Output only the rewritten question as a plain string"""

SYSTEM_QUERY_NORMALIZE = """Normalize the user's topic into an arXiv-friendly search query.

Rules:
- Strip conversational filler ("recent work on", "latest papers about", "what's new in")
- Keep technical terms, method names, dataset names
- Extract 1-3 key search terms
- Suggest 1-2 relevant arXiv categories (e.g., cs.CL, cs.LG, stat.ML)
- Return JSON: {"query": "...", "categories": ["cs.CL", "cs.LG"]}"""

SYSTEM_SELECT_PAPER = """You are an expert researcher evaluating paper relevance. Score each paper 0-10 for relevance to the user's query.

Criteria:
- 10: Directly addresses the query with novel method/results
- 7-9: Highly relevant, addresses core topic
- 4-6: Partially relevant, tangential
- 1-3: Minimally relevant
- 0: Not relevant

Return JSON array: [{"index": int, "score": int, "reason": "..."}, ...]"""

SYSTEM_LIMITATIONS_PASS = """Extract limitations, failure cases, threats to validity, or future work from the provided text chunks.

Rules:
- Look for explicit statements about limitations
- Also identify implicit limitations (small datasets, assumptions, compute constraints)
- If nothing found, return empty list
- Be specific and quote from text where possible

Return JSON array of strings: ["limitation 1", "limitation 2", ...]"""


# User prompt templates
def summarize_map_prompt(section_title: str, section_text: str) -> str:
    return f"""Section: {section_title}

Text:
{section_text[:3000]}

Extract 3-5 factual bullets with section prefix as JSON array."""


def summarize_reduce_prompt(paper_meta: dict[str, Any], all_bullets: list[str]) -> str:
    meta_str = f"""Paper: {paper_meta.get('title', '')}
Authors: {', '.join(paper_meta.get('authors', []))}
Published: {paper_meta.get('published', '')}
Categories: {', '.join(paper_meta.get('categories', []))}
URL: {paper_meta.get('url', '')}
PDF URL: {paper_meta.get('pdf_url', '')}
arXiv ID: {paper_meta.get('arxiv_id', '')}"""

    bullets_str = "\n".join(f"- {b}" for b in all_bullets)

    return f"""{meta_str}

Extracted bullets from all sections:
{bullets_str}

Produce the Briefing JSON with EXACTLY this structure (no extra fields, no nested objects unless specified):

{{
  "arxiv_id": "{paper_meta.get('arxiv_id', '')}",
  "title": "{paper_meta.get('title', '')}",
  "authors": {json.dumps(paper_meta.get('authors', []))},
  "published": "{paper_meta.get('published', '')}",
  "categories": {json.dumps(paper_meta.get('categories', []))},
  "url": "{paper_meta.get('url', '')}",
  "pdf_url": "{paper_meta.get('pdf_url', '')}",
  "why_it_matters": "1 paragraph plain English summary of significance",
  "problem_statement": "What problem does this paper solve?",
  "method": ["bullet 1", "bullet 2", ...],
  "key_results": [
    {{"claim": "specific claim", "evidence": "table/figure/eq reference", "source_section": "section name"}},
    ...
  ],
  "limitations": ["limitation 1", "limitation 2", ...],
  "followup_questions": ["question 1", "question 2", "question 3", "question 4", "question 5"],
  "meta": {{}}
}}

Rules:
- why_it_matters: 1 paragraph, plain English
- method: list of strings (3-7 bullets from the extracted bullets)
- key_results: list of objects with claim, evidence, source_section (3-7 items)
- limitations: list of strings (MUST be non-empty, use "Not explicitly stated by the authors; reviewer-inferred: <one item>" if none found)
- followup_questions: 3-5 specific research questions
- ALL fields are required - do not omit any
- Return ONLY valid JSON, no markdown, no extra text"""


def qa_prompt(question: str, context_blocks: list[dict]) -> str:
    context_str = ""
    for i, block in enumerate(context_blocks, 1):
        context_str += f"\n[S{i}] Section: {block.get('section', 'Unknown')}, Page: {block.get('page_start', 0)}\n{block.get('text', '')[:1500]}\n"

    return f"""Question: {question}

Context blocks:
{context_str}

Answer from context only. Return JSON."""


def _clip(text: str, limit: int = 300) -> str:
    """Shorten long history messages so the rewriter is not tempted to copy their lists."""
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def query_rewrite_prompt(question: str, history: list[dict]) -> str:
    history_str = ""
    for msg in history[-2:]:
        history_str += f"{msg.get('role', 'user')}: {_clip(msg.get('content', ''))}\n"

    return f"""Conversation history:
{history_str}

Current question: {question}

Rewrite to be self-contained:"""


def query_normalize_prompt(raw_input: str) -> str:
    return f"""User input: "{raw_input}"

Normalize to arXiv search query. Return JSON."""


def select_paper_prompt(query: str, candidates: list[dict]) -> str:
    cand_str = ""
    for i, c in enumerate(candidates[:10]):
        cand_str += f"\n[{i}] Title: {c.get('title', '')}\nAbstract: {c.get('summary', '')[:600]}\n"

    return f"""User query: "{query}"

Candidate papers:
{cand_str}

Score each 0-10 for relevance. Return JSON array."""


def limitations_pass_prompt(chunks: list[dict]) -> str:
    chunks_str = ""
    for i, c in enumerate(chunks, 1):
        chunks_str += f"\n[C{i}] Section: {c.get('section', 'Unknown')}\n{c.get('text', '')[:2000]}\n"

    return f"""Text chunks from paper:
{chunks_str}

Extract limitations. Return JSON array of strings."""


def get_prompt(name: str, **kwargs) -> str:
    """Get a prompt template by name with kwargs."""
    prompts = {
        "summarize_map": summarize_map_prompt,
        "summarize_reduce": summarize_reduce_prompt,
        "qa": qa_prompt,
        "query_rewrite": query_rewrite_prompt,
        "query_normalize": query_normalize_prompt,
        "select_paper": select_paper_prompt,
        "limitations_pass": limitations_pass_prompt,
    }
    if name not in prompts:
        raise ValueError(f"Unknown prompt: {name}")
    return prompts[name](**kwargs)