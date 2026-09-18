# BUILD_SPEC.md — Autonomous arXiv Digest & QA Agent

> **This file is the contract for the coding agent.** Keep it at the repo root.
> Use the **standard stage-launch prompt** in §0.1 verbatim for every stage — it is self-contained and requires no per-stage editing.

---

## 0.1 Standard stage-launch prompt (use verbatim, every stage)

Copy this whole block into opencode, replacing only `<N>`:

```
Read BUILD_SPEC.md in full, including §0 (ground rules) and §6 (exact interfaces).
Implement STAGE <N> only, from the table in §9.

Touch only the files listed for STAGE <N>. Do not modify any other file.

Every function, method, and class you write for this stage MUST match its signature
in §6 "Exact interfaces" exactly — same name, same parameter names, same return type
(the Pydantic model from models.py, never a raw dict, unless §6 explicitly allows dict).
If §6 does not cover something you need to build, choose the smallest reasonable
signature yourself, add it to a new "### Stage <N> additions" subsection under §6
in BUILD_SPEC.md, and use that from then on.

Before declaring this stage complete, you must:
1. Run every acceptance criterion for STAGE <N> from §9 yourself, in the real
   terminal — not described, actually executed.
2. Paste the real, unedited terminal output for each check.
3. Confirm `python -c "import agent"` succeeds after all edits.
4. Confirm every function/class you added matches §6 exactly — restate each
   signature you implemented next to its §6 spec and confirm they match, or
   explain the deviation and update §6 yourself before proceeding.
5. List every file touched and the specific function/class added or changed in
   each — not just a file list.

Do not say "complete" or "acceptance criteria pass" until steps 1-4 are done and
their real output is shown. If a check fails, fix it and re-run it — do not move on
and do not summarize a fix as done without re-running the check.

Stop after this stage. Do not start the next stage without being asked.
```

---

## 0. Ground rules for the coding agent

1. **Python 3.11.** Type hints on every public function. Google-style docstrings, one line minimum.
2. **No frameworks beyond the pinned stack.** No FastAPI, no Streamlit, no React. CLI only.
3. **Every node is a pure-ish function** `(state: AgentState) -> dict` returning only the keys it changed. No hidden globals.
4. **No silent failures.** Every external call (arXiv, HTTP, LLM, embedding) is wrapped, retried where sensible, and appends to `state["errors"]` with a machine-readable `code`.
5. **No network calls at import time.** Models and clients are lazily constructed via `config.py`.
6. **Do not invent APIs.** If unsure of a library signature, write the smallest possible adapter in `services/` and add a TODO comment rather than guessing across three files.
7. **Small commits.** One stage = one commit = one green test run.
8. **Never hardcode secrets.** Read from env via `config.py` only.

---

## 1. Stack (fixed — do not substitute)

| Concern | Choice | Why |
|---|---|---|
| Orchestration | **LangGraph** (`langgraph`) | Explicit nodes/edges/shared state + built-in checkpointer, which is exactly what the rubric asks for |
| LLM | **Groq free tier** (`llama-3.3-70b-versatile`), pluggable | Fast, free, generous limits |
| LLM fallback | **Google AI Studio (Gemini free)** and **Ollama** (`qwen2.5:7b-instruct`) | Grader can run with zero keys via Ollama |
| Embeddings | **`BAAI/bge-small-en-v1.5`** via `sentence-transformers`, local | No API key, 512-token window, 384-dim, CPU-fast |
| Vector DB | **ChromaDB**, persistent local (`./.data/chroma`) | Local, zero setup, survives process restart |
| PDF parse | **PyMuPDF** primary, **pdfplumber** fallback | Fast + layout fallback |
| arXiv | **`arxiv`** Python package (official Atom API) | No scraping |
| Validation | **Pydantic v2** | Structured briefing + repair loop |
| CLI | **Typer** + **Rich** | Clean UX, zero effort |
| State persistence | **LangGraph `SqliteSaver`** (`./.data/sessions.sqlite`) | Real answer to "how does state persist between summarize and QA" |
| Tests | **pytest** | 6–8 focused tests, no coverage theatre |

---

## 2. Repository layout (create exactly this)

```
arxiv-digest-agent/
├── README.md
├── BUILD_SPEC.md
├── pyproject.toml            # or requirements.txt
├── .env.example
├── .gitignore                # .env, .data/, __pycache__, *.pdf cache
├── Makefile                  # make install / run / test / demo
├── docs/
│   └── architecture.md       # mermaid graph + state table
├── examples/
│   ├── briefing_2401.example.json
│   └── sample_session.md     # pasted demo run for the README
├── src/agent/
│   ├── __init__.py
│   ├── config.py             # env loading, settings dataclass, paths
│   ├── models.py             # Pydantic: PaperMeta, Chunk, Briefing, QAAnswer
│   ├── state.py              # AgentState TypedDict
│   ├── graph.py              # build_graph(): nodes, edges, conditional routing
│   ├── cli.py                # Typer app: digest / ask / sessions
│   ├── nodes/
│   │   ├── query_understanding.py
│   │   ├── retrieval.py      # arXiv search + broaden-on-zero
│   │   ├── selection.py      # rank + select
│   │   ├── fetch_parse.py    # download PDF, parse, fallback chain
│   │   ├── indexing.py       # chunk + embed + upsert
│   │   ├── summarize.py      # map-reduce briefing + limitations guard
│   │   └── qa.py             # retrieve → ground → answer / abstain
│   └── services/
│       ├── arxiv_client.py
│       ├── pdf_parser.py
│       ├── chunker.py
│       ├── vectorstore.py
│       ├── llm.py            # provider abstraction + JSON repair
│       └── prompts.py        # all prompt templates, one place
└── tests/
    ├── test_query_understanding.py
    ├── test_chunker.py
    ├── test_parser_fallback.py
    ├── test_selection_zero_results.py
    └── test_qa_abstain.py
```

---

## 3. State shape (`state.py`) — implement verbatim

```python
class AgentState(TypedDict, total=False):
    # --- input ---
    raw_input: str                 # user's topic or arXiv id/url
    intent: Literal["paper_lookup", "topic_search", "unclear"]
    arxiv_id: str | None
    search_query: str | None       # normalized query sent to arXiv

    # --- retrieval ---
    candidates: list[dict]         # PaperMeta dicts from arXiv
    selection_reason: str | None
    paper: dict | None             # chosen PaperMeta

    # --- parsing ---
    pdf_path: str | None
    sections: list[dict]           # [{"title","text","page_start"}]
    full_text: str | None
    parse_mode: Literal["full", "degraded_abstract_only", "failed"]

    # --- indexing ---
    collection: str | None         # chroma collection name = f"paper_{arxiv_id_safe}"
    n_chunks: int

    # --- output ---
    briefing: dict | None          # Briefing.model_dump()

    # --- QA ---
    question: str | None
    messages: list[dict]           # [{"role","content","citations":[...]}]
    retrieved: list[dict]          # last retrieval, for transparency

    # --- control ---
    errors: list[dict]             # [{"code","node","detail","recoverable":bool}]
    retries: dict[str, int]        # node_name -> attempts
    warnings: list[str]
```

---

## 4. The graph (`graph.py`)

```
                       ┌──────────────────────┐
                       │  query_understanding │
                       └──────────┬───────────┘
              intent=paper_lookup │ intent=topic_search
                  ┌───────────────┴───────────────┐
                  ▼                               ▼
          ┌───────────────┐              ┌────────────────┐
          │ fetch_metadata│              │  search_arxiv  │◄──┐
          └───────┬───────┘              └───────┬────────┘   │ zero results
                  │                              │            │ & retries<2
                  │                    ┌─────────┴─────────┐  │
                  │                 0  │        1..N       │  │
                  │                    ▼                   ▼  │
                  │            ┌──────────────┐   ┌────────────────┐
                  │            │ broaden_query├───┘   │ select_paper │
                  │            └──────┬───────┘       └───────┬──────┘
                  │        exhausted  │                       │
                  │                   ▼                       │
                  │              ┌─────────┐                  │
                  └──────────────►  fetch_pdf  ◄──────────────┘
                                 └────┬────┘
                                      ▼
                                 ┌─────────┐   parse fails / thin text
                                 │  parse  ├──────────────┐
                                 └────┬────┘              ▼
                                      │            ┌──────────────┐
                                      │            │ degrade_mode │ (abstract-only)
                                      │            └──────┬───────┘
                                      ▼◄──────────────────┘
                                 ┌──────────────┐
                                 │ chunk_embed  │
                                 └──────┬───────┘
                                        ▼
                                 ┌──────────────┐
                                 │  summarize   │
                                 └──────┬───────┘
                                        ▼
                                 ┌──────────────┐  next question
                                 │   qa_node    │◄──────────┐
                                 └──────┬───────┘           │
                                        └───────────────────┘
                                        │ user exits
                                        ▼
                                       END
```

Implementation notes:

* Use `add_conditional_edges` for the three decision points: intent routing, candidate-count routing, parse-quality routing.
* Compile with `SqliteSaver`; `thread_id = arxiv_id` so **QA re-attaches to a previous session without re-parsing**.
* `qa_node` is entered via a separate `graph.invoke` call with the same `thread_id` — the checkpointer restores `paper`, `collection`, `messages`. Document this in the README; it is the answer to §5 of the brief.

---

## 5. Node contracts

### 5.1 `query_understanding`
* Regex first (deterministic, zero cost):
  * `r"(\d{4}\.\d{4,5})(v\d+)?"` → arXiv ID (also matches inside a URL)
  * old-style `r"([a-z\-]+(\.[A-Z]{2})?/\d{7})"`
* If no match → `intent="topic_search"`; ask the LLM once to normalize the phrase into an arXiv-friendly query (strip "recent work on", keep technical terms, return JSON `{"query": "...", "categories": ["cs.CL"]}`).
* If the LLM fails → fall back to the raw string. Never hard-fail here.

### 5.2 `search_arxiv`
* `arxiv.Search(query=..., max_results=15, sort_by=Relevance)`.
* Build the query as `all:"term1" AND all:"term2"` from extracted key terms; add `cat:` filter if the LLM gave categories.
* Retry with exponential backoff (3 attempts) on network error.

### 5.3 `broaden_query` (zero-result handler — **required failure case #1**)
* Attempt 1: drop quotes and `cat:` filter, `AND` → `OR`.
* Attempt 2: keep only the two highest-IDF terms, widen `sort_by=SubmittedDate`.
* After 2 attempts: terminate cleanly with an actionable message + the queries tried. **No exception traceback to the user.**

### 5.4 `select_paper` (many-result handler)
* Score = `0.6 * llm_relevance + 0.25 * recency_decay + 0.15 * has_full_text`.
* `llm_relevance`: one batched call — send the top 10 `(index, title, abstract[:600])` pairs, ask for JSON `[{"index":int,"score":0-10,"reason":str}]`.
* Interactive mode: Rich table of the top 5, user picks by number. `--auto` picks rank 1.
* Store `selection_reason` in state and print it — shows judgement, not magic.

### 5.5 `fetch_pdf`
* Cache at `.data/pdfs/{arxiv_id}.pdf`; skip download if present.
* 60s timeout, 2 retries, size guard (> 40 MB → warn and continue; > 100 MB → degrade).

### 5.6 `parse` (**required failure case #2**)
* PyMuPDF text extraction per page → join.
* **Quality gate:** `chars_per_page < 120` OR `alpha_ratio < 0.6` → assume scanned/broken → try pdfplumber → still bad → `parse_mode="degraded_abstract_only"`, seed `full_text` from the arXiv abstract, add a warning that the briefing is abstract-only. **Never crash, never silently pretend.**
* Section splitting: regex on lines matching `^\s*(\d+\.?\d*)?\s*(Abstract|Introduction|Related Work|Background|Method|Methodology|Approach|Experiments|Results|Evaluation|Discussion|Limitations|Conclusion|References|Appendix)\b` (case-insensitive) plus a font-size heuristic from `page.get_text("dict")`. Fall back to a single `body` section.
* Truncate `References` to 4 000 chars — keep them (asked for) but don't let them dominate the index.

### 5.7 `chunk_embed`
* **Section-aware recursive chunking:** never cross a section boundary. Target **400 tokens, 80 overlap** (bge-small has a 512-token window — anything larger is silently truncated at embed time; say this in the README).
* Metadata per chunk: `{arxiv_id, section, chunk_index, page_start, char_start}`.
* Upsert to Chroma with deterministic IDs `f"{arxiv_id}:{chunk_index}"` → re-running is idempotent.
* Skip embedding if the collection already has `n_chunks > 0` for this paper (cheap resume).

### 5.8 `summarize`
* **Map:** per section (cap ~6 most informative: abstract, intro, method, results, discussion, conclusion) → 3–5 factual bullets with the section name attached.
* **Reduce:** one call with all bullets → the `Briefing` Pydantic model as JSON.
* **Limitations guard (the brief explicitly calls this out):** if `limitations` is empty or contains filler ("none stated", "N/A"), run a second targeted pass over chunks retrieved for the query *"limitations, failure cases, threats to validity, future work"*. If still nothing, emit `["Not explicitly stated by the authors; reviewer-inferred: <one item>"]` and mark it as inferred. Never leave the field blank.
* **JSON repair loop:** validate with Pydantic → on `ValidationError`, re-prompt once with the error text → on second failure, fall back to a Markdown briefing and record the error.
* Write `examples/briefing_<id>.json` and print Markdown via Rich.

### 5.9 `qa_node` (grounding — 25% of the grade)
1. **Query rewrite** if `len(messages) > 0`: rewrite pronouns using the last 2 turns ("does it scale?" → "does the proposed KV-cache compression method scale?").
2. **Retrieve** top 20 → **MMR rerank** (λ=0.6) → keep 6.
3. **Abstain gate:** if `max_similarity < ABSTAIN_THRESHOLD` (default 0.35 cosine, configurable) → return the canned "That isn't covered in this paper" answer *without calling the LLM*.
4. **Answer prompt rules:** answer *only* from the numbered context blocks; cite as `[S1]`, `[S2]`; if the context is insufficient, say so explicitly; never use outside knowledge.
5. **Post-check:** every `[Sn]` cited must exist in the retrieved set; strip invalid ones and flag.
6. Return `QAAnswer{answer, citations:[{chunk_id, section, page, snippet}], grounded: bool}`.
7. Append the Q and A to `state["messages"]` and checkpoint.

**Citation numbering — required implementation detail:** `prompts.qa_prompt()` labels context blocks `[S1]`, `[S2]`, ... purely by their position in the list passed to it. That numbering has no inherent link back to a chunk's real `chunk_id` in the vector store. `qa_node` MUST build and keep an explicit `{"S1": chunk_id, "S2": chunk_id, ...}` mapping at retrieval time (same order used to build `qa_prompt`'s `context_blocks`), and use that mapping for step 5's post-check and step 6's `citations` list. Without this mapping, "every `[Sn]` cited must exist in the retrieved set" is unverifiable — there's nothing to check `[Sn]` against.

---

## 6. Exact interfaces (contract — do not deviate)

Every function below returns the named Pydantic model from `models.py` — never a raw
`dict` — unless marked `# dict OK`. If a return type needs coercion (e.g. ISO string
→ `datetime`, plain string → `HttpUrl`), do the coercion inside the model constructor
call, never by loosening the model itself.

### `services/arxiv_client.py`
```python
class ArxivClient:
    def __init__(self) -> None: ...
    def search(self, query: str, max_results: int | None = None,
               sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance
               ) -> list[PaperMeta]: ...
    def fetch_metadata(self, arxiv_id: str) -> PaperMeta | None: ...
    def build_search_query(self, terms: list[str],
                            categories: list[str] | None = None) -> str: ...
    @staticmethod
    def _extract_arxiv_id(entry_id: str) -> str: ...  # strip trailing vN via regex, not .replace("v","")

def get_arxiv_client() -> ArxivClient: ...
```

### `services/pdf_parser.py`
```python
def parse_pdf(pdf_path: str, arxiv_id: str, abstract_fallback: str
              ) -> tuple[list[dict], str, ParseMode]:
    """Returns (sections, full_text, parse_mode).
    sections: [{"title": str, "text": str, "page_start": int}]
    Never raises on a bad PDF — degrades to parse_mode="degraded_abstract_only"."""
```

### `services/chunker.py`
```python
def chunk_sections(sections: list[dict], arxiv_id: str,
                    target_tokens: int = 400, overlap_tokens: int = 80
                    ) -> list[Chunk]:
    """One Chunk per piece. Never crosses a section boundary.
    id field = f"{arxiv_id}:{chunk_index}" (deterministic, for idempotent upsert)."""
```

### `services/vectorstore.py`
```python
def get_collection(name: str): ...  # returns a chromadb Collection, dict OK internally
def upsert_chunks(collection_name: str, chunks: list[Chunk]) -> int: ...  # returns n upserted
def query_chunks(collection_name: str, query_text: str, n_results: int = 20
                  ) -> list[dict]:
    """Each dict: {"chunk_id","text","section","page_start","distance"}"""  # dict OK — retrieval hits, not a stored model
```

### `services/llm.py`
```python
def complete_json(schema: type[BaseModel], system_prompt: str, user_prompt: str,
                   provider: str | None = None) -> BaseModel:
    """Calls the LLM, validates against `schema`, retries once with the
    ValidationError text on failure. Raises only after the retry also fails."""
def complete_text(system_prompt: str, user_prompt: str,
                   provider: str | None = None) -> str: ...
```

### `nodes/*.py`
```python
def <node_name>(state: AgentState) -> dict:
    """Returns ONLY the keys this node changed, per §0 rule 3.
    Never mutates `state` in place and returns the whole thing."""
```

If the coding agent needs a helper not listed here, it must add its signature to a
new `### Stage <N> additions` subsection under this section before using it —
see §0.1 step 4.

---

## 6b. Pydantic `Briefing` model (exact fields — the brief's minimum)

```python
class KeyResult(BaseModel):
    claim: str
    evidence: str            # number / table / figure reference from the paper
    source_section: str

class Briefing(BaseModel):
    arxiv_id: str
    title: str
    authors: list[str]
    published: str           # ISO date
    categories: list[str]
    url: str
    pdf_url: str
    why_it_matters: str      # 1 paragraph, plain English
    problem_statement: str
    method: list[str]        # bullets
    key_results: list[KeyResult]
    limitations: list[str]   # MUST be non-empty
    followup_questions: list[str]  # 3-5
    meta: dict               # {model, parse_mode, n_chunks, warnings, generated_at}
```

---

## 6c. CLI surface

```bash
# topic search → interactive selection → briefing → QA REPL
python -m agent.cli digest "recent work on KV-cache compression for LLMs"

# direct paper, non-interactive, JSON only
python -m agent.cli digest 2401.12345 --auto --json-out examples/briefing.json --no-qa

# re-attach to an existing session (state comes from the checkpointer — no re-parsing)
python -m agent.cli ask 2401.12345 "What datasets did they evaluate on?"

python -m agent.cli sessions          # list saved threads
```

Global flags: `--provider {groq,gemini,ollama}`, `--model`, `--top-k`, `--verbose` (prints node transitions — nice for the demo GIF).

---

## 9. Staged implementation plan (one opencode session per stage)

| Stage | Files | Acceptance criteria |
|---|---|---|
| **S1 Skeleton** | `pyproject.toml`, `.env.example`, `config.py`, `models.py`, `state.py`, `Makefile`, `.gitignore` | `make install` works; `python -c "import agent"` clean; Pydantic models import |
| **S2 arXiv + parsing** | `services/arxiv_client.py`, `services/pdf_parser.py`, `nodes/retrieval.py`, `nodes/fetch_parse.py` | Script fetches 2401.12345, prints title + ≥10 sections + char count. Test: a broken PDF triggers degraded mode, no exception |
| **S3 Index + retrieve** | `services/chunker.py`, `services/vectorstore.py`, `nodes/indexing.py` | 100+ chunks in Chroma; a query returns chunks from the right section. Test: no chunk crosses a section boundary; overlap is respected |
| **S4 LLM layer** | `services/llm.py`, `services/prompts.py` | `llm.complete_json(schema, prompt)` returns validated objects on all three providers; repair loop covered by a test with deliberately malformed JSON |
| **S5 Graph wiring** | `graph.py`, `nodes/query_understanding.py`, `nodes/selection.py` | `build_graph().get_graph().draw_mermaid()` renders; ID input and topic input both route correctly; zero-result path terminates cleanly |
| **S6 Summarize** | `nodes/summarize.py` | Valid `Briefing` JSON for 3 different papers; `limitations` never empty |
| **S7 QA** | `nodes/qa.py` | 3 in-paper questions answered with citations; 1 out-of-paper question ("what does this say about the 2026 World Cup?") returns the abstain message |
| **S8 CLI + polish** | `cli.py`, tests, `docs/architecture.md` | `make demo` runs end to end; `pytest` green; README example pasted from a real run |

---

## 10. Definition of done

- [ ] `git clone && make install && make demo` works on a clean machine with **no API key** (Ollama path) and with a Groq key.
- [ ] `pytest` passes (≥5 tests, including one failure-path test).
- [ ] `examples/briefing_*.json` and `examples/sample_session.md` are committed from a real run.
- [ ] `docs/architecture.md` has a Mermaid graph generated from the actual compiled LangGraph.
- [ ] README has: architecture, state table, setup, example run, rate-limit note, **Design Decisions & Tradeoffs**, known limitations, "what I'd do next".
- [ ] No secrets committed; `.env` is gitignored.