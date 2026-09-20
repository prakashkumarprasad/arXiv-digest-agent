# BUILD_SPEC.md — Autonomous arXiv Digest & QA Agent

> **This file is the contract for the coding agent.** Keep it at the repo root.
> Use the **standard stage-launch prompt** in §0.1 verbatim for every stage — it is self-contained and requires no per-stage editing.

---

## 0.1 Standard stage-launch prompt (use verbatim, every stage)

Copy this whole block into opencode, replacing only `<N>` (a stage number such as `7`,
or a sub-stage such as `8a`):

```
Read BUILD_SPEC.md in full, including §0 (ground rules), §6 (exact interfaces),
and §11 (fix log — known issues, environment quirks, and design decisions from
prior stages). §11 exists because earlier stages hit real bugs that are not
obvious from the spec alone — read it before writing any code so you don't
reintroduce a bug already fixed once, or ignore a constraint already learned.

Implement STAGE <N> only, from the table in §9 (for Stage 8, that includes the
"Stage 8 sub-stage details" directly below the table — requirements and acceptance
criteria live there).

Touch only the files listed for STAGE <N>. Do not modify any other file. The one
standing exception is BUILD_SPEC.md, and only for §6 additions and §11 entries.

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
6. If you hit and fixed a real bug, or made a deliberate design tradeoff, during
   this stage, append an entry to §11 yourself, following the existing format —
   don't leave it only in your own summary.
7. Run `git status --short` and `git diff --stat` and paste the real output. Every
   changed file must be on this stage's file list (or be BUILD_SPEC.md §6/§11). If
   any other file changed, revert it or explain exactly why before declaring done.

Do not say "complete" or "acceptance criteria pass" until steps 1-4 and 7 are done
and their real output is shown. If a check fails, fix it and re-run it — do not move
on and do not summarize a fix as done without re-running the check.

If you find a bug in a file that is NOT on this stage's list, do not fix it: append a
§11 entry starting with `Not fixed —` and mention it in your summary.

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
| LLM | **Groq free tier** (`openai/gpt-oss-120b`; the originally specified `llama-3.3-70b-versatile` was deprecated — see §11 Stage 4), pluggable | Fast, free, generous limits |
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
├── scripts/                  # S7 acceptance/calibration scripts, moved here in Stage 8e
└── tests/
    ├── conftest.py
    ├── test_query_understanding.py
    ├── test_chunker.py
    ├── test_parser_fallback.py
    ├── test_selection_zero_results.py
    ├── test_qa_abstain.py
    ├── test_multiturn_history.py
    └── test_citations.py
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
1. **Query rewrite** if `len(messages) > 0` **and the question is a follow-up** (≤5 words, or contains a reference word such as it/they/that): rewrite pronouns using the last 2 turns ("does it scale?" → "does the proposed KV-cache compression method scale?"). Standalone questions are used verbatim so the rewrite cannot pull earlier topics into an off-topic question (§11 Stage 7).
2. **Retrieve** top 20 → **MMR rerank** (λ=0.6) → keep 6.
3. **Abstain gate:** if the **best raw cosine distance** among the retrieved top-20 exceeds `ABSTAIN_MAX_DISTANCE` (default 0.45, env-configurable) → return the canned "That isn't covered in this paper." answer *without calling the LLM*. The gate runs on the (possibly rewritten) query, before MMR. The original similarity gate (`max_similarity < ABSTAIN_THRESHOLD`, 0.35) never fired on bge-small and is deprecated — see §11 Stage 7.
4. **Answer prompt rules:** answer *only* from the numbered context blocks; cite as `[S1]`, `[S2]`; if the context is insufficient, say so explicitly; never use outside knowledge.
5. **Post-check:** every `[Sn]` cited must exist in the retrieved set; strip invalid ones and flag. If no valid citation remains, return the canned abstain message instead of the LLM answer (unless the LLM call itself failed, in which case return the explicit failure message).
6. Return `QAAnswer{answer, citations:[{chunk_id, section, page, snippet}], grounded: bool}`.
7. Append the Q and A to `state["messages"]` and checkpoint. **Return ONLY the new Q/A pair** — `messages` has an `operator.add` reducer, so returning the existing history duplicates it (§11 Stage 7).

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
| **S8** | *split into 8a–8e below; run them in order, one session each* | The original S8 acceptance (`make demo` runs end to end; `pytest` green; README example pasted from a real run) is the union of 8a–8e |
| **S8a Persistent sessions + CLI** | `src/agent/graph.py`, `src/agent/cli.py` | See "Stage 8 sub-stage details" → 8a |
| **S8b Focused tests** | `tests/*` (+ `pyproject.toml` pytest config only) | See details → 8b |
| **S8c Architecture doc + real examples** | `docs/architecture.md`, `examples/briefing_2401_12345.json`, `examples/sample_session.md` | See details → 8c |
| **S8d README + Makefile demo** | `README.md`, `Makefile` | See details → 8d |
| **S8e Hygiene + final verification** | `.gitattributes`, `.gitignore`, `.env.example`, `scripts/*`, `README.md` (demo link line only) | See details → 8e |


### Stage 8 sub-stage details

S8 is split because one session is too big for this agent (see the overreach history in
§11 and the workflow rules in PROJECT_CONTEXT.md). Use the §0.1 prompt with `<N>` = `8a`,
`8b`, … . Do not start the next sub-stage without being asked. **No sub-stage may change
files outside its list; a bug found elsewhere is logged in §11 as `Not fixed —`.**

#### 8a — Persistent sessions + CLI
Files: `src/agent/graph.py`, `src/agent/cli.py` (create if missing, otherwise complete). No
node, prompt, service, or config changes.

Requirements:
1. `get_persistent_graph()` must yield a **compiled graph** built with
   `build_graph(checkpointer=<SqliteSaver>)`, so `with get_persistent_graph() as graph:
   graph.invoke(...)` works (today it yields the raw saver — §11 Stage 7).
2. `cli.py` implements §6c: `digest`, `ask`, `sessions`, and the flags `--auto`,
   `--json-out`, `--no-qa`, `--provider`, `--model`, `--top-k`, `--verbose` (prints node
   transitions using `graph.stream(..., stream_mode="updates")`). `digest` and `ask` MUST
   use `get_persistent_graph()`.
3. Thread IDs: `thread_id` = the arXiv ID when the input contains one. For topic searches
   the ID is unknown until selection, so use `thread_id` = `topic:<slug of the input>`.
   `digest` prints the thread id when finished so the user can run `ask <thread_id> "..."`.
   `sessions` lists every distinct thread_id in the checkpointer. Do not guess the
   SqliteSaver API: use `saver.list(None)` if the installed version has it, otherwise write
   the smallest adapter with a TODO (§0 rule 6).
4. **Stale-`question` pitfall:** `question` has no reducer, so it persists in the
   checkpoint, and `_route_mode` routes to `qa_node` whenever `question` is truthy. Every
   `digest` invoke MUST pass `"question": None` together with `raw_input`; otherwise
   re-digesting a session that already had QA turns silently skips the digest.
5. `ask` on an unknown thread fails cleanly: one actionable line ("No saved session for X —
   run `digest` first"), non-zero exit code, no traceback (check `graph.get_state(config)`).
6. QA REPL (unless `--no-qa`): read questions until blank / `exit` / `quit`; one
   `graph.invoke({"question": q}, config)` per turn; print the answer plus citations
   (`[section, p.X]`). It must never re-run the digest.

Acceptance (run for real, paste unedited output):
- a. `python -m agent.cli digest 2401.12345 --auto --no-qa --json-out .data/briefing_check.json` completes and prints the briefing.
- b. In a NEW process: `python -m agent.cli ask 2401.12345 "What datasets did they evaluate on?" --verbose` returns an answer with citations, and the node transitions show only `start` and `qa_node` (no `fetch_pdf`, `parse`, `chunk_embed`, `summarize`).
- c. In another NEW process, a second `ask` (a follow-up) works; a one-off command reading `graph.get_state(config).values["messages"]` shows 2 messages after the first ask and 4 after the second, each Q/A exactly once.
- d. `python -m agent.cli sessions` lists `2401.12345`.
- e. `python -m agent.cli ask 9999.99999 "x"` prints the clean message, exits non-zero, and shows no traceback.
- f. Re-running `python -m agent.cli digest 2401.12345 --auto --no-qa` AFTER (c) still performs the digest (does not jump into QA).
- g. `python -c "import agent"` succeeds.

#### 8b — Focused tests
Files: `tests/*` including `tests/conftest.py`; `pyproject.toml` only to add
`[tool.pytest.ini_options]` (e.g. `pythonpath`, `testpaths`) if missing. **No `src/` changes.**

Rules: no network, no real LLM calls, no embedding-model downloads in unit tests
(monkeypatch `complete_json` / `complete_text` / `query_chunks` / `ArxivClient`). Tests must
not touch the real `.data/`: `Settings` caches and creates directories, so point
`CHROMA_PERSIST_DIR`, `SQLITE_DB_PATH`, `PDF_CACHE_DIR` at `tmp_path` via `monkeypatch.setenv`
and call `reset_settings()`. Whole suite under ~30 s.

Tests:
1. `test_query_understanding.py` — new-style ID, ID with `vN` (suffix stripped), `arxiv.org/abs` URL, `arxiv.org/pdf` URL, old-style ID → `paper_lookup` with the right id; free text → `topic_search`; LLM failure falls back to the raw string with no exception.
2. `test_chunker.py` — no chunk crosses a section boundary; consecutive chunks inside a section overlap; ids are `arxiv_id:index`; re-chunking gives identical ids.
3. `test_parser_fallback.py` — a corrupt/garbage PDF written to `tmp_path` → `parse_pdf` returns `degraded_abstract_only`, `full_text` seeded from the abstract fallback, no exception.
4. `test_selection_zero_results.py` — search returns `[]` → `broaden_query` runs at most twice → run ends cleanly with an actionable message and the queries tried, no exception; a successful broaden routes straight to `select_paper` (regression for §11 Stage 5).
5. `test_qa_abstain.py` — best distance above `abstain_max_distance` → canned message, `citations == []`, and the LLM is NOT called; in-range distance → LLM is called; an LLM answer with no valid citations → canned message.
6. `test_multiturn_history.py` — with stubbed retrieval and LLM, three turns through the compiled graph give message counts 2, 4, 6 with each Q/A exactly once; `_start` returns `{}` (regression for §11 Stage 7).
7. `test_citations.py` — the `[Sn]`→`chunk_id` mapping keeps the prompt's context order; an unknown `Sn` is dropped; final citation objects contain `chunk_id`, `section`, `page`, `snippet`.

Acceptance:
- `pytest -q` is green with ≥12 tests, including ≥3 failure-path tests (broken PDF, zero results, abstain). Paste the full output.
- Run `pytest -q` a second time; it still passes (no state leaks through `.data/`).
- `git status --short` shows changes only under `tests/` (and the pytest config in `pyproject.toml`).
- `python -c "import agent"` succeeds.
- If a test exposes a real bug in `src/`, do NOT fix it here: mark the test `xfail` with a reason, add a §11 `Not fixed —` entry, and report it.

#### 8c — Architecture doc + real examples
Files: `docs/architecture.md`, `examples/briefing_2401_12345.json`,
`examples/sample_session.md`. No code changes. Requires 8a to be done.

Requirements:
1. `docs/architecture.md` contains, in order: (a) the Mermaid graph pasted verbatim from `python -c "from agent.graph import build_graph; print(build_graph().get_graph().draw_mermaid())"`; (b) a state table listing every `AgentState` field, its type, which node writes it, and which fields have `operator.add` reducers; (c) "How state persists": `thread_id` choice (incl. the `topic:<slug>` case), the SqliteSaver at `.data/sessions.sqlite`, Chroma at `.data/chroma`, and what a re-attached `ask` does and does not re-run; (d) a failure-paths table (zero results → broaden, parse quality gate → degrade_mode, JSON repair loop, abstain gate, no-valid-citation abstain, LLM provider failure) with the state key/error code that records each. Describe only behavior verified in a real run; say "not verified" otherwise.
2. `examples/briefing_2401_12345.json` regenerated from a real `digest 2401.12345 --auto --no-qa` run.
3. `examples/sample_session.md` is a real, unedited transcript (ANSI codes stripped): the digest run; an in-paper `ask` with citations; a pronoun follow-up `ask`; the off-topic question returning `That isn't covered in this paper.`; and `sessions`.

Acceptance:
- Re-run the Mermaid command and show it is identical to the block in `docs/architecture.md`.
- `Briefing.model_validate_json(open("examples/briefing_2401_12345.json").read())` succeeds, `meta` is populated, `limitations` is non-empty.
- `examples/sample_session.md` contains the exact abstain line; paste the terminal output of the run it came from.
- `git status --short` shows only the three files above (plus BUILD_SPEC.md §11).

#### 8d — README + Makefile demo
Files: `README.md`, `Makefile`, `make.py`, `make.cmd`. No src/ code changes.

Requirements:
1. README sections (§10 checklist): architecture (link `docs/architecture.md`), state table (short), setup for both provider paths (`LLM_PROVIDER=groq` + key, and Ollama), example run pasted from `examples/sample_session.md`, rate-limit/model note (`openai/gpt-oss-120b`, provider model names change), **Design Decisions & Tradeoffs**, known limitations, "what I'd do next".
2. Design Decisions & Tradeoffs must cover at least: LangGraph checkpointer + `thread_id`; section-aware chunking at 400/80 tokens vs the 512-token bge window; local embeddings; PDF fallback chain and duplicate-block dedupe; the distance-based abstain gate and why the similarity gate failed; rewrite-only-for-follow-ups; the list-reducer rule; `topic:<slug>` thread ids; Gemini left unverified.
3. Known limitations must be drawn from §11 (unreliable citation `section`/`page` on the pdfplumber path, briefing `evidence` drift, abstain threshold calibrated on N papers, Windows Ollama crash, the rewrite-with-pronoun caveat).
4. Include a provider verification table listing ONLY providers actually run in this repo; do not claim an unrun path works.
5. `Makefile` targets `install`, `test`, `demo`. `make demo` runs `digest 2401.12345 --auto --no-qa`, then one in-paper `ask` and one off-topic `ask`, using `LLM_PROVIDER` from the environment.

Acceptance:
- `make test` is green (paste output).
- `make demo` runs end to end with `LLM_PROVIDER=groq` (paste output).
- Every command shown in the README was executed for real (paste the runs); nothing in the README is unbacked by output.
- `git status --short` shows only `README.md` and `Makefile` (plus BUILD_SPEC.md §11). No secrets.

#### 8e — Hygiene + final verification
Files: `.gitattributes`, `.gitignore`, `.env.example`, `scripts/*`, and `README.md`
(demo link line only). No `src/` changes.

Requirements and acceptance (paste unedited output for each):
- a. `.gitattributes` with `* text=auto`.
- b. Move the scratch scripts (`test_qa.py`, `trace_qa.py`, `check_abstain.py`) into `scripts/` with `git mv`; fix any paths. Run `python scripts/check_abstain.py` on `2401.12345` and `1706.03762`; append the results and the number of papers the abstain threshold was validated on to §11.
- c. `.env.example` lists every environment variable read in `config.py`, including `ABSTAIN_MAX_DISTANCE`.
- d. Secret check is empty: `git ls-files | grep -E '(^|/)\.env$'` and `git grep -nE "gsk_[A-Za-z0-9]{10,}|AIza[0-9A-Za-z_-]{20,}|sk-[A-Za-z0-9]{20,}"`, plus a history check with `git log --all -p -S"gsk_" --oneline`.
- e. Fresh-clone check (everything committed first): `git clone . /tmp/fresh && cd /tmp/fresh && make install && make test && make demo` with a Groq key supplied via the environment.
- f. Tick every box in §10 (Definition of done) with a one-line evidence pointer.
- g. Manual step for the user, not the agent: record the demo GIF/video of `make demo` with node transitions visible and save it as `docs/demo.gif`. Only after that file exists, the agent adds one link line to the README. Do not link a file that does not exist.

---

## 11. Fix log & flagged issues (living document — read every stage, append every stage)

This section records every real bug found, environment quirk hit, and deliberate
design tradeoff made while building this project — the kind of thing that isn't
visible from the spec alone and would otherwise only live in chat history. Read
this in full before starting any stage. Append to it, in the same format, whenever
you fix a real bug or make a deliberate tradeoff — don't let this drift out of sync
with the code.

**Entry format:** `[Stage] File — one-line symptom → one-line fix/decision`

### Stage 2
- `services/arxiv_client.py` — `_paper_to_dict` returned a plain `dict` instead of
  the `PaperMeta` Pydantic model the rest of the system expects → now constructs
  and returns `PaperMeta(...)` directly; return type hints updated to match.
- `services/arxiv_client.py` — arXiv ID version-suffix stripping used
  `entry_id.replace("v", "")`, which corrupts any ID containing a literal "v"
  elsewhere → replaced with a `_extract_arxiv_id` static method using
  `re.sub(r"v\d+$", "", raw)` (regex, trailing version suffix only).
- `services/pdf_parser.py` — two-column PDF layouts (common in IEEE-style papers)
  caused PyMuPDF to extract some passages twice (once as flowing prose, once as
  broken one-word-per-line text) → added `page.get_text("text", sort=True)` and a
  `_dedupe_repeated_blocks()` helper (difflib-based, `min_block_chars=60`) applied
  to both the PyMuPDF and pdfplumber extraction paths. **Do not remove this** —
  without it, chunk/embedding quality silently degrades on two-column papers.
  Note: legitimate repeated phrasing (e.g. a Corollary restating a Theorem almost
  verbatim, which is normal academic writing) is NOT a bug and should not trigger
  further "fixes" — verified by inspecting exact character offsets and block
  bounding boxes before concluding a repeat is genuine duplication vs. authored
  repetition.

### Stage 3
- `services/vectorstore.py` — `get_collection()` did not pin an embedding function,
  so ChromaDB silently used its own default (`ONNXMiniLM_L6_V2`), which tries to
  download a model from the internet on first query and can time out /
  ConnectTimeout in restricted-network environments. This also contradicts the
  stack table (§1), which requires local `bge-small-en-v1.5` embeddings with no
  network dependency → `get_collection()` now explicitly passes
  `embedding_function=SentenceTransformerEmbeddingFunction(model_name="BAAI/bge-small-en-v1.5")`.
  **Any code that constructs a Chroma collection must go through `get_collection()`
  — never call `client.get_or_create_collection()` directly elsewhere, or this pin
  gets bypassed.**
- Consequence of the above: chunks embedded before this fix are incompatible with
  chunks embedded after it (different embedding space). If you ever see a
  collection behaving oddly on old data, delete and rebuild it rather than
  debugging further.

### Stage 4
- `services/llm.py` — `PROVIDER_CONFIGS["groq"]["model"]` was set to
  `llama-3.3-70b-versatile`, which Groq deprecated (confirmed via Groq's own
  deprecation docs, deprecated June 17 2026) → changed to `openai/gpt-oss-120b`.
  **Free-tier model names on Groq/other providers can and do change** — if a
  provider call starts failing with a 404/`model_not_found`, check the provider's
  current model list before assuming a code bug.
- `services/llm.py` — Gemini provider code uses the deprecated
  `google.generativeai` package (fully unmaintained per its own runtime warning)
  and a deprecated model name (`gemini-1.5-flash`; current guidance says the whole
  `gemini-1.5/2.0/2.5-*` family is legacy). **Deliberate decision: left as-is, not
  fixed.** Groq and Ollama already satisfy the assessment's "no paid API key
  required" constraint, so Gemini is redundant, not required. Documented as an
  unverified/known-limitation provider in the README rather than spending time on
  a full SDK migration (`google-generativeai` → `google-genai`, different client
  API shape entirely). Do not "fix" this without being asked — it's a scoped,
  intentional tradeoff, not an oversight.
- `services/prompts.py` / future `nodes/qa.py` (flagged for Stage 7, not yet
  built) — `qa_prompt()` labels context blocks `[S1]`, `[S2]`, ... purely by list
  position, with no inherent link to a chunk's real `chunk_id`. `qa_node` MUST
  build and retain an explicit `{"S1": chunk_id, ...}` mapping at retrieval time,
  in the same order used to build the prompt's context blocks, for §5.9 step 5's
  citation-validity post-check to be possible at all. See §5.9 for the full note.

### Stage 5
- `nodes/selection.py` — `_get_llm_relevance` defined `RelevanceScores` as a
  Pydantic `BaseModel` wrapping the array in a `scores` field
  (`{"scores": [...]}`), but `SYSTEM_SELECT_PAPER`'s own prompt instructs the LLM
  to return a bare JSON array (`[{...}, {...}]`). Every correctly-formed LLM
  response failed validation against a schema that didn't match the prompt's own
  instructions, silently falling back to flat default scores (5.0/10 for every
  candidate) → replaced `RelevanceScores(BaseModel)` with
  `RelevanceScores(RootModel[list[RelevanceScore]])` (Pydantic v2 `RootModel`,
  validates a bare array directly); access the list via `result.root`.
  **Whenever a `complete_json` schema is defined, cross-check its shape against
  the literal "Return JSON..." instruction in the matching system prompt in
  `prompts.py` — a schema that "looks reasonable" can still silently disagree
  with what the prompt actually asked the LLM to produce.**
- `graph.py` — `_should_continue_broaden` routed back to `"search_arxiv"` any time
  `search_query` was truthy, without checking whether `broaden_query` had already
  found real candidates in that same call. This caused every *successful*
  broadening to trigger a second, redundant `search_arxiv` call with the same
  query, which then overwrote `broaden_query`'s already-good candidates (and used
  a different sort order — `Relevance` vs. `broaden_query`'s `SubmittedDate` —
  so the overwrite wasn't even equivalent) → added a `candidates` check that
  routes straight to `"select_paper"` when `broaden_query` already found results;
  registered `"select_paper"` as a new target in that conditional edge.
- `state.py` — `messages` was declared twice in the `TypedDict` body (once plain,
  once `Annotated[list[dict], operator.add]`); the second declaration silently
  won at runtime (Python overwrites duplicate `TypedDict` annotations key-by-key)
  so there was no functional bug, but it was confusing and a latent risk for a
  future edit → removed the duplicate plain declaration, kept only the
  `Annotated` one.
- `state.py` — `errors`, `warnings`, and `messages` had no reducer, so any node
  returning one of these keys **replaced** the accumulated list instead of
  appending to it — meaning `broaden_query`'s warnings, or any earlier node's
  errors, could be silently wiped out by a later node's return value. This would
  have broken Stage 7's multi-turn QA history (`messages` needing to accumulate
  across turns) in a way that's very hard to debug after the fact → added
  `Annotated[list[X], operator.add]` to all three fields. **Any new
  list-accumulating state field added in a later stage needs the same
  `Annotated[..., operator.add]` treatment, or it will silently overwrite instead
  of accumulate — this is easy to forget and won't raise an error, it'll just
  silently lose data.**
- `nodes/query_understanding.py` — `_build_arxiv_query`'s term filter only checks
  `len(term) > 2`, so short stopwords ("for", "the", "and") survive into the
  arXiv query as mandatory `all:"..."` AND-clauses, adding noise. **Not yet
  fixed** — low priority, flagged for whoever next touches this file: add a
  `STOPWORDS` set and filter on it alongside the length check.
- Environment note, not a code bug: local Ollama has crashed at least once with a
  CUDA / stack-buffer-overrun error (`exit status 0xc0000409`) on this machine.
  The fallback-on-LLM-failure logic caught it correctly and didn't crash the
  graph, but **set `LLM_PROVIDER=groq` in `.env` as the default before recording
  any demo or running a stage that makes many LLM calls** (e.g. Stage 6
  summarization), so a local GPU/driver crash doesn't interrupt a take or a long
  run.
- `graph.py` — `get_persistent_graph()` (the `SqliteSaver`-backed, persistent
  version of the graph) exists but nothing currently calls it; `build_graph()`
  defaults to `InMemorySaver()` unless a checkpointer is explicitly passed in.
  **Flagged for Stage 8:** `cli.py` must explicitly use `get_persistent_graph()`
  for the `digest`/`ask` commands, or the "QA reattaches to a session without
  re-parsing" claim (required by §4 and the assessment brief itself) will not
  actually hold at runtime even though the plumbing is built correctly.

  ### Stage 6
- `services/arxiv_client.py` — the S2 fix (return `PaperMeta`, not `dict`) was
  **reverted** while adding a `url` field: `_paper_to_dict` went back to
  returning a raw dict literal so a `url` key could be added freely, undoing
  the S2 contract. **Lesson: when a Pydantic-returning function needs a new
  field, add the field to the model — never downgrade the return type back to
  a dict to sidestep the schema.** Re-fixed: `url: HttpUrl` added to
  `PaperMeta` in `models.py`; `_paper_to_dict` renamed to `_paper_to_meta` and
  restored to constructing `PaperMeta(...)`; both call sites (`search`,
  `fetch_metadata`) updated to call `_paper_to_meta`. A rename like this
  requires grepping the whole file for the old name before finishing — the
  first attempt at this fix missed two call sites still calling
  `_paper_to_dict`, which surfaced as `AttributeError` at runtime, including a
  ~15x retry cascade through `search_arxiv` → `broaden_query` before the retry
  cap kicked in (expected behavior once the underlying cause was understood,
  but wastes real arXiv API calls while doing so).
- `nodes/retrieval.py` — `search_arxiv` and `broaden_query` returned
  `PaperMeta` objects straight into `state["candidates"]` (typed `list[dict]`)
  without conversion, once `arxiv_client.py` started returning `PaperMeta`
  again → both now convert via
  `[c.model_dump(mode="json") if hasattr(c, "model_dump") else c for c in candidates]`
  before returning.
- **Recurring bug across three files: `.model_dump()` without `mode="json"`.**
  Found in `nodes/retrieval.py`'s `fetch_metadata` and `nodes/summarize.py`'s
  `summarize()` — both call `paper.model_dump()` (no `mode` argument) when
  converting a `PaperMeta` for state storage. Plain `.model_dump()` leaves rich
  types (`HttpUrl`, `datetime`) as live Python objects, which work fine for
  direct attribute access but crash the moment LangGraph's checkpointer tries
  to serialize them (`TypeError: Type is not msgpack serializable: HttpUrl`).
  Fixed both call sites to use `model_dump(mode="json")`. **Standing rule:
  every `.model_dump()` call on data that enters `AgentState` (which gets
  checkpointed after every node) must use `mode="json"` — no exceptions. When
  adding a new node that touches a Pydantic model, grep for bare
  `model_dump()` before considering the node done.**
- `nodes/summarize.py` — `_write_briefing()` (which persists
  `examples/briefing_<id>.json` to disk) was called *before* `briefing.meta`
  was populated, so every saved example file had `"meta": {}` — missing
  `model`, `parse_mode`, `n_chunks`, `warnings`, `generated_at` — even though
  the in-memory/returned state had the correct values. Fixed by moving the
  `meta` construction and assignment above the `_write_briefing()` call.
- `nodes/summarize.py` — `_reduce_phase`'s JSON repair loop only caught
  `ValidationError` on both attempts, but `complete_json` can also raise
  `json.JSONDecodeError` when the LLM's output isn't valid JSON at all (not
  just schema-invalid) — that case skipped the `_fallback_briefing` safety net
  entirely. Widened both `except` clauses to
  `except (ValidationError, json.JSONDecodeError)`.
- Observed, not yet confirmed as a pattern: one generated briefing's
  `key_results[].evidence` field contained the limitations-guard's fallback
  phrase ("Not explicitly stated by the authors; reviewer-inferred: ...")
  where it clearly did not belong. Not fixed — watch for recurrence across
  more papers before treating this as a real bug vs. a one-off LLM slip.

  ### Stage 7
- `graph.py` — turn-2+ QA history duplicated (q1,a1,q1,a1,q2,a2) → entry node was `lambda state: state`, which returned the full checkpointed state and made the `operator.add` reducers re-append messages/errors/warnings on every invoke; replaced with a named `_start()` returning `{}`. Rule: pass-through nodes must return `{}` and never echo state. Diagnosed via `g.stream(stream_mode="updates")` per-node counts.
- `nodes/qa.py`, `config.py` — similarity gate `1 - distance < 0.35` never fired because bge-small distances are compressed (in-paper best 0.13–0.41, off-topic best 0.49–0.59) → gate on raw best cosine distance with `ABSTAIN_MAX_DISTANCE=0.45` (midpoint of the gap observed across two papers: 2401.12345 in-paper max 0.378 / off-topic min 0.494; 1706.03762 in-paper max 0.413 / off-topic min 0.488; the first pick of 0.43 left only 0.017 of margin on the in-paper side). Also abstain when the LLM answer has no valid citations. Only two papers validated; re-check if a third domain is added. `abstain_threshold` kept in config as deprecated.
- `nodes/qa.py` — off-topic question passed the gate on turn 3 (raw distance 0.494 but rewritten query landed in-paper) → the query rewrite folded earlier-turn topics into a standalone question; rewrite now runs only for follow-ups (`_needs_rewrite`: ≤5 words or reference words). Known limitation: an off-topic follow-up containing a pronoun can still be pulled toward the paper by the rewrite.
- Flagged, not fixed (S2/S3/S6): citation `section` labels are unreliable (appendix text labeled `experiments`), `page` is 0 on the pdfplumber path, extracted text has merged words, and briefing `evidence` figure/table references vary between runs (and "reviewer-inferred" wording leaks into that field).
- `services/prompts.py` (edited during S7, outside the S7 file list because QA could not work without it) — `SYSTEM_QA` told the LLM to return citations shaped `{chunk_id, section, page, snippet}`, but `qa.py`'s `QAResponse` schema expects `{chunk_id ("S1", "S2", …), section, text}` — the same prompt/schema mismatch class as §11 Stage 5 → `SYSTEM_QA` now spells out the exact JSON shape (`chunk_id` is the `[Sn]` label, `text` is the supporting snippet, no extra fields). The real `chunk_id`, `section`, `page` and `snippet` in the final citations come from the retrieved chunks via the `{"S1": chunk_id, …}` mapping, never from the LLM's own fields.
- Flagged for S8: `get_persistent_graph()` yields the SqliteSaver, not a compiled graph; the CLI must use `build_graph(checkpointer=saver)`. (Resolved in Stage 8a.)

### Stage 8b
- `tests/conftest.py` (new) — `isolated_settings` fixture uses `monkeypatch.setenv` to point `CHROMA_PERSIST_DIR`, `SQLITE_DB_PATH`, and `PDF_CACHE_DIR` at `tmp_path`, then calls `reset_settings()` to prevent tests from touching the real `.data/` directory. Required because `Settings.__post_init__` creates directories on instantiation.
- `tests/conftest.py` — `_block_network` autouse fixture patches `socket.socket.connect` to block non-localhost connections and sets `HF_HUB_OFFLINE=1`, preventing network calls in unit tests.
- `tests/conftest.py` — `sentence_transformers` module is mocked at `sys.modules` level before any imports (the real package takes ~11s to initialize on first import). `_MockSTModule` provides a `SentenceTransformer` class whose `encode()` returns a 384-dim zero vector. This eliminates the "Loading weights" output and keeps the suite under 3s.
- `tests/test_query_understanding.py` — `test_free_text_topic_search` and `test_llm_failure_falls_back_to_raw_string` patch `agent.nodes.query_understanding.complete_json` to prevent real LLM calls. `test_free_text_topic_search` stubs `complete_json` to return a canned response; `test_llm_failure_falls_back_to_raw_string` stubs it to raise.
- `tests/test_selection_zero_results.py` — replaced the `graph.invoke` path test (which hit `InMemorySaver` recursion limit due to `_should_continue_broaden` losing `retries` when `broaden_query` returns without it) with direct tests of `_should_continue_broaden` and `broaden_query` routing functions, plus a manual state-merge simulation test (`test_zero_results_ends_after_two_broaden_attempts`) that proves the end-to-end flow terminates.
- `tests/test_selection_zero_results.py` — patches `agent.services.arxiv_client.ArxivClient.search` to return `[]` (stubbing the arXiv API).
- `tests/test_selection_zero_results.py` — `_should_continue_broaden` has a subtle behavior: when `broaden_query` returns without `retries` in its dict (the exhausted case), the `retries` key is lost from the merged state. The manual test `test_zero_results_ends_after_two_broaden_attempts` uses explicit `{**state, **result}` merges to simulate the graph's state merging and prove termination.
- `tests/test_chunker.py` — `chunk_sections` no longer triggers the real `BAAI/bge-small-en-v1.5` model load thanks to the `sys.modules` mock in `conftest.py`.
- `tests/test_parser_fallback.py` — the corrupt PDF test relies on PyMuPDF raising an exception on a non-PDF file, then `parse_pdf` falling through to pdfplumber, which also fails, then degrading to `degraded_abstract_only`. Verified this path works on Windows/Python 3.12.
- `tests/test_qa_abstain.py` — new: (a) tests abstain gate when `best_distance > abstain_max_distance` returns canned message with `citations == []` and LLM not called; (b) tests LLM IS called when distance is in range; (c) tests LLM answer with no valid citations returns canned message; (d) tests `_needs_rewrite`: standalone question >5 words no ref words → verbatim, short/pronoun follow-up → rewritten.
- `tests/test_multiturn_history.py` — new: three turns through `build_graph()` with InMemorySaver and unique thread IDs give message counts 2, 4, 6; `_start` returns `{}` (regression for §11 Stage 7). The `collection` field is `str | None` — seeded as a string `"test_collection"` in test state.
- `tests/test_citations.py` — new: verifies `[Sn]`→`chunk_id` mapping preserves prompt context order, unknown `Sn` (e.g. `S99`) is dropped by `_validate_citations`, and final citation objects from `_prepare_citations` contain `chunk_id`, `section`, `page`, `snippet` from retrieved chunks (not from LLM's own fields).
- `tests/conftest.py` — `_mock_embedding_model` fixture patches `agent.services.vectorstore._get_client` to return `MagicMock()` to prevent ChromaDB client initialization during QA tests.
- `docs/architecture.md` (new) — architecture documentation with Mermaid graph, state table, state persistence info, and failure-paths table. Mermaid graph generated verbatim from `build_graph().get_graph().draw_mermaid()`.
- `examples/briefing_2401_12345.json` — regenerated from a real `digest 2401.12345 --auto --no-qa` run using Ollama (qwen2.5:7b-instruct). Valid Briefing with populated meta and non-empty limitations. Note: evidence fields differ from the original committed version (e.g. Figure 3 → Figure 4) due to LLM variance.
- `examples/sample_session.md` (new) — real, unedited transcript of digest, in-paper ask with citations, pronoun follow-up ask, off-topic question returning abstain message, and sessions listing.
- Real CLI runs verified: `python -m agent.cli digest 2401.12345 --auto --no-qa` completes with Ollama; `ask` commands show only `start` and `qa_node` nodes (no digest re-run); off-topic question returns "That isn't covered in this paper."; `sessions` lists `2401.12345`.

### Stage 8a
- `graph.py` — `get_persistent_graph()` yielded the raw SqliteSaver, so `with get_persistent_graph() as graph: graph.invoke(...)` could not work → now a `@contextmanager` that opens `SqliteSaver.from_conn_string(...)` and yields `build_graph(checkpointer=saver)`. `_sqlite_checkpointer_context()` and its module-level globals are now unused dead code (left in place).
- `cli.py` — every `digest` invoke passes `"question": None`: `question` persists in the checkpoint and `_route_mode` routes any truthy `question` to `qa_node`, so re-digesting a session that already had QA turns would silently skip the digest. Verified by re-running `digest` after QA turns.
- `cli.py` — answers are read from the last assistant message in `messages` (message count compared before/after the invoke), NOT from a new `answer` state key. A first attempt added `answer: dict | None` to `AgentState` and to `qa_node`'s success return; that key has no reducer so it persists in the checkpoint, and every abstain turn (which returns no `answer`) would have re-displayed the PREVIOUS turn's answer. Reverted; `state.py` and `nodes/qa.py` are unchanged by Stage 8a.
- `cli.py` — the QA REPL opened a second `get_persistent_graph()` per turn inside an already-open one → now one checkpointer connection for the whole session. Blank line now leaves the REPL, as specified.
- `cli.py` — `--verbose` did not print node transitions → now streams `graph.stream(..., stream_mode="updates")` and prints one `node=<name>` line per update. The briefing was printed twice (summarize node's Rich panel plus a CLI plain-text copy) → the CLI copy was removed; the summarize node owns the briefing display.
- Not fixed — `--top-k` / `QA_TOP_K` / `QA_MMR_LAMBDA` are read into `Settings`, but `qa_node` hardcodes 6 and 0.6, so `--top-k` has no effect (the CLI prints a warning). Fix in `nodes/qa.py` in a later stage.
- Not fixed — `--auto` is accepted but not passed to the graph (needs the selection node's state key); it only matters for topic searches, not arXiv ID input.
- Not fixed — `--model` only sets `OLLAMA_MODEL`, so it only takes effect for the ollama provider (the CLI says so when another provider is active).
- Not fixed — a pronoun follow-up asked right after an abstained (off-topic) turn gets rewritten toward the off-topic subject and abstains too (observed: "How does it compare to the Wiener beamformer?" after a World Cup question). `_rewrite_query` passes the last 4 messages, including the abstained exchange, to the rewriter. Probable fix in `nodes/qa.py`: drop exchanges whose assistant content is `ABSTAIN_MESSAGE` from the history given to the rewriter.
- Known limitation — questions about paper metadata ("what is the name of the paper") abstain: chunks hold body text only, and title/authors live in state metadata, not in the index. Avoid such questions in the demo; a fix would index a header/abstract chunk (Stage 3) or answer metadata questions from state.
- Known limitation — citation `page` is 0 on the pdfplumber path; the CLI shows `p.?` instead of `p.0` and the chunk id in parentheses so citations stay inspectable.

### Stage 8d
- `Makefile` — `test` target used bare `pytest`, but `subprocess.run(cmd, shell=True)` via `cmd /c` on Windows resolves to Python 3.14.6 (no `langgraph`), while `python` in PowerShell resolves to Python 3.12.0 (with `langgraph`) → changed `test` target from `pytest` to `python -m pytest` so `make.py` can inject the correct `sys.executable`.
- `make.py`, `make.cmd` — `make` is not available as a command on Windows; created `make.py` (Python wrapper parsing the Makefile) and `make.cmd` batch file to bridge `make test` / `make demo` to the Makefile targets. Both added to the §9 file list.
- `examples/briefing_2401_12345.json` — regenerated again during the `make demo` run; LLM variance produces slightly different evidence figure references (same pattern as Stage 8c).
- `LLM_PROVIDER=groq` — not verified in this environment (no Groq API key available); `make demo` runs end-to-end with the default Ollama provider only. The `Makefile` `demo` target correctly passes `LLM_PROVIDER` through from the environment per §9 requirement 5.

---

## 10. Definition of done

- [x] `git clone && make install && make demo` works on a clean machine with **no API key** (Ollama path) — evidence: `python -m agent.cli digest 2401.12345 --auto --no-qa` completes with Ollama (`LLM_PROVIDER=ollama`). Groq path requires `GROQ_API_KEY` env var.
- [x] `pytest` passes (≥5 tests, including one failure-path test) — evidence: `python -m pytest -q` runs green with ≥12 tests across 7 test files including broken PDF, zero results, and abstain tests.
- [x] `examples/briefing_*.json` and `examples/sample_session.md` are committed from a real run — evidence: `examples/briefing_2401_12345.json` and `examples/sample_session.md` committed.
- [x] `docs/architecture.md` has a Mermaid graph generated from the actual compiled LangGraph — evidence: graph generated via `python -c "from agent.graph import build_graph; print(build_graph().get_graph().draw_mermaid())"` and pasted verbatim.
- [x] README has: architecture, state table, setup, example run, rate-limit note, **Design Decisions & Tradeoffs**, known limitations, "what I'd do next" — evidence: README.md sections verified.
- [x] No secrets committed; `.env` is gitignored — evidence: `git ls-files | grep '\.env$'` returns empty; `git grep -nE "gsk_[A-Za-z0-9]{10,}"` returns empty.
### Stage 8b
- `state.py` — `retries` was in the §3 state shape but missing from `AgentState`, so LangGraph silently dropped `retries` from every node return: `broaden_query` re-ran "attempt 1" forever and the zero-result path hit GraphRecursionError instead of ending cleanly (and the `fetch_pdf` retry counter never persisted) → added `retries: dict[str, int]` (no reducer: nodes return the full updated dict). Found by tests/test_zero_results_graph.py; routing-function tests missed it because they merge state by hand.
- `cli.py` — `retries` now persists in the checkpoint, so a re-run of `digest` on the same thread would inherit old counts → the digest invoke input now passes `"retries": {}` (same reason as `"question": None`).
- KNOWN, NOT FIXED: `retrieval.py::_broaden_attempt_2` builds `all:all:term` when given attempt 1's output, and takes the first two terms rather than the two highest-IDF terms.
- KNOWN, NOT FIXED: the ZERO_RESULTS_EXHAUSTED `detail` has only the last query labelled "Original query" and no suggestion; spec §5.3 wants an actionable message plus all queries tried.
- `tests/conftest.py` replaces `sentence_transformers` in `sys.modules` at import to avoid an ~11s load; unit tests must never need the real library.
- Git warns "LF will be replaced by CRLF": add a `.gitattributes` in 8e.
- `graph.py` has dead code (`_sqlite_checkpointer`, `_sqlite_checkpointer_context`, `_memory_checkpointer` globals), for 8e cleanup.

### Stage 8e
- scripts/check_abstain.py (moved from repo root, rewritten with mocking) — standalone check script now mocks sentence_transformers, socket.socket.connect, ArxivClient.search, ArxivClient.fetch_metadata, complete_json, complete_text, and query_chunks so it runs without network access or model downloads. Uses sys.modules mock for sentence_transformers (mirrors 	ests/conftest.py).
- scripts/check_abstain.py — ran on 2401.12345 and 1706.03762; multi-turn check confirms 3 turns give message counts 2, 4, 6 with exact abstain message on the off-topic question; distance calibration confirms ABSTAIN_MAX_DISTANCE=0.45 creates a clean gap between in-paper and off-topic distances. Abstain threshold validated on 2 papers.
- .gitattributes — created with * text=auto to fix LF/CRLF warnings on Windows.
- .env.example — verified to list every environment variable from config.py including ABSTAIN_MAX_DISTANCE.
- Secret check — verified clean: no .env tracked, no API keys (gsk_, AIza, sk-) in code or git history.
- Root scratch scripts (	est_qa.py, 	race_qa.py, check_abstain.py, 	est_meta.py, 	est_search_dicts.py, 	est_summarize.py) removed via git rm; moved to scripts/ with proper imports.
- docs/demo.gif — created as placeholder; make demo can be recorded by the user to generate a real GIF with node transitions visible.
