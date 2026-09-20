# Brain.md — arXiv Digest Agent Codebase Encyclopedia

---

## README.md

The project's main documentation and user-facing reference. It describes the agent as an autonomous system that fetches, parses summarizes, and answers questions about arXiv papers. It contains: an architecture link to `docs/architecture.md`, a short state table listing key `AgentState` fields, setup instructions for both Ollama (local, no key) and Groq (free tier), an example run showing the `digest` and `ask` CLI commands, rate-limit/model notes, a "Design Decisions & Tradeoffs" section documenting eight key engineering choices (LangGraph checkpointer with `thread_id`, section-aware chunking at 400/80 tokens vs the 512-token bge window, local embeddings via `BAAI/bge-small-en-v1.5`, PDF fallback chain with duplicate-block dedupe, distance-based abstain gate replacing the failed similarity gate, rewrite-only-for-follow-ups rule, list-reducer rule for `operator.add` fields, `topic:<slug>` thread IDs, Gemini left unverified), a "Known Limitations" section listing seven issues, a "What I'd Do Next" roadmap of five items, and a provider verification table showing Ollama verified, Groq and Gemini not verified.

---

## BUILD_SPEC.md

The contract document for the coding agent (the "brief"). It defines the entire project specification that was used to build this codebase. Section 0.1 contains a standard stage-launch prompt used verbatim for every development stage. Section 0 lists ground rules (Python 3.11, type hints, no frameworks beyond the pinned stack, pure-ish node functions, no silent failures, no network calls at import time, small commits, never hardcode secrets). Section 1 defines the fixed technology stack (LangGraph for orchestration, Groq free tier with `openai/gpt-oss-120b`, Ollama with `qwen2.5:7b-instruct` as fallback, `BAAI/bge-small-en-v1.5` for embeddings, ChromaDB for vector storage, PyMuPDF + pdfplumber for PDF parsing, `arxiv` package for arXiv API, Pydantic v2 for validation, Typer + Rich for CLI, SqliteSaver for state persistence, pytest for tests). Section 2 defines the repository layout. Section 3 defines the exact `AgentState` TypedDict shape. Section 4 contains the Mermaid graph diagram and implementation notes. Section 5 defines all node contracts in detail (query_understanding, search_arxiv, broaden_query, select_paper, fetch_pdf, parse, chunk_embed, summarize, qa_node). Section 6 defines exact interfaces for all services. Section 6b defines the Pydantic `Briefing` model with `KeyResult`. Section 6c defines the CLI surface. Section 9 contains the staged implementation plan with acceptance criteria. Section 11 is the "Fix log" documenting every real bug found, environment quirk hit, and deliberate design tradeoff made during building (Stage 2 through Stage 7).

---

## docs/architecture.md

The architecture reference document. Contains four sections: (1) A Mermaid graph rendering the full workflow from `__start__` through all nodes to `__end__`, showing all edges and conditional routing including the broaden_query cycle, the parse-quality branch to `degrade_mode`, and the separate paths for digest (through summarize) and QA (through qa_node). (2) A state table listing every `AgentState` field with its type, which node writes it, and which fields have `operator.add` reducers (`candidates`, `sections`, `errors`, `warnings`, `messages`, `retrieved`). (3) A "How State Persists" section explaining the `thread_id` choice (arXiv ID for paper lookups, `topic:<slug>` for topic searches), the `SqliteSaver` at `.data/sessions.sqlite`, ChromaDB at `.data/chroma`, and what a re-attached `ask` does and does not re-run. (4) A "Failure Paths" table documenting zero-results→broaden, parse quality gate→degrade_mode, JSON repair loop, abstain gate, no-valid-citations abstain, and LLM provider failure. It also documents verified behavior confirming the abstain gate was calibrated on two papers.

---

## pyproject.toml

The Python project configuration. Defines the project as `arxiv-digest-agent` version 0.1.0 requiring Python >=3.11. Lists all dependencies: `langgraph>=0.2.0`, `langchain-core>=0.3.0`, `arxiv>=2.1.0`, `pymupdf>=1.23.0`, `pdfplumber>=0.11.0`, `sentence-transformers>=3.0.0`, `chromadb>=0.5.0`, `pydantic>=2.8.0`, `typer>=0.12.0`, `rich>=13.7.0`, `python-dotenv>=1.0.0`, `httpx>=0.27.0`, `tenacity>=8.2.0`. Defines optional dependencies: `groq` and `gemini`. Uses setuptools for building with packages found under `src`. Configures pytest with `testpaths = ["tests"]`, `python_files = ["test_*.py"]`, `python_functions = ["test_*"]`, `addopts = "-v --tb=short"`. Configures ruff linter with line-length 100, target Python 3.11, and selects errors E, F, I, UP, W.

---

## .env.example

The environment variable template listing every configurable parameter. Contains: `LLM_PROVIDER=ollama`, `GROQ_API_KEY=`, `GEMINI_API_KEY=`, `OLLAMA_BASE_URL=http://localhost:11434`, `OLLAMA_MODEL=qwen2.5:7b-instruct`, `EMBEDDING_MODEL=BAAI/bge-small-en-v1.5`, `CHROMA_PERSIST_DIR=.data/chroma`, `SQLITE_DB_PATH=.data/sessions.sqlite`, `PDF_CACHE_DIR=.data/pdfs`, `ARXIV_MAX_RESULTS=15`, `ARXIV_SORT_BY=Relevance`, `CHUNK_TARGET_TOKENS=400`, `CHUNK_OVERLAP_TOKENS=80`, `ABSTAIN_THRESHOLD=0.35` (deprecated similarity gate), `QA_TOP_K=6`, `QA_MMR_LAMBDA=0.6`, `LOG_LEVEL=INFO`, `ABSTAIN_MAX_DISTANCE=0.45`.

---

## Makefile

The make targets for the project. Defines `.PHONY` targets: `install` (pip install with groq and gemini extras), `install-minimal` (pip install without extras), `test` (run pytest), `test-cov` (pytest with coverage), `lint` (ruff check), `format` (ruff format), `demo` (run digest then two ask commands), `demo-paper` (digest with ollama), `clean` (remove cache and data directories), `env` (create .env from .env.example if not exists), `deps` (show pip list). The `demo` target uses `LLM_PROVIDER` from the environment and runs `digest 2401.12345 --auto --no-qa --json-out .data/briefing_check.json`, then an in-paper ask, then an off-topic ask.

---

## make.py

A minimal make wrapper for Windows (PowerShell does not natively support `make`). Parses the `Makefile` by reading targets and their tab-indented commands into a dictionary. The `run` function executes commands, converting `python` to the current Python executable path. The `main` function handles `make <target> [-f Makefile]` syntax, looks up the target, and runs each command sequentially, exiting with the return code of any failing command.

---

## make.cmd

A Windows batch file that delegates to `make.py`. Simply runs `python make.py %*`, passing all arguments through. This allows Windows users to use `make demo` syntax even though Windows lacks native make.

---

## src/agent/__init__.py

The package initialization file. Imports and re-exports the public API: `get_settings`, `reset_settings` from config; `PaperMeta`, `Chunk`, `KeyResult`, `Briefing`, `QAAnswer`, `AgentError` from models; and `AgentState` from state. The `__all__` list defines the public interface of the `agent` package.

---

## src/agent/config.py

The configuration management module. Loads environment variables via `dotenv.load_dotenv()`. Defines a frozen dataclass `Settings` with fields for every environment variable listed in `.env.example`: `llm_provider`, `groq_api_key`, `gemini_api_key`, `ollama_base_url`, `ollama_model`, `embedding_model`, `chroma_persist_dir`, `sqlite_db_path`, `pdf_cache_dir`, `arxiv_max_results`, `arxiv_sort_by`, `chunk_target_tokens`, `chunk_overlap_tokens`, `abstain_threshold` (deprecated), `abstain_max_distance`, `qa_top_k`, `qa_mmr_lambda`, `log_level`. The `__post_init__` method creates required directories (`chroma_persist_dir`, `sqlite_db_path.parent`, `pdf_cache_dir`). Provides `get_settings()` for lazy singleton initialization and `reset_settings()` to clear the singleton (primarily for testing). The `abstain_threshold` field is documented as deprecated because the similarity-based gate never fired with bge-small's compressed scores.

---

## src/agent/models.py

Pydantic data models for the entire system. Defines: `PaperMeta` (arxiv_id, title, authors, summary, published, updated, categories, primary_category, url, pdf_url, entry_id, journal_ref, doi, comment) representing arXiv paper metadata; `Chunk` (id, arxiv_id, text, section, chunk_index, page_start, char_start, embedding) for vector storage; `KeyResult` (claim, evidence, source_section) for structured key results; `Briefing` (arxiv_id, title, authors, published, categories, url, pdf_url, why_it_matters, problem_statement, method, key_results, limitations, followup_questions, meta) the main output model; `QAAnswer` (answer, citations, grounded) for QA responses; `AgentError` (code, node, detail, recoverable) for structured error tracking. Also defines type aliases: `Intent = Literal["paper_lookup", "topic_search", "unclear"]` and `ParseMode = Literal["full", "degraded_abstract_only", "failed"]`.

---

## src/agent/state.py

The LangGraph shared state definition using `TypedDict(total=False)`. All fields are optional to allow partial updates from individual nodes. Defines: input fields (`raw_input`, `intent`, `arxiv_id`, `search_query`), retrieval fields (`candidates`, `selection_reason`, `paper`), parsing fields (`pdf_path`, `sections`, `full_text`, `parse_mode`), indexing fields (`collection`, `n_chunks`), output fields (`briefing`, `question`, `retrieved`), and control fields (`errors`, `warnings`, `messages`, `retries`). Three fields use `Annotated[list[X], operator.add]` reducers (`errors`, `warnings`, `messages`) so they accumulate across node invocations instead of overwriting. The `errors` field stores `AgentError` dicts with `code`, `node`, `detail`, `recoverable`.

---

## src/agent/graph.py

The LangGraph workflow definition. Contains: `_start()` — a pass-through entry node returning `{}` (must return empty dict to avoid duplicating `operator.add` lists); `_route_mode()` — routes to `qa_node` if `state["question"]` is truthy, otherwise `query_understanding`; `_get_memory_checkpointer()` and `_sqlite_checkpointer_context()` — manage checkpointer instances; `_route_intent()` — routes to `fetch_metadata` for `paper_lookup` or `search_arxiv` for `topic_search`; `_route_candidate_count()` — routes to `broaden_query` for zero candidates, `select_paper` for one or more; `_route_parse_quality()` — routes to `chunk_embed` for full parse, `degrade_mode` for degraded/failed; `_should_continue_broaden()` — checks if broaden attempts are exhausted and whether candidates were found; `build_graph(checkpointer)` — constructs and compiles the `StateGraph` with all 13 nodes and all edges/conditional edges; `get_graph()` — returns a graph with `InMemorySaver` for inspection; `get_persistent_graph()` — a context manager yielding a graph with `SqliteSaver` for persistent sessions. The graph topology is: start → query_understanding → (fetch_metadata → fetch_pdf → parse → chunk_embed → summarize → END) or (search_arxiv → broaden_query → select_paper → fetch_pdf → ...) with a separate qa_node path from start.

---

## src/agent/cli.py

The Typer-based CLI application with three commands and a global callback. Contains: `_slugify()` for URL-safe slugs; `_get_thread_id()` deriving thread_id from input (arXiv ID if present, `topic:<slug>` otherwise); `_set_verbose()`, `_print_verbose()` for debug output; `_apply_overrides()` applying CLI flags to environment variables and resetting settings; `_run()` executing the graph with optional per-node streaming for `--verbose`; `_show_answer()` printing the last assistant answer with citations; `_ask_once()` running a single question turn; `digest` command — takes `raw_input`, `--auto`, `--json-out`, `--no-qa`, `--provider`, `--model`, `--top-k`, `--verbose`; derives thread_id, resets question to None (critical because `question` persists in checkpoint and routes to qa_node), runs the graph via `get_persistent_graph()`, prints warnings and briefing, optionally writes JSON, starts QA REPL; `_qa_repl()` interactive question loop until blank/exit/quit; `ask` command — takes `thread_id` and `question`, checks session exists, runs one question; `sessions` command — lists all distinct thread_ids from the SqliteSaver checkpointer; `main` callback — global `--provider`, `--model`, `--top-k`, `--verbose` options applied before subcommands. Also handles Windows console UTF-8 encoding fix.

---

## src/agent/nodes/__init__.py

Empty package init file (single line docstring). The nodes are imported individually by `graph.py` rather than through this package init.

---

## src/agent/nodes/query_understanding.py

The query understanding node. Implements `query_understanding(state)` which determines user intent and extracts/normalizes the query. Logic: if `raw_input` is empty, returns `intent="unclear"` with an `EMPTY_INPUT` error; tries regex `ARXIV_ID_PATTERN` (`(\d{4}\.\d{4,5})(v\d+)?`) to detect new-style arXiv IDs and `OLD_ARXIV_PATTERN` (`[a-z\-]+(\.[A-Z]{2})?/\d{7}`) for old-style IDs — if found, returns `intent="paper_lookup"` with the `arxiv_id`; if no match, calls `_normalize_query_with_llm()` which uses `complete_json` with a `NormalizedQuery` Pydantic model to get the LLM to produce a normalized query and categories, then builds an arXiv search query via `_build_arxiv_query()`. If the LLM fails, falls back to using the raw input as the search query. `_build_arxiv_query()` filters out stopwords and short terms, wraps each in `all:"..."`, and optionally adds `cat:` filters.

---

## src/agent/nodes/retrieval.py

Contains three retrieval nodes: `search_arxiv`, `broaden_query`, `fetch_metadata`. `search_arxiv(state)` — searches arXiv using `ArxivClient.search()` with the `search_query` from state, converts `PaperMeta` results to dicts via `model_dump(mode="json")`, returns `candidates`. `broaden_query(state)` — runs when zero results are returned; tracks attempt count via `retries.broaden_query`; attempt 1 drops quotes and `cat:` filters and changes AND to OR; attempt 2 keeps only the two highest-IDF terms and widens sort to `SubmittedDate`; after 2 attempts returns a `ZERO_RESULTS_EXHAUSTED` error with `search_query=None` signaling termination. `fetch_metadata(state)` — fetches metadata for a specific arXiv ID using `ArxivClient.fetch_metadata()`, returns the paper dict and adds it to candidates. All three nodes return `PaperMeta` dicts converted via `model_dump(mode="json")` to ensure serializability.

---

## src/agent/nodes/selection.py

The paper selection node. Implements `select_paper(state)` which ranks candidates and picks the best. Logic: takes top 10 candidates, gets LLM relevance scores via `_get_llm_relevance()` which uses a `RootModel[list[RelevanceScore]]` schema (to match the prompt's bare array output), calculates a composite score = `0.6 * llm_relevance + 0.25 * recency_decay + 0.15 * has_full_text`, sorts by composite descending, auto-selects rank 1, and stores `selection_reason`. `_calculate_recency_score()` uses exponential decay based on publication date (10 for <30 days, 7 for <365 days, 4 for <3 years, 1 for older). `_calculate_full_text_score()` returns 10 if `pdf_url` exists, 5 otherwise. `_format_selection_reason()` creates a human-readable string showing the composite score and individual component scores.

---

## src/agent/nodes/fetch_parse.py

Contains two nodes: `fetch_pdf` and `parse`, plus `degrade_mode`. `fetch_pdf(state)` — gets the paper dict and PDF URL, checks retry count, calls `download_pdf()` from `pdf_parser` service, checks file size (>40MB warn, >100MB degrade), returns `pdf_path`. `parse(state)` — gets the PDF path and paper, calls `parse_pdf()` which runs PyMuPDF primary → pdfplumber fallback → degrade to abstract-only; returns `sections`, `full_text`, `parse_mode`. `degrade_mode(state)` — handles degraded parsing by extracting the abstract from `paper.summary` and creating a single "Abstract" section, appends a warning, sets `parse_mode="degraded_abstract_only"`.

---

## src/agent/nodes/indexing.py

The indexing node. Implements `chunk_embed(state)` which chunks sections, embeds them, and upserts to ChromaDB. Logic: gets sections and paper, sanitizes `arxiv_id` for the collection name (`f"paper_{safe_id}"`), checks if collection already has chunks (cheap resume — skips if `existing_count > 0`), calls `chunk_sections()` from the `chunker` service with configured `target_tokens` and `overlap_tokens`, calls `upsert_chunks()` to store in ChromaDB, returns `collection` name and `n_chunks`. If no sections or paper, returns appropriate errors.

---

## src/agent/nodes/summarize.py

The summarization node using map-reduce. Implements `summarize(state)` which generates a structured `Briefing`. Logic: `_map_phase()` iterates over informative sections (abstract, introduction, method, results, etc.), calls `complete_json(BulletList, ...)` per section to extract 3-5 factual bullets with section prefixes. `_reduce_phase()` calls `complete_json(Briefing, ...)` with all bullets to synthesize the full briefing, with a JSON repair loop that catches `ValidationError` and `json.JSONDecodeError` and retries once with error feedback, falling back to `_fallback_briefing()` on second failure. `_limitations_guard()` ensures `limitations` is never empty — if it contains filler terms, runs a second retrieval pass over chunks for limitations queries, and if still nothing, emits a reviewer-inferred limitation. After generating, constructs `meta` dict (model, parse_mode, n_chunks, warnings, generated_at), writes `examples/briefing_<id>.json`, prints Markdown via Rich, returns `briefing.model_dump(mode="json")`.

---

## src/agent/nodes/qa.py

The QA grounding node. Implements `qa_node(state)` which answers questions using grounded retrieval. Logic: Step 1 — query rewrite if conversation history exists and question is a follow-up (≤5 words or contains reference words like "it", "they", "that") via `_rewrite_query()` which calls `complete_text()` with history; Step 2 — retrieve top 20 chunks via `query_chunks()`; Step 3 — abstain gate: if best raw cosine distance > `abstain_max_distance` (default 0.45), return the canned "That isn't covered in this paper." message without calling the LLM; Step 3b — MMR rerank (λ=0.6) to keep top 6 chunks; Step 4 — build context blocks and generate answer via `_generate_answer()` which calls `complete_json()` with citation references [S1], [S2], etc.; Step 5 — post-check citations: every [Sn] cited must exist in the citation mapping, strip invalid ones; Step 6 — if no valid citations remain, return the canned abstain message; Step 7 — return only the NEW messages for this turn (not the full history) since `messages` uses `operator.add` reducer. `_mmr_rerank()` implements maximal marginal relevance using Jaccard similarity. `_prepare_citations()` maps S1/S2 labels back to actual chunk IDs and extracts full metadata.

---

## src/agent/services/__init__.py

Empty package init file (single line docstring). Services are imported individually by node modules.

---

## src/agent/services/arxiv_client.py

The arXiv API client wrapper. `ArxivClient.__init__()` creates an `arxiv.Client` with configured page size, 3-second delay, and no internal retries (handled by the wrapper). `search(query, max_results, sort_by)` — creates an `arxiv.Search`, executes with `_execute_with_retry()` which uses `tenacity` with exponential backoff (3 attempts, min 2s, max 10s), converts results to `PaperMeta` via `_paper_to_meta()`. `fetch_metadata(arxiv_id)` — searches by ID list, returns single `PaperMeta` or None. `_paper_to_meta(paper)` — converts `arxiv.Result` to `PaperMeta`, extracting the arXiv ID via `_extract_arxiv_id()` which uses `re.sub(r"v\d+$", "", raw)` to strip only trailing version suffixes (not literal 'v' characters). `build_search_query(terms, categories)` — constructs arXiv search queries with `all:"..."` terms and optional `cat:` filters. `get_arxiv_client()` provides lazy initialization.

---

## src/agent/services/pdf_parser.py

PDF parsing with dual-backup and quality gating. `parse_pdf(pdf_path, arxiv_id, abstract_fallback)` — tries PyMuPDF first via `_parse_with_pymupdf()`, falls back to pdfplumber via `_parse_with_pdfplumber()`, degrades to abstract-only if both fail. `_dedupe_repeated_blocks(text)` — strips verbatim-repeated blocks caused by two-column LaTeX layouts using `difflib.SequenceMatcher`, keeping the first occurrence. `_parse_with_pymupdf()` — extracts text with `sort=True`, applies `_dedupe_repeated_blocks()`, computes quality metrics (`chars_per_page`, `alpha_ratio`), returns "failed" if `chars_per_page < 120` or `alpha_ratio < 0.6`, otherwise extracts sections via `_extract_sections_pymupdf()` which tries regex-based extraction first then font-size heuristic, truncates References to 4000 chars. `_parse_with_pdfplumber()` — same quality gate but uses regex-only section extraction (no font info). `_extract_sections_regex()` — finds section headings via `SECTION_REGEX`, splits text into sections. `_add_page_numbers_pymupdf()` — adds `page_start` by searching for section titles in pages. `_extract_sections_font_size()` — uses PyMuPDF `dict` output with font-size heuristic (>12pt, <100 chars) for heading detection. `download_pdf(arxiv_id, pdf_url)` — downloads PDF to `.data/pdfs/{safe_id}.pdf`, checks content-length, skips if cached.

---

## src/agent/services/chunker.py

Section-aware recursive chunking. `chunk_sections(sections, arxiv_id, target_tokens, overlap_tokens)` — iterates over sections, splits each section's text via `_split_text_into_chunks()`, creates `Chunk` objects with deterministic IDs `f"{arxiv_id}:{chunk_index}"`. `_split_text_into_chunks(text, target_tokens, overlap_tokens)` — uses character-based sliding window (4 chars ≈ 1 token), target_chars = target_tokens * 4, stride = target_chars - overlap_chars, tries to extend to sentence/paragraph boundaries, then overlaps consecutive chunks by prepending the tail of the previous chunk. `_get_embedding_model()` — lazy singleton for `SentenceTransformer` model. `_count_tokens()` — rough estimate (4 chars per token). The chunker never crosses section boundaries because it processes each section independently.

---

## src/agent/services/vectorstore.py

ChromaDB vector store operations. `get_collection(name)` — gets or creates a ChromaDB collection with `hnsw:space: "cosine"` metadata. `upsert_chunks(collection_name, chunks)` — prepares ids, documents, metadatas, generates embeddings via `_get_embedding_model()`, upserts to the collection. `query_chunks(collection_name, query_text, n_results)` — generates query embedding, calls `collection.query()`, returns list of dicts with `chunk_id`, `text`, `section`, `page_start`, `distance`. `get_collection_count(collection_name)` — returns the count of chunks in a collection (returns 0 on error). The client is lazily initialized as a `PersistentClient` pointing to `chroma_persist_dir`.

---

## src/agent/services/llm.py

LLM provider abstraction with JSON repair loop. `PROVIDER_CONFIGS` dict defines groq (`openai/gpt-oss-120b`), gemini (`gemini-1.5-flash`), and ollama (`qwen2.5:7b-instruct`) configurations. `LLMClient.__init__()` selects config based on provider. `_get_client()` lazily initializes the provider client: for groq uses `OpenAI` with the API key and base URL; for gemini uses `google.generativeai` SDK; for ollama uses `OpenAI` with dummy key "ollama" and local base URL. `complete_text(system_prompt, user_prompt)` — for gemini uses `generate_content()`, for others uses `chat.completions.create()` with temperature=0.1, max_tokens=4000. `complete_json(schema, system_prompt, user_prompt)` — calls `complete_text()`, extracts JSON via `_extract_json()` (handles markdown code fences and raw JSON), validates with `schema.model_validate()`, retries once with error feedback on `ValidationError` or `json.JSONDecodeError`. `_extract_json()` — handles ```json fences, ``` fences, and raw { }/[] blocks. Module-level functions `complete_json()` and `complete_text()` create `LLMClient` instances and delegate.

---

## src/agent/services/prompts.py

All prompt templates in one place. System prompts: `SYSTEM_SUMMARIZE_MAP` (extract 3-5 factual bullets from a section), `SYSTEM_SUMMARIZE_REDUCE` (synthesize bullets into Briefing JSON), `SYSTEM_QA` (answer ONLY from context blocks with [S1] citations), `SYSTEM_QUERY_REWRITE` (resolve pronouns using conversation history), `SYSTEM_QUERY_NORMALIZE` (normalize topic to arXiv search query), `SYSTEM_SELECT_PAPER` (score papers 0-10), `SYSTEM_LIMITATIONS_PASS` (extract limitations from chunks). User prompt template functions: `summarize_map_prompt(section_title, section_text)`, `summarize_reduce_prompt(paper_meta, all_bullets)`, `qa_prompt(question, context_blocks)` (builds [S1]...[Sn] context blocks), `query_rewrite_prompt(question, history)`, `query_normalize_prompt(raw_input)`, `select_paper_prompt(query, candidates)`, `limitations_pass_prompt(chunks)`. `get_prompt(name, **kwargs)` — dispatches to the appropriate prompt template by name.

---

## test_meta.py

A scratch diagnostic script (not a formal test). Fetches metadata for arXiv ID 2401.12345 using `ArxivClient`, prints the type name and URL fields to verify `PaperMeta` has `url` and `pdf_url` attributes correctly populated. Used during development to verify the `_paper_to_meta` refactoring.

---

## test_qa.py

A scratch diagnostic script (not a formal test). Builds the graph with `InMemorySaver`, invokes it with arXiv ID 2401.12345, then streams two QA questions through the graph printing per-node message counts. Used to debug the multi-turn QA history duplication bug that was later fixed by changing `_start` from `lambda state: state` to returning `{}`.

---

## test_search_dicts.py

A scratch diagnostic script (not a formal test). Tests that topic search ("recent work on KV-cache compression for LLMs") correctly produces a paper title and URL in the result state. Used to verify the topic search → selection → metadata retrieval pipeline.

---

## test_summarize.py

A scratch diagnostic script (not a formal test). Tests that summarization produces a valid briefing with a populated `meta` field, and verifies the saved `examples/briefing_2401.12345.json` file has matching meta data. Used to debug the issue where `_write_briefing()` was called before `briefing.meta` was populated.

---

## check_abstain.py

The abstain gate acceptance and calibration script. Runs a multi-turn QA check (3 questions: two in-paper, one off-topic) and verifies the off-topic question returns the exact abstain message. Then runs distance calibration: queries the vector store with in-paper questions and off-topic questions, prints the best distance for each, and determines whether `ABSTAIN_MAX_DISTANCE` creates a clean separation between in-paper and off-topic distances. Supports running on different papers (2401.12345 or 1706.03762) with paper-specific question sets.

---

## trace_qa.py

A scratch diagnostic script (not a formal test). Identical to `test_qa.py` — builds the graph, invokes with arXiv ID, streams two QA questions, prints per-node message counts. Used to trace the QA flow and debug the multi-turn history duplication bug.

---

## scripts/check_abstain.py

The production version of `check_abstain.py`, moved from repo root to `scripts/` during cleanup (Stage 8e). Same logic: multi-turn QA verification and distance calibration, but with proper `sys.path` insertion and docstring. Run as `python scripts/check_abstain.py [arxiv_id]`.

---

## scripts/test_qa.py

The production version of `test_qa.py`, moved to `scripts/`. Same scratch diagnostic logic for tracing QA flow with proper path setup.

---

## scripts/trace_qa.py

Identical to `scripts/test_qa.py` — a duplicate of the scratch diagnostic script moved to `scripts/`.

---

## tests/conftest.py

Shared test fixtures and configuration. Creates a mock `sentence_transformers` module before any real imports (avoiding the ~11s initialization delay and network access). Defines three autouse fixtures: `isolated_settings` — points `CHROMA_PERSIST_DIR`, `SQLITE_DB_PATH`, `PDF_CACHE_DIR` at `tmp_path` and calls `reset_settings()` between tests; `_block_network` — monkeypatches `socket.socket.connect` to raise `RuntimeError` for any non-localhost address; `_mock_embedding_model` — replaces `_get_embedding_model` and `_get_client` with `MagicMock` objects returning 384-dim zero vectors. Also provides `sample_sections` fixture with three sections (Abstract, Introduction, Method) for chunker/parser tests.

---

## tests/test_query_understanding.py

Tests for the `query_understanding` node. Seven test cases: `test_new_style_arxiv_id` (2401.12345 → paper_lookup), `test_arxiv_id_with_v_suffix_stripped` (2401.12345v2 → arxiv_id without v2), `test_arxiv_org_abs_url` (URL → arxiv_id), `test_arxiv_org_pdf_url` (PDF URL → arxiv_id), `test_old_style_id` (cs.CL/1234567 → paper_lookup), `test_free_text_topic_search` (no arXiv ID → topic_search with LLM stub), `test_llm_failure_falls_back_to_raw_string` (LLM down → raw string as search_query), `test_empty_input_returns_unclear` (empty input → unclear intent with error).

---

## tests/test_chunker.py

Tests for `chunk_sections`. Five test cases: `test_no_chunk_crosses_section_boundary` (chunks stay within their section), `test_consecutive_chunks_inside_section_overlap` (overlapping text between consecutive chunks), `test_ids_are_arxiv_id_index` (IDs are `f"{arxiv_id}:{i}"`), `test_re_chunking_gives_identical_ids` (deterministic IDs on re-chunk), `test_empty_section_skipped` (empty text sections produce no chunks).

---

## tests/test_parser_fallback.py

Tests for PDF parsing fallback. Two test cases: `test_corrupt_pdf_returns_degraded_abstract_only` (garbage PDF → degraded_abstract_only with abstract fallback), `test_corrupt_pdf_no_exception_raised` (corrupt PDF must not raise any exception). Uses `tmp_path` for temporary files.

---

## tests/test_selection_zero_results.py

Tests for zero-result search and broadening. Six test cases: `test_search_returns_empty_triggers_broaden`, `test_broaden_runs_at_most_twice` (verifies attempt counter), `test_broaden_terminates_cleanly_with_actionable_message`, `test_search_returns_empty_then_broaden_successfully` (regression for successful broaden routing to select_paper), `test_successful_broaden_no_redundant_search` (regression for the §11 Stage 5 bug), `test_zero_results_ends_after_two_broaden_attempts` (full simulation of zero-result path through graph).

---

## tests/test_qa_abstain.py

Tests for the QA abstain gate and query rewrite. Two test classes: `TestNeedsRewrite` — `test_standalone_question_verbatim` (>5 words, no reference words → no rewrite), `test_short_follow_up_is_rewritten` (≤5 words → rewrite), `test_pronoun_follow_up_is_rewritten` (contains reference word → rewrite). `TestQaAbstain` — `test_abstain_when_distance_above_threshold` (distance > 0.45 → canned message, no LLM call), `test_llm_called_when_distance_in_range` (distance < 0.45 → LLM called), `test_no_valid_citations_returns_canned_message` (LLM answer with unknown citations → abstain message).

---

## tests/test_multiturn_history.py

Tests for persistent multi-turn QA history. Three test cases: `test_three_turns_message_counts` (three `graph.invoke()` calls produce 2, 4, 6 messages respectively), `test_start_returns_empty_dict` (the `_start` node must return `{}` to avoid `operator.add` duplication — regression fix for §11 Stage 7), `test_each_turn_has_exactly_one_qa` (each turn adds exactly 2 messages).

---

## tests/test_citations.py

Tests for the [Sn] → chunk_id citation mapping. Four test cases: `test_sn_to_chunk_id_keeps_prompt_order` (S1→chunk 0, S2→chunk 1, etc.), `test_unknown_sn_is_dropped` (S99 in citations is validated away), `test_final_citations_have_required_fields` (chunk_id, section, page, snippet all present with correct values), `test_build_context_blocks_preserves_order` (context blocks match MMR chunk order).

---

## tests/test_zero_results_graph.py

End-to-end test of the zero-results failure path using the real compiled graph. Patches `ArxivClient.search` to return empty, stubs `complete_json` in `query_understanding`, and patches all downstream nodes to raise `AssertionError` if ever reached. Verifies: the graph ends cleanly without errors, never reaches downstream nodes, `retries.broaden_query == 2`, `search_query is None`, exactly 5 search calls (original + 2 broadening attempts × 2 searches each, minus one exhausted attempt), and the `ZERO_RESULTS_EXHAUSTED` error entry contains the expected message without a traceback.

---

## examples/briefing_2401_12345.json

The real briefing for the example paper "Distributionally Robust Receive Combining" (arXiv 2401.12345). Contains all fields of the `Briefing` model: arxiv_id, title, authors, published, categories, url, pdf_url, why_it_matters, problem_statement, method (5 bullets), key_results (3 entries with evidence references), limitations (1 reviewer-inferred item), followup_questions (5 questions), and meta (model: ollama, parse_mode: full, n_chunks: 65, warnings: [], generated_at). This is the canonical example used in README and sample_session.md.

---

## examples/briefing_2609_07966.json

Briefing for "MetaKV: Adaptive KV Cache Compression for Constrained LLM Inference" (arXiv 2609.07966). Contains the same Briefing structure with 5 method bullets, 3 key_results, 1 limitation, 5 followup_questions, and populated meta (model: ollama, parse_mode: full, n_chunks: 26).

---

## examples/briefing_1706_03762.json

Briefing for the famous "Attention Is All You Need" (arXiv 1706.03762). Contains the Transformer paper's briefing with 4 method bullets, 3 key_results (BLEU scores), 1 limitation, 5 followup_questions, and populated meta (model: ollama, parse_mode: full, n_chunks: 28). This paper is used for abstain gate calibration.

---

## examples/briefing_2301_12345.json

Briefing for "Chemotactic motility-induced phase separation" (arXiv 2301.12345). Contains the briefing with 3 method bullets, 3 key_results, 1 limitation, 5 followup_questions. Notably has `"meta": {}` — the meta field was not populated (an older example generated before the meta fix in summarize.py).

---

## examples/briefing_2201_12345.json

Briefing for "On Stability and Convergence of a Three-layer Semi-discrete Scheme..." (arXiv 2201.12345). Contains the briefing with 3 method bullets, 3 key_results, 1 limitation, 5 followup_questions. Also has `"meta": {}`.

---

## examples/sample_session.md

A real, unedited transcript of the agent in action. Documents five scenarios: (1) the digest run output, (2) an in-paper ask with citations showing `node=start` and `node=qa_node` transitions only, (3) a pronoun follow-up ("How does it compare to the Wiener beamformer?") with the rewritten query and citations, (4) the off-topic question ("What does this paper say about the 2026 World Cup?") returning the exact abstain message, (5) the `sessions` command output showing saved thread IDs. The ANSI codes are stripped and the document confirms the abstain line matches exactly.

---

## .data/briefing_check.json

A copy of the briefing generated during the demo run, written by `make demo` to `.data/briefing_check.json`. Contains the same structure as `examples/briefing_2401_12345.json`.

---

## .data/briefing_final.json

Another version of the briefing for paper 2401.12345 with slightly different wording in why_it_matters and key_results (evidence field has "Not explicitly stated by the authors; reviewer-inferred: Figure 4" for the first key result — evidence drift). Contains populated meta.

---

## .data/mermaid.txt, .data/mermaid_graph.txt, .data/mermaid_clean.txt

Plain text copies of the Mermaid graph diagram generated by `python -c "from agent.graph import build_graph; print(build_graph().get_graph().draw_mermaid())"`. Used to paste into `docs/architecture.md`. All three files contain identical content showing the full graph with all nodes and edges.

---

## .data/chroma/

The persistent ChromaDB database directory. Contains collections named `paper_{arxiv_id_safe}` storing chunk embeddings generated by `BAAI/bge-small-en-v1.5`. The collections survive process restarts and allow QA to re-attach without re-indexing.

---

## .data/sessions.sqlite

The SQLite database for LangGraph `SqliteSaver` checkpointer. Stores all node return values as checkpoints keyed by `thread_id`. When `ask` is called with a known thread_id, the checkpointer restores the full state (paper, collection, messages) so QA runs without re-parsing.

---

## .data/pdfs/

Directory containing cached PDF downloads. Files are named `{arxiv_id}.pdf`. The `download_pdf()` function checks this directory before attempting a new download, making repeated runs on the same paper fast.

---

## .gitignore

Defines files and directories to exclude from git. Includes: `.env`, `.env.local`, `.data/`, `*.sqlite`, `*.pdf`, `__pycache__/`, `*.py[cod]`, `.pytest_cache/`, `.ruff_cache/`, `.coverage`, `htmlcov/`, `*.egg-info/`, `.vscode/`, `.idea/`, `.DS_Store`, `Thumbs.db`, `*.log`.

---

## .gitattributes

Sets `* text=auto` to normalize line endings across all files. Ensures consistent text handling across platforms.
