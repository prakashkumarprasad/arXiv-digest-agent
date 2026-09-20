# arXiv Digest Agent

Autonomous agent that fetches, parses, summarizes, and answers questions about arXiv papers.

## Architecture

See [docs/architecture.md](docs/architecture.md) for the full architecture, including the LangGraph workflow diagram, state table, and failure paths.

## State Table (short)

The agent uses a `TypedDict`-based `AgentState`. Key fields: `raw_input`, `intent`, `arxiv_id`, `candidates`, `paper`, `sections`, `full_text`, `parse_mode`, `collection`, `briefing`, `question`, `messages`. Fields marked `operator.add` (`candidates`, `messages`, `errors`, `warnings`) accumulate across node invocations instead of overwriting.

## Setup

### Ollama (no API key needed)

```bash
pip install -e .
ollama pull qwen2.5:7b-instruct
ollama serve
LLM_PROVIDER=ollama python -m agent.cli digest 2401.12345 --auto --no-qa
```

### Groq (free tier)

```bash
pip install -e ".[groq]"
export GROQ_API_KEY="your-key"
LLM_PROVIDER=groq python -m agent.cli digest 2401.12345 --auto --no-qa
```

The LLM provider model is `openai/gpt-oss-120b` on Groq. Provider model names can and do change — check the provider's current model list if you see a 404.

## Example Run

From [examples/sample_session.md](examples/sample_session.md):

```
$ python -m agent.cli digest 2401.12345 --auto --no-qa --json-out .data/briefing_check.json
```

Then ask questions:

```
$ python -m agent.cli ask 2401.12345 "What datasets did they evaluate on?" --verbose
$ python -m agent.cli ask 2401.12345 "How does it compare to the Wiener beamformer?" --verbose
$ python -m agent.cli ask 2401.12345 "What does this paper say about the 2026 World Cup?"
That isn't covered in this paper.
$ python -m agent.cli sessions
```

## Rate Limits & Model Notes

The default LLM model on Groq is `openai/gpt-oss-120b`. Provider model names change frequently — if a provider call fails with a 404/`model_not_found`, check the provider's current model list before assuming a code bug. The Ollama model is `qwen2.5:7b-instruct` (local, no rate limits).

## Design Decisions & Tradeoffs

- **LangGraph checkpointer + `thread_id`**: The graph is compiled with `SqliteSaver` for persistent sessions. `thread_id` is the arXiv ID when available, or `topic:<slug>` for topic searches. This allows `ask` to re-attach to a previous session without re-parsing.
- **Section-aware chunking at 400/80 tokens vs the 512-token bge window**: Chunks are bounded to 400 tokens with 80-token overlap to fit within bge-small's 512-token window. Larger chunks would be silently truncated at embedding time.
- **Local embeddings**: Uses `BAAI/bge-small-en-v1.5` via `sentence-transformers`. No API key, CPU-fast, 384-dim embeddings pinned via `get_collection()`.
- **PDF fallback chain and duplicate-block dedupe**: PyMuPDF primary, pdfplumber fallback. `_dedupe_repeated_blocks()` strips verbatim repeated blocks common in two-column LaTeX layouts. Do not remove this — without it, chunk quality silently degrades.
- **Distance-based abstain gate vs. similarity gate**: The original similarity gate (`1 - distance < 0.35`) never fired because bge-small distances are compressed. We gate on raw cosine distance with `ABSTAIN_MAX_DISTANCE=0.45` (calibrated on 2 papers).
- **Rewrite-only-for-follow-ups**: `_needs_rewrite` only triggers for short questions (≤5 words) or those containing reference words (it, they, that). Standalone questions are used verbatim so the rewrite cannot pull earlier topics into an off-topic question and defeat the abstain gate.
- **List-reducer rule**: `messages`, `errors`, `warnings`, `candidates`, `sections`, `retrieved` use `operator.add` reducers so they accumulate instead of overwriting. Any new list-accumulating state field needs the same treatment.
- **`topic:<slug>` thread IDs**: When the input is a topic search, the thread ID is `topic:<slug>`. This allows QA to attach to a session even before paper selection.
- **Gemini left unverified**: The Gemini provider code uses the deprecated `google.generativeai` package. Left as-is — Groq and Ollama already satisfy the no-paid-key constraint. Documented as unverified.

## Known Limitations

- Citation `section`/`page` is unreliable on the pdfplumber path (page is 0).
- Briefing `evidence` figure/table references vary between LLM runs (evidence drift).
- Abstain threshold (`ABSTAIN_MAX_DISTANCE=0.45`) was calibrated on 2 papers only.
- Windows Ollama has crashed with a CUDA stack-buffer-overrun error in this environment.
- Pronoun follow-ups asked right after an abstained turn can be rewritten toward the off-topic subject.
- Questions about paper metadata ("what is the name of the paper") abstain — chunks hold body text only.
- `--top-k` has no effect on `qa_node` (hardcoded to 6). `--auto` is not passed to the graph.

## What I'd Do Next

1. Index a header/abstract chunk so metadata questions work.
2. Add `node.py` with `top_k` and `mmr_lambda` read from settings instead of hardcoded values.
3. Verify the Gemini provider path with a valid API key.
4. Add a fresh-clone verification script (`scripts/check_abstain.py`).
5. Record `docs/demo.gif` — run `make demo` with node transitions visible (`--verbose`) and save the recording as `docs/demo.gif`.

## Demo

![Demo](docs/demo.gif)

## Provider Verification

| Provider | Model | Status |
|---|---|---|
| Ollama | `qwen2.5:7b-instruct` | ✅ Verified — `digest` and `ask` runs confirmed |
| Groq | `openai/gpt-oss-120b` | ❌ Not verified — no API key in this repo |
| Gemini | `gemini-1.5-flash` | ❌ Not verified — deprecated SDK, left as-is |
