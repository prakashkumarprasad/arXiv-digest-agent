"""CLI for the arXiv Digest Agent.

Commands:
- digest: Fetch, parse, index, and summarize a paper (or search by topic)
- ask: Ask questions about a previously digested paper (re-attaches via checkpointer)
- sessions: List all saved session threads
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

# Fix Windows console encoding issue
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer)
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer)

from agent.config import get_settings, reset_settings
from agent.graph import get_persistent_graph

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()

# Global verbose flag
_verbose = False


def _set_verbose(v: bool) -> None:
    global _verbose
    _verbose = v


def _slugify(text: str) -> str:
    """Convert text to a URL-safe slug."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def _get_thread_id(raw_input: str) -> str:
    """Derive thread_id from raw input.

    If the input contains an arXiv ID, use that.
    Otherwise, use 'topic:<slug of input>'.
    """
    arxiv_id_pattern = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")
    match = arxiv_id_pattern.search(raw_input)
    if match:
        return match.group(1)
    old_style = re.compile(r"([a-z\-]+(\.[A-Z]{2})?/\d{7})")
    match = old_style.search(raw_input)
    if match:
        return match.group(1).replace("/", "_").replace(".", "_")
    return f"topic:{_slugify(raw_input)}"


def _print_verbose(msg: str) -> None:
    if _verbose:
        console.print(msg, style="dim", markup=False)


def _apply_overrides(provider: Optional[str], model: Optional[str], top_k: Optional[int]) -> None:
    """Apply CLI overrides through environment variables, then re-read settings."""
    if provider:
        os.environ["LLM_PROVIDER"] = provider
    if model:
        # Only the ollama model name is read from the environment in this build.
        os.environ["OLLAMA_MODEL"] = model
        if os.environ.get("LLM_PROVIDER", "ollama") != "ollama":
            console.print("[yellow]--model only takes effect for the ollama provider; ignored.[/yellow]")
    if top_k:
        os.environ["QA_TOP_K"] = str(top_k)
        # qa_node currently hardcodes its retrieval size (see BUILD_SPEC §11, Stage 8a).
        console.print("[yellow]--top-k is accepted, but qa_node currently ignores it (fixed at 6).[/yellow]")
    reset_settings()


def _run(graph: Any, inputs: dict, config: dict) -> dict:
    """Run the graph once and return the final state values.

    With --verbose, stream per-node updates and print one `node=<name>` line each.
    `stream` does not return the final state, so it is read back from the checkpoint.
    """
    if _verbose:
        for update in graph.stream(inputs, config=config, stream_mode="updates"):
            for node in update:
                console.print(f"node={node}", style="dim", markup=False)
    else:
        graph.invoke(inputs, config=config)
    return graph.get_state(config).values


def _show_answer(values: dict, n_msgs_before: int, n_errs_before: int) -> bool:
    """Print the answer produced by the last run. Return False if there was none.

    The answer is the last assistant message in `messages`, and it only counts if the
    run actually appended messages: comparing counts stops a failed run from
    re-displaying the previous turn's answer.
    """
    messages = values.get("messages", [])
    if len(messages) > n_msgs_before and messages[-1].get("role") == "assistant":
        last = messages[-1]
        console.print()
        console.print(last.get("content", ""), markup=False, highlight=False)
        citations = last.get("citations") or []
        if citations:
            console.print("\nCitations:", style="dim", markup=False)
            seen: set[str] = set()
            for c in citations:
                chunk_id = c.get("chunk_id", "?")
                if chunk_id in seen:
                    continue
                seen.add(chunk_id)
                page = c.get("page") or "?"
                console.print(
                    f"  • {c.get('section', 'Unknown')}, p.{page}  ({chunk_id})",
                    style="dim",
                    markup=False,
                )
        return True

    console.print("[red]No answer generated[/red]")
    for e in values.get("errors", [])[n_errs_before:]:
        console.print(f"  {e.get('code', 'ERROR')}: {e.get('detail', '')}", style="red", markup=False)
    return False


def _ask_once(graph: Any, config: dict, question: str) -> bool:
    """Ask one question on an existing thread and print the result."""
    before = graph.get_state(config).values
    n_msgs = len(before.get("messages", []))
    n_errs = len(before.get("errors", []))
    values = _run(graph, {"question": question}, config)
    return _show_answer(values, n_msgs, n_errs)


@app.command()
def digest(
    raw_input: str = typer.Argument(..., help="arXiv ID, URL, or topic query"),
    auto: bool = typer.Option(False, "--auto", help="Auto-select first paper (no interactive prompt)"),
    json_out: Optional[Path] = typer.Option(None, "--json-out", help="Write briefing JSON to file"),
    no_qa: bool = typer.Option(False, "--no-qa", help="Skip QA REPL after digest"),
    provider: Optional[str] = typer.Option(None, "--provider", help="LLM provider: groq, gemini, ollama"),
    model: Optional[str] = typer.Option(None, "--model", help="Override model name (ollama only)"),
    top_k: Optional[int] = typer.Option(None, "--top-k", help="Top-k for QA retrieval"),
    verbose: bool = typer.Option(False, "--verbose", help="Print node transitions"),
) -> None:
    """Digest a paper: fetch, parse, index, and summarize."""
    _set_verbose(verbose or _verbose)
    _apply_overrides(provider, model, top_k)
    # TODO(auto): `auto` is not passed to the graph yet; it needs the selection node's
    # state key. It only matters for topic searches, not for arXiv ID input.

    thread_id = _get_thread_id(raw_input)
    _print_verbose(f"Thread ID: {thread_id}")
    _print_verbose(f"Input: {raw_input}")

    config = {"configurable": {"thread_id": thread_id}}

    with get_persistent_graph() as graph:
        before = graph.get_state(config).values
        n_errs = len(before.get("errors", []))
        n_warns = len(before.get("warnings", []))
        # `question` persists in the checkpoint and `_route_mode` sends any truthy
        # question to qa_node, so a re-digest must reset it or it silently skips the digest.
        values = _run(graph, {"raw_input": raw_input, "question": None, "retries": {}}, config)

    for w in values.get("warnings", [])[n_warns:]:
        console.print(f"[yellow]Warning: {escape(w)}[/yellow]")

    briefing = values.get("briefing")
    if not briefing:
        console.print("[red]Error: no briefing generated[/red]")
        for e in values.get("errors", [])[n_errs:]:
            console.print(f"  {e.get('code', 'ERROR')}: {e.get('detail', '')}", style="red", markup=False)
        sys.exit(1)

    # The briefing itself is printed once, by the summarize node.
    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(briefing, indent=2, ensure_ascii=False), encoding="utf-8")
        console.print(f"[green]Briefing written to {escape(str(json_out))}[/green]")

    console.print(f"\n[cyan]Session saved. Thread ID: {escape(thread_id)}[/cyan]")
    console.print(
        f"[dim]Use 'python -m agent.cli ask {escape(thread_id)} \"your question\"' to ask questions[/dim]"
    )

    if no_qa:
        return

    _qa_repl(thread_id)


def _qa_repl(thread_id: str) -> None:
    """Interactive QA REPL for a digested paper (one checkpointer connection for the session)."""
    console.print("\n[bold cyan]QA mode[/bold cyan] — ask about the paper; blank line, 'exit' or 'quit' to leave")
    config = {"configurable": {"thread_id": thread_id}}

    with get_persistent_graph() as graph:
        while True:
            try:
                question = console.input("\n[bold cyan]Question>[/bold cyan] ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not question or question.lower() in ("exit", "quit"):
                break
            _ask_once(graph, config, question)


@app.command()
def ask(
    thread_id: str = typer.Argument(..., help="Thread ID (arXiv ID or topic:<slug>)"),
    question: str = typer.Argument(..., help="Question to ask"),
    provider: Optional[str] = typer.Option(None, "--provider", help="LLM provider: groq, gemini, ollama"),
    model: Optional[str] = typer.Option(None, "--model", help="Override model name (ollama only)"),
    top_k: Optional[int] = typer.Option(None, "--top-k", help="Top-k for QA retrieval"),
    verbose: bool = typer.Option(False, "--verbose", help="Print node transitions"),
) -> None:
    """Ask a question about a previously digested paper."""
    _set_verbose(verbose or _verbose)
    _apply_overrides(provider, model, top_k)

    config = {"configurable": {"thread_id": thread_id}}

    with get_persistent_graph() as graph:
        try:
            has_session = bool(graph.get_state(config).values)
        except Exception as e:
            console.print(f"[red]Could not read session {escape(thread_id)}: {escape(str(e))}[/red]")
            sys.exit(1)

        if not has_session:
            console.print(f"[red]No saved session for {escape(thread_id)} — run 'digest' first[/red]")
            sys.exit(1)

        ok = _ask_once(graph, config, question)

    if not ok:
        sys.exit(1)


@app.command()
def sessions() -> None:
    """List all saved session threads."""
    settings = get_settings()

    try:
        from langgraph.checkpoint.sqlite import SqliteSaver

        thread_ids = set()
        with SqliteSaver.from_conn_string(str(settings.sqlite_db_path)) as saver:
            for checkpoint_tuple in saver.list(None):
                config = checkpoint_tuple.config
                if config and "configurable" in config:
                    tid = config["configurable"].get("thread_id")
                    if tid:
                        thread_ids.add(tid)

        if not thread_ids:
            console.print("[dim]No saved sessions[/dim]")
            return

        table = Table(title="Saved Sessions")
        table.add_column("Thread ID", style="cyan")
        for tid in sorted(thread_ids):
            table.add_row(tid)
        console.print(table)
    except Exception as e:
        console.print(f"[red]Error listing sessions: {escape(str(e))}[/red]")
        sys.exit(1)


@app.callback()
def main(
    ctx: typer.Context,
    provider: Optional[str] = typer.Option(None, "--provider", help="LLM provider: groq, gemini, ollama"),
    model: Optional[str] = typer.Option(None, "--model", help="Override model name (ollama only)"),
    top_k: Optional[int] = typer.Option(None, "--top-k", help="Top-k for QA retrieval"),
    verbose: bool = typer.Option(False, "--verbose", help="Print node transitions"),
) -> None:
    """Global options for all commands (may be given before the subcommand)."""
    _set_verbose(verbose)
    _apply_overrides(provider, model, top_k)


if __name__ == "__main__":
    app()