"""dataset_search — flag-style CLI for autonomous dataset discovery.

Usage:
    dataset_search "<subject>" [--format FMT[,FMT]] [--time RANGE] [--depth 1|2|3]
                              [--model NAME] [--hours N | --unlimited]

Examples:
    dataset_search "military aircraft imagery" --format geotiff,tiff --time "2020-2024" --depth 2
    dataset_search "SAR ship detection" --format all --time "any" --depth 3 --unlimited
    dataset_search "aerial orthoimagery numpy" --format npy --depth 1
"""
import sys
import click
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.table import Table
from rich import box

from .agent import DatasetDiscoveryAgent
from .storage import Storage
from .models import Dataset
from .config import (
    OLLAMA_MODEL, OLLAMA_HOST, DB_PATH, DEFAULT_HOURS, DEFAULT_DEPTH,
)
from . import web_ui

console = Console()


DEPTH_DESCRIPTIONS = {
    1: "50% — fast first pass over the 12 known portals",
    2: "80% — deep coverage including papers + government portals",
    3: "100% — frontier mode, scans the entire internet, no iteration cap",
}


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("subject")
@click.option(
    "--format", "fmt",
    default="all",
    show_default=True,
    help='Required dataset format(s). Comma-separate multiple, or "all" for any format. e.g. geotiff,tiff,cog',
)
@click.option(
    "--time", "time_range",
    default="",
    show_default=False,
    help='Time range filter, free-form. e.g. "2020-2024", "last 5 years", or omit for any time.',
)
@click.option(
    "--depth",
    type=click.IntRange(1, 3),
    default=DEFAULT_DEPTH,
    show_default=True,
    help="How deep to scan: 1=fast, 2=deep (default), 3=frontier (whole internet).",
)
@click.option(
    "--model",
    default=None,
    help=f"Ollama model name. Default: {OLLAMA_MODEL} (auto-fallback if not pulled).",
)
@click.option(
    "--hours",
    type=float,
    default=DEFAULT_HOURS,
    show_default=True,
    help="Max search duration in hours. Ignored if --unlimited is set.",
)
@click.option(
    "--unlimited",
    is_flag=True,
    default=False,
    help="No time limit. Run until the agent calls mark_search_complete or you Ctrl+C. "
         "All findings are saved continuously, so Ctrl+C never loses data.",
)
@click.option(
    "--min-relevance",
    type=float,
    default=0.0,
    show_default=True,
    help="Minimum relevance score to display in the live stream (does not affect what gets stored).",
)
@click.option(
    "--web/--no-web",
    "web",
    default=True,
    show_default=True,
    help="Serve a live web dashboard at http://127.0.0.1:7860 during the run.",
)
@click.option(
    "--web-port",
    type=int,
    default=7860,
    show_default=True,
    help="Port for the web dashboard.",
)
@click.option(
    "--interactive/--no-interactive",
    default=False,
    help="Pause and prompt for user input between iterations.",
)
def main(
    subject: str,
    fmt: str,
    time_range: str,
    depth: int,
    model: str,
    hours: float,
    unlimited: bool,
    min_relevance: float,
    web: bool,
    web_port: int,
    interactive: bool,
):
    """Find datasets matching SUBJECT by autonomously scanning the internet.

    \b
    SUBJECT is a free-form description of what you need. Be specific.
    Examples of SUBJECT:
      "military aircraft imagery"
      "satellite SAR ship detection annotated"
      "aerial orthoimagery for building footprint extraction"
    """
    # Parse formats: "all" or comma-separated list
    formats: list[str] = []
    if fmt and fmt.lower() != "all":
        formats = [f.strip().lower() for f in fmt.split(",") if f.strip()]

    max_hours = None if unlimited else hours

    storage = Storage(DB_PATH)
    agent = DatasetDiscoveryAgent(storage, model=model)

    # Start web dashboard if requested.
    actual_web_port = None
    if web:
        actual_web_port = web_ui.start(port=web_port)
        if actual_web_port is not None:
            web_ui.publish({
                "type": "run_start",
                "subject": subject,
                "model": agent.model,
                "depth": depth,
                "max_iters": {1: 80, 2: 200, 3: 2000}.get(depth, 200),
                "min_needed": {1: 10, 2: 25, 3: 50}.get(depth, 25),
                "elapsed": 0,
            })

    # Header banner
    banner = (
        f"[bold]Subject:[/bold] [yellow]{subject}[/yellow]\n"
        f"[bold]Formats:[/bold] {', '.join(formats) if formats else 'ANY'}\n"
        f"[bold]Time:[/bold] {time_range or 'ANY'}\n"
        f"[bold]Depth:[/bold] {depth} — {DEPTH_DESCRIPTIONS[depth]}\n"
        f"[bold]Model:[/bold] {agent.model} @ {OLLAMA_HOST}\n"
        f"[bold]Duration:[/bold] {'UNLIMITED' if unlimited else f'{hours}h'}\n"
        f"[bold]Database:[/bold] {DB_PATH}"
    )
    console.print(Panel(banner, title="[bold cyan]dataset_search[/bold cyan]"))
    console.print(
        "[dim]Output format: [bold]page link  —  what is it  —  dataset link[/bold]\n"
        "Findings save to disk immediately. Ctrl+C is safe — nothing will be lost.[/dim]"
    )
    if actual_web_port is not None:
        console.print(
            f"[bold green]🌐 Live dashboard:[/bold green] [link]http://127.0.0.1:{actual_web_port}[/link]\n"
        )
    console.print("[dim]── live findings ──[/dim]")

    found_count = {"n": 0}

    def _fmt_elapsed(secs: float) -> str:
        secs = int(max(0, secs))
        h, rem = divmod(secs, 3600)
        m, s = divmod(rem, 60)
        if h:
            return f"{h}h{m:02d}m{s:02d}s"
        return f"{m:02d}m{s:02d}s"

    def _prefix(elapsed: float, iteration: int) -> str:
        return f"[dim][T+{_fmt_elapsed(elapsed)} · iter {iteration:>3}][/dim]"

    def on_stored(d: Dataset):
        if d.relevance_score < min_relevance:
            return
        found_count["n"] += 1
        # Mirror confirmed datasets to the web dashboard.
        if actual_web_port is not None:
            try:
                host = ""
                try:
                    from urllib.parse import urlparse
                    host = (urlparse(d.url).hostname or "").lower().removeprefix("www.")
                except Exception:
                    pass
                is_main = host in ("kaggle.com", "huggingface.co", "github.com") \
                    or any(host.endswith("." + m) for m in ("kaggle.com", "huggingface.co", "github.com"))
                web_ui.publish({
                    "type": "dataset_stored",
                    "is_mainstream": is_main,
                    "dataset": {
                        "name": d.name,
                        "url": d.url,
                        "download_url": d.download_url,
                        "relevance_score": d.relevance_score,
                        "license_spdx": d.license_spdx,
                        "country": d.country,
                        "institution": d.institution,
                        "doi": d.doi,
                    },
                })
            except Exception:
                pass
        page = d.url
        what = d.name
        if d.relevance_reasoning:
            what += f" — {d.relevance_reasoning[:80]}"
        download = d.download_url or d.url
        score_color = "green" if d.relevance_score >= 0.8 else ("yellow" if d.relevance_score >= 0.5 else "red")
        console.print(
            f"[green]⭐[/green]  "
            f"[{score_color}][{d.relevance_score:.2f}][/{score_color}]  "
            f"[blue]{page}[/blue]  —  [bold]{what}[/bold]  —  [cyan]{download}[/cyan]"
        )

    def on_activity(ev: dict):
        # Mirror every event to the web dashboard.
        if actual_web_port is not None:
            try:
                web_ui.publish(ev)
            except Exception:
                pass
        elapsed = float(ev.get("elapsed", 0.0))
        iteration = int(ev.get("iteration", 0))
        prefix = _prefix(elapsed, iteration)
        kind = ev.get("type")
        name = ev.get("name", "")
        if kind == "tool_call":
            args = ev.get("args") or {}
            if name == "web_search":
                q = (args.get("query") or "").strip()
                console.print(f"{prefix} [magenta]🔎 search[/magenta]  {q}")
            elif name == "fetch_page":
                u = (args.get("url") or "").strip()
                console.print(f"{prefix} [cyan]📄 fetch[/cyan]   {u}")
            elif name == "fetch_page_js":
                u = (args.get("url") or "").strip()
                console.print(f"{prefix} [cyan]🌐 fetch_js[/cyan]  {u}")
            elif name == "read_pdf":
                u = (args.get("url") or "").strip()
                console.print(f"{prefix} [magenta]📑 read_pdf[/magenta]  {u}")
            elif name == "read_github_readme":
                u = (args.get("repo_url") or args.get("url") or "").strip()
                console.print(f"{prefix} [magenta]📘 readme[/magenta]   {u}")
            elif name == "arxiv_search":
                q = (args.get("query") or "").strip()
                console.print(f"{prefix} [magenta]📚 arxiv[/magenta]    {q}")
            elif name == "store_dataset":
                nm = (args.get("name") or "").strip()
                u = (args.get("url") or "").strip()
                score = args.get("relevance_score")
                score_str = f" [{score}]" if score is not None else ""
                console.print(f"{prefix} [yellow]💾 store[/yellow]{score_str}  {nm}  [dim]{u}[/dim]")
            elif name == "mark_search_complete":
                console.print(f"{prefix} [bold]✅ mark_complete[/bold]")
            else:
                console.print(f"{prefix} [white]· {name}[/white]")
        elif kind == "tool_result":
            status = ev.get("status", "rejected")
            msg = (ev.get("message") or "").strip()
            short = (msg[:140] + "…") if len(msg) > 140 else msg
            color = "red" if status == "rejected" else "yellow"
            icon = "⛔" if status == "rejected" else "⚠️"
            console.print(f"{prefix} [{color}]{icon} {name} {status}[/{color}]  [dim]{short}[/dim]")

    if interactive:
        console.print("[yellow]Interactive mode enabled. Press Enter after each iteration to continue.[/yellow]")

    try:
        result = agent.run(
            subject=subject,
            formats=formats,
            time_range=time_range,
            depth=depth,
            max_hours=max_hours,
            on_dataset_stored=on_stored,
            on_activity=on_activity,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
        return

    console.print()
    elapsed = result.elapsed_seconds or 0.0
    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)
    elapsed_str = f"{h}h {m}m {s}s" if h else (f"{m}m {s}s" if m else f"{s}s")

    console.print(
        f"[green]Search {result.status}.[/green] Stored "
        f"[bold]{result.datasets_found}[/bold] new datasets in this run "
        f"(query id [cyan]{result.id}[/cyan])."
    )
    console.print(
        f"[bold]Runtime:[/bold] {elapsed_str}   "
        f"[bold]Iterations:[/bold] {result.iterations}   "
        f"[bold]Existing in DB at start:[/bold] {result.existing_at_start}   "
        f"[bold]Skipped (already seen):[/bold] fetch={result.dedup_skipped_fetch}, store={result.dedup_skipped_store}"
    )
    console.print(f"[dim]Browse: [cyan]datasets_explorer results --query-id {result.id}[/cyan][/dim]")

    # Final summary table for everything ≥ min_relevance from this query
    datasets = storage.get_datasets(query_id=result.id, min_relevance=max(min_relevance, 0.5))
    if datasets:
        _render_summary_table(datasets)


def _render_summary_table(datasets: list[Dataset]) -> None:
    table = Table(box=box.ROUNDED, title="High-relevance findings", show_lines=True)
    table.add_column("Score", style="cyan", width=5, no_wrap=True)
    table.add_column("What it is", max_width=40, overflow="fold")
    table.add_column("Page", style="blue", overflow="fold", no_wrap=False)
    table.add_column("Download", style="cyan", overflow="fold", no_wrap=False)
    for d in datasets:
        table.add_row(
            f"{d.relevance_score:.2f}",
            d.name,
            d.url,
            d.download_url or d.url,
        )
    console.print(table)
    # Plain-text mirror of the URLs so the user can copy without ellipsis surprises.
    console.print("\n[dim]── full URLs (copyable) ──[/dim]")
    for d in datasets:
        console.print(f"[cyan]{d.relevance_score:.2f}[/cyan]  {d.url}")
        if d.download_url and d.download_url != d.url:
            console.print(f"      ↳ download: {d.download_url}")


if __name__ == "__main__":
    main()
