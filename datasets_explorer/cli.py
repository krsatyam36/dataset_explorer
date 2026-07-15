import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, TextColumn
from rich.panel import Panel
from rich import box

from .agent import DatasetDiscoveryAgent
from .storage import Storage
from .models import Dataset
from .config import OLLAMA_MODEL, OLLAMA_HOST, DB_PATH

console = Console()


@click.group()
def cli():
    """datasets_explorer — autonomous dataset discovery for AI/ML teams.

    \b
    Examples:
      datasets_explorer search "military aircraft imagery GeoTIFF 2020-2024"
      datasets_explorer search "SAR satellite imagery annotated" --hours 3
      datasets_explorer results --query-id 1 --min-relevance 0.6
      datasets_explorer queries
      datasets_explorer annotate 5 "Priority — check license before use"
    """
    pass


@cli.command()
@click.argument("query")
@click.option("--hours", default=2.0, show_default=True, help="Max search duration in hours")
@click.option(
    "--min-relevance",
    default=0.5,
    show_default=True,
    help="Minimum relevance score to display in summary (0.0-1.0)",
)
@click.option(
    "--model",
    default=None,
    help=f"Ollama model to use (default: {OLLAMA_MODEL})",
)
def search(query: str, hours: float, min_relevance: float, model: str):
    """Search for datasets matching QUERY.

    QUERY should describe the kind of dataset you need, including subject matter,
    format preferences, and time period if relevant.

    \b
    Examples:
      datasets_explorer search "military aircraft imagery GeoTIFF 2020-2024"
      datasets_explorer search "satellite SAR imagery ship detection annotated" --hours 3
      datasets_explorer search "aerial orthoimagery numpy arrays labeled" --hours 1
    """
    effective_model = model or OLLAMA_MODEL
    storage = Storage(DB_PATH)
    agent = DatasetDiscoveryAgent(storage, model=effective_model)

    console.print()
    console.print(Panel(
        f"[bold]Query:[/bold] [yellow]{query}[/yellow]\n"
        f"[bold]Model:[/bold] {effective_model} (via Ollama @ {OLLAMA_HOST})\n"
        f"[bold]Max duration:[/bold] {hours}h\n"
        f"[bold]Database:[/bold] {DB_PATH}",
        title="[bold cyan]datasets_explorer[/bold cyan]",
    ))
    console.print("[dim]Press Ctrl+C at any time to stop and save results so far.[/dim]\n")

    result_query = None
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Searching the internet for datasets...", total=None)
        result_query = agent.run(subject=query, formats=[], time_range="", depth=2, max_hours=hours)
        progress.update(task, description=f"[green]Search {result_query.status}[/green]")

    console.print()
    console.print(
        f"[green]Search {result_query.status}.[/green] "
        f"Found [bold]{result_query.datasets_found}[/bold] datasets "
        f"(query id: [cyan]{result_query.id}[/cyan])"
    )
    console.print(
        f"[dim]Run [cyan]datasets_explorer results --query-id {result_query.id}[/cyan] to browse all results.[/dim]\n"
    )

    datasets = storage.get_datasets(query_id=result_query.id, min_relevance=min_relevance)
    if datasets:
        console.print(f"[bold]Top results (relevance ≥ {min_relevance}):[/bold]")
        _render_dataset_table(datasets)
    else:
        console.print(
            f"[yellow]No results with relevance ≥ {min_relevance}. "
            f"Try: datasets_explorer results --query-id {result_query.id} --min-relevance 0.0[/yellow]"
        )


@cli.command()
@click.option("--query-id", type=int, default=None, help="Filter by search query ID")
@click.option("--min-relevance", default=0.0, show_default=True, help="Minimum relevance score")
@click.option("--format", "fmt", default=None, help="Filter by format (geotiff, jpeg, npy, ...)")
@click.option("--limit", default=50, show_default=True, help="Max results to show")
@click.option("--unreviewed", is_flag=True, help="Show only unreviewed datasets")
def results(query_id, min_relevance, fmt, limit, unreviewed):
    """Browse previously found datasets."""
    storage = Storage(DB_PATH)
    formats = [fmt] if fmt else None
    reviewed = False if unreviewed else None
    datasets = storage.get_datasets(
        query_id=query_id,
        min_relevance=min_relevance,
        formats=formats,
        reviewed=reviewed,
        limit=limit,
    )
    if not datasets:
        console.print("[yellow]No datasets found matching the filters.[/yellow]")
        return
    _render_dataset_table(datasets)


@cli.command()
def queries():
    """List all past search sessions."""
    storage = Storage(DB_PATH)
    qs = storage.list_queries()
    if not qs:
        console.print("[yellow]No search sessions found.[/yellow]")
        return

    table = Table(title="Search Sessions", box=box.ROUNDED)
    table.add_column("ID", style="dim", width=4)
    table.add_column("Query", max_width=50)
    table.add_column("Status", width=12)
    table.add_column("Found", justify="right", width=6)
    table.add_column("Started")

    for q in qs:
        status_style = {
            "completed": "green",
            "running": "yellow",
            "interrupted": "red",
        }.get(q.status, "white")
        table.add_row(
            str(q.id),
            q.raw_query,
            f"[{status_style}]{q.status}[/{status_style}]",
            str(q.datasets_found),
            str(q.started_at)[:16],
        )
    console.print(table)


@cli.command()
@click.option("--interval", default=5, help="Poll interval in seconds", show_default=True)
@click.option("--query-id", type=int, default=None, help="Watch a specific query only")
def watch(interval, query_id):
    """Watch for new datasets arriving in real-time."""
    import time as _time
    from datetime import datetime
    storage = Storage(DB_PATH)
    seen = {d.id for d in storage.get_datasets(query_id=query_id, limit=10_000)}
    console.print(f"[green]Watching for new datasets (every {interval}s)...[/green]")
    try:
        while True:
            _time.sleep(interval)
            fresh = [d for d in storage.get_datasets(query_id=query_id, limit=10_000) if d.id not in seen]
            for d in fresh:
                ts = d.discovered_at.strftime("%H:%M:%S") if d.discovered_at else ""
                console.print(f"  [{ts}] [#{d.id}] {d.name}  [dim]({d.source.value})[/dim]")
                seen.add(d.id)
            if fresh:
                console.print(f"  [dim]— {len(fresh)} new —[/dim]")
    except KeyboardInterrupt:
        console.print("\nStopped.")

@cli.command()
@click.argument("dataset_id", type=int)
@click.argument("notes")
def annotate(dataset_id: int, notes: str):
    """Add notes to a dataset and mark it as reviewed.

    \b
    Example:
      datasets_explorer annotate 5 "Priority — large scale, check license"
    """
    storage = Storage(DB_PATH)
    storage.update_dataset_notes(dataset_id, notes)
    storage.mark_reviewed(dataset_id)
    console.print(f"[green]Dataset {dataset_id} updated and marked as reviewed.[/green]")


@cli.command()
@click.argument("dataset_id", type=int)
def show(dataset_id: int):
    """Show full details for a single dataset."""
    storage = Storage(DB_PATH)
    datasets = storage.get_datasets(limit=1000)
    match = next((d for d in datasets if d.id == dataset_id), None)
    if not match:
        console.print(f"[red]Dataset {dataset_id} not found.[/red]")
        return
    _render_dataset_detail(match)


def _render_dataset_table(datasets: list[Dataset]) -> None:
    table = Table(box=box.ROUNDED, show_lines=True)
    table.add_column("ID", style="dim", width=4)
    table.add_column("Score", style="cyan", width=6)
    table.add_column("Name", max_width=35)
    table.add_column("Source", width=14)
    table.add_column("Formats", width=18)
    table.add_column("License", width=14)
    table.add_column("Samples", justify="right", width=8)
    table.add_column("URL", max_width=40, style="blue")

    for d in datasets:
        score_color = "green" if d.relevance_score >= 0.8 else ("yellow" if d.relevance_score >= 0.5 else "red")
        table.add_row(
            str(d.id),
            f"[{score_color}]{d.relevance_score:.2f}[/{score_color}]",
            d.name,
            d.source if isinstance(d.source, str) else d.source.value,
            ", ".join(d.formats[:3]),
            d.license or "?",
            str(d.num_samples) if d.num_samples else "?",
            d.url,
        )
    console.print(table)
    console.print(f"[dim]{len(datasets)} dataset(s) shown.[/dim]")


def _render_dataset_detail(d: Dataset) -> None:
    score_color = "green" if d.relevance_score >= 0.8 else ("yellow" if d.relevance_score >= 0.5 else "red")
    details = (
        f"[bold]Name:[/bold] {d.name}\n"
        f"[bold]URL:[/bold] [blue]{d.url}[/blue]\n"
        f"[bold]Source:[/bold] {d.source}\n"
        f"[bold]Formats:[/bold] {', '.join(d.formats) or 'unknown'}\n"
        f"[bold]License:[/bold] {d.license or 'unknown'}\n"
        f"[bold]Size:[/bold] {d.size_human or 'unknown'}\n"
        f"[bold]Samples:[/bold] {d.num_samples or 'unknown'}\n"
        f"[bold]Date range:[/bold] {d.date_range_start or '?'} – {d.date_range_end or '?'}\n"
        f"[bold]Tags:[/bold] {', '.join(d.tags) or 'none'}\n"
        f"[bold]Relevance:[/bold] [{score_color}]{d.relevance_score:.2f}[/{score_color}] — {d.relevance_reasoning}\n"
        f"[bold]Reviewed:[/bold] {'yes' if d.reviewed else 'no'}\n"
        f"[bold]Notes:[/bold] {d.notes or '(none)'}\n\n"
        f"[bold]Description:[/bold]\n{d.description or '(none)'}"
    )
    console.print(Panel(details, title=f"Dataset #{d.id}"))
# feat/cli-watch: Refine cli-watch CLI output
# feat/cli-watch: Add cli-watch error messages
# feat/cli-watch: Add cli-watch help examples
