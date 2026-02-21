"""TECO CLI entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

app = typer.Typer(
    name="teco",
    help="TECO — The Experienced Code Optimizer. Agentic GPU kernel optimizer with learned experience.",
    add_completion=False,
)


@app.command()
def optimize(
    kernel: Annotated[Path, typer.Argument(help="Path to the kernel file to optimize")],
    language: Annotated[str, typer.Option("--language", "-l", help="Source language")] = "triton",
    model: Annotated[str, typer.Option("--model", "-m", help="Anthropic model ID")] = "claude-opus-4-6",
    knowledge_root: Annotated[
        Path, typer.Option("--knowledge-root", help="Knowledge store root directory")
    ] = Path("knowledge"),
    iterations: Annotated[
        int, typer.Option("--iterations", "-n", help="Maximum optimization iterations")
    ] = 10,
    target_efficiency: Annotated[
        float,
        typer.Option(
            "--target-efficiency",
            help="Stop when this %% of hardware ceiling is reached",
        ),
    ] = 90.0,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Optimize a GPU kernel file using agentic search and profiling."""
    if not kernel.exists():
        typer.echo(f"Error: kernel file not found: {kernel}", err=True)
        raise typer.Exit(1)

    typer.echo(f"TECO: optimizing {kernel} ({language}) with model {model}")
    typer.echo(f"  Knowledge root: {knowledge_root}")
    typer.echo(f"  Max iterations: {iterations}, target efficiency: {target_efficiency:.0f}%")

    from teco.orchestration.loop import run_optimization

    context = run_optimization(
        kernel_path=kernel,
        language=language,
        model_id=model,
        knowledge_root=knowledge_root,
        max_iterations=iterations,
        target_efficiency_pct=target_efficiency,
        verbose=verbose,
    )

    # Summary
    speedup = context._compute_overall_speedup() if hasattr(context, "_compute_overall_speedup") else 1.0
    typer.echo(f"\nDone. Run ID: {context.run_id}")
    typer.echo(f"Strategies deployed: {sum(1 for s in context.strategy_tree.strategies if s.status == 'regime_winner')}")


@app.command()
def query(
    hardware: Annotated[str, typer.Argument(help="Hardware ID (e.g. a100-sxm4-80gb)")],
    tags: Annotated[
        Optional[str],
        typer.Option("--tags", help="Comma-separated kernel characteristic tags"),
    ] = None,
    knowledge_root: Annotated[
        Path, typer.Option("--knowledge-root")
    ] = Path("knowledge"),
    top_k: Annotated[int, typer.Option("--top-k", "-k")] = 5,
) -> None:
    """Query the knowledge store for patterns matching a hardware target."""
    from teco.knowledge.schema import KernelCharacteristics
    from teco.knowledge.store import KnowledgeStore

    store = KnowledgeStore(knowledge_root=knowledge_root)
    chars = KernelCharacteristics(
        characteristic_tags=tags.split(",") if tags else [],
    )
    result = store.query(chars, hardware_id=hardware, max_results=top_k)

    if result.warnings:
        for w in result.warnings:
            typer.echo(f"WARNING: {w}", err=True)

    if not result.patterns:
        typer.echo("No patterns found.")
        return

    typer.echo(f"\nTop {len(result.patterns)} patterns for {hardware}:")
    for match in result.patterns:
        p = match.pattern
        typer.echo(
            f"  [{match.score:.2f}] {p.strategy_family} — "
            f"{p.effect.typical_speedup_range[0]:.1f}–{p.effect.typical_speedup_range[1]:.1f}x speedup, "
            f"confidence {p.confidence:.0%}"
        )

    if result.lessons:
        typer.echo(f"\nRelevant lessons ({len(result.lessons)}):")
        for lesson in result.lessons:
            typer.echo(f"  [{lesson.confidence:.0%}] {lesson.title}")


@app.command()
def rebuild_index(
    knowledge_root: Annotated[
        Path, typer.Option("--knowledge-root")
    ] = Path("knowledge"),
) -> None:
    """Rebuild the knowledge store index from scratch (use after manual edits)."""
    from teco.knowledge.store import KnowledgeStore

    store = KnowledgeStore(knowledge_root=knowledge_root)
    store.rebuild_index()
    typer.echo(f"Index rebuilt: {knowledge_root}/index.json")


if __name__ == "__main__":
    app()
