"""Interactive CLI and Testable Interface for Agent Memory Research Intelligence.

Provides an intuitive terminal interface for researchers and reviewers:
- Interactive proposal exploration
- Direct query input
- Standalone knowledge state inspection
- 15-case unseen evaluation benchmark runner
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

from src.models.schema import KnowledgeState, EntityType, RelationshipType
from src.graph.builder import AgentMemoryGraph
from src.reasoning.engine import ReasoningEngine, ResearchIntelligenceReport
from src.reasoning.evaluator import run_evaluation_suite

console = Console()

def load_engine() -> ReasoningEngine:
    """Load the pre-built knowledge state into the reasoning engine."""
    kg_path = PROJECT_ROOT / "knowledge" / "knowledge_state.json"
    if not kg_path.exists():
        console.print(
            "[bold red]Error:[/] Knowledge state not found at 'knowledge/knowledge_state.json'.\n"
            "Please run: [cyan]python scripts/build_knowledge.py[/] first.",
            style="red",
        )
        sys.exit(1)

    with open(kg_path, "r", encoding="utf-8") as f:
        k_data = json.load(f)

    graph = AgentMemoryGraph(KnowledgeState(**k_data))
    return ReasoningEngine(graph)

def render_report(report: ResearchIntelligenceReport):
    """Render structured research intelligence report using Rich."""
    console.print("\n")
    console.print(
        Panel(
            f"[bold white]{report.proposal_summary}[/]",
            title="[bold cyan]Input Research Proposal[/]",
            border_style="cyan",
        )
    )

    # 1. Concept Resolution
    memtypes = ", ".join(report.resolved_concepts.identified_memory_types)
    ops = ", ".join(report.resolved_concepts.identified_operations)
    keywords = ", ".join(report.resolved_concepts.salient_keywords)
    concept_text = (
        f"[bold cyan]Memory Paradigms:[/] [yellow]{memtypes}[/]\n"
        f"[bold cyan]Memory Operations:[/] [green]{ops}[/]\n"
        f"[bold cyan]Extracted Salient Tokens:[/] {keywords}"
    )
    console.print(Panel(concept_text, title="[bold green]1. Resolved Ontological Concepts[/]", border_style="green"))

    # 2. Prior Art Table
    art_table = Table(title="2. Closest Prior Art & Architectural Overlap", border_style="blue", show_header=True)
    art_table.add_column("Mechanism Name", style="bold white", width=26)
    art_table.add_column("Proposing Paper", style="cyan", width=34)
    art_table.add_column("Year", style="dim", width=6)
    art_table.add_column("Overlap", style="bold green", width=9)
    art_table.add_column("Shared Operations", style="magenta", width=22)

    for art in report.closest_prior_art:
        art_table.add_row(
            art.mechanism_name,
            art.proposing_paper_title[:32] + "...",
            str(art.paper_year),
            f"{art.overlap_score * 100:.0f}%",
            ", ".join(art.shared_operations),
        )
    console.print(art_table)

    # 3. Known Failure Modes & Pitfalls (with Verbatim Evidence Quotes)
    if report.known_pitfalls_and_failure_modes:
        pitfall_text = ""
        for p in report.known_pitfalls_and_failure_modes:
            pitfall_text += (
                f"[bold red]• {p.limitation_name}[/] (Category: [italic]{p.severity_category}[/])\n"
                f"  [dim]Exhibited by:[/] {p.exhibited_by_mechanism}\n"
                f"  [dim]Reported in:[/] [cyan]{p.reported_in_paper}[/] [[bold]{p.evidence_section}[/]]\n"
                f"  [yellow]\"{p.evidence_excerpt}\"[/]\n\n"
            )
        console.print(
            Panel(
                pitfall_text.strip(),
                title="[bold red]3. Historical Pitfalls & Verifiable Failure Modes[/]",
                border_style="red",
            )
        )

    # 4. Lineage and Counter-Approaches
    if report.architectural_lineage:
        lineage_table = Table(title="4. Architectural Lineage & Evolution Path", border_style="yellow")
        lineage_table.add_column("Base Mechanism", style="cyan")
        lineage_table.add_column("Relation", style="bold yellow")
        lineage_table.add_column("Successor Mechanism", style="green")
        lineage_table.add_column("Evolution Rationale", style="white")

        for step in report.architectural_lineage:
            lineage_table.add_row(
                step.earlier_mechanism,
                step.relationship,
                step.later_mechanism,
                step.rationale,
            )
        console.print(lineage_table)

    # 5. Recommended Evaluation Benchmarks
    if report.recommended_benchmarks:
        bm_table = Table(title="5. Recommended Evaluation Benchmarks & Task Suites", border_style="magenta")
        bm_table.add_column("Benchmark", style="bold cyan")
        bm_table.add_column("Target Domain", style="yellow")
        bm_table.add_column("Standard Evaluation Metrics", style="green")
        bm_table.add_column("Validation Papers", style="dim")

        for bm in report.recommended_benchmarks:
            bm_table.add_row(
                bm.benchmark_name,
                bm.domain,
                ", ".join(bm.metrics),
                ", ".join(bm.used_by_papers[:2]),
            )
        console.print(bm_table)

    # 6. Guided 5-Stage Reading Path
    reading_text = ""
    for stage in report.guided_reading_path:
        reading_text += (
            f"[bold cyan]{stage.stage_name}[/]: [bold white]{stage.paper_title}[/]\n"
            f"  [dim]Citation:[/] {stage.authors} ({stage.year}) [{stage.paper_id}]\n"
            f"  [italic green]Pedagogical Goal:[/] {stage.pedagogical_purpose}\n"
            f"  [bold yellow]Key Takeaway:[/] {stage.key_takeaway}\n\n"
        )
    console.print(
        Panel(
            reading_text.strip(),
            title="[bold magenta]6. Guided 5-Stage Pedagogical Reading Path[/]",
            border_style="magenta",
        )
    )
    console.print("\n")

def inspect_knowledge_state(engine: ReasoningEngine):
    """Print structural overview of the active knowledge state."""
    stats = engine.graph_wrapper.summary_stats()
    table = Table(title="Active Knowledge State Structural Invariants", border_style="cyan")
    table.add_column("Component / Entity / Edge Type", style="bold white")
    table.add_column("Verified In-Graph Count", style="bold green", justify="right")

    for k, v in stats.items():
        table.add_row(k, str(v))
    console.print(table)

def main():
    parser = argparse.ArgumentParser(
        description="Agent Memory Research Intelligence & Decision Reasoning System"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Input research proposal to reason over",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Inspect active knowledge state statistics and structural counts",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run the 15-case unseen evaluation benchmark suite",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        help="Optional path to export structured report JSON",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch interactive terminal exploration mode",
    )

    args = parser.parse_args()
    engine = load_engine()

    if args.inspect:
        inspect_knowledge_state(engine)
        return

    if args.eval:
        console.print("[bold cyan]Running 15-case unseen evaluation benchmark...[/]")
        metrics, results = run_evaluation_suite(engine)
        eval_table = Table(title="Unseen Research Proposal Evaluation Scorecard", border_style="green")
        eval_table.add_column("Evaluation Metric", style="bold white")
        eval_table.add_column("Performance Score", style="bold green", justify="right")

        eval_table.add_row("Total Unseen Evaluation Proposals", str(metrics.total_test_cases))
        eval_table.add_row("Concept Resolution Precision", f"{metrics.concept_resolution_precision * 100:.1f}%")
        eval_table.add_row("Prior Art Recall@1", f"{metrics.prior_art_recall_at_1 * 100:.1f}%")
        eval_table.add_row("Prior Art Recall@3", f"{metrics.prior_art_recall_at_3 * 100:.1f}%")
        eval_table.add_row("Pitfall Identification Rate", f"{metrics.pitfall_detection_rate * 100:.1f}%")
        eval_table.add_row("Benchmark Alignment Rate", f"{metrics.benchmark_alignment_rate * 100:.1f}%")
        eval_table.add_row("Reading Path Completeness", f"{metrics.reading_path_completeness * 100:.1f}%")
        console.print(eval_table)
        return

    query = args.query
    if not query:
        # Default to interactive prompt
        console.print(
            Panel(
                "[bold cyan]Agent Memory Research Intelligence System[/]\n"
                "[dim]Enter an unseen research proposal or memory architecture idea to reason over the verified knowledge graph.[/]\n"
                "[italic]Type 'exit' or 'quit' to exit.[/]",
                border_style="cyan",
            )
        )
        try:
            query = console.input("\n[bold yellow]Enter research idea / proposal: [/]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Exiting...[/]")
            return

        if not query or query.lower() in ("exit", "quit"):
            return

    report = engine.reason(query)
    render_report(report)

    if args.json_out:
        out_p = Path(args.json_out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(report.model_dump(), f, indent=2)
        console.print(f"[bold green]Report exported successfully to:[/] {out_p}")

if __name__ == "__main__":
    main()
