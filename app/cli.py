"""Interactive CLI and Testable Interface for Agent Memory Research Intelligence.

Renders the complete 14-step evidence-grounded research architecture report.
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

from src.models.schema import KnowledgeState
from src.graph.builder import AgentMemoryGraph
from src.reasoning.engine import ReasoningEngine, ComprehensiveResearchReport
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

def render_report(report: ComprehensiveResearchReport):
    """Render the full 14-step evidence-grounded research report."""
    console.print("\n")

    # 1. Proposal Understanding
    understanding_text = (
        f"[bold white]Summary:[/] {report.proposal_summary}\n\n"
        f"[bold cyan]Stored Information:[/] {report.stored_information}\n"
        f"[bold cyan]Retrieval Trigger:[/] {report.retrieval_trigger}\n"
        f"[bold cyan]Update Trigger:[/] {report.update_trigger}\n"
        f"[bold cyan]Intended Benefit:[/] {report.intended_benefit}"
    )
    console.print(Panel(understanding_text, title="[bold cyan]1. Proposal Understanding[/]", border_style="cyan"))

    # 2. Resolved Concepts Table
    c_table = Table(title="2. Resolved Ontological Concepts", border_style="green", show_header=True)
    c_table.add_column("Proposal Concept", style="bold white", width=26)
    c_table.add_column("Resolved Entity", style="yellow", width=28)
    c_table.add_column("Type", style="cyan", width=18)
    c_table.add_column("Confidence", style="bold green", justify="right", width=12)
    c_table.add_column("Evidence Grounding", style="dim", width=36)

    for item in report.resolved_concepts:
        c_table.add_row(
            item.input_concept,
            item.resolved_entity_name,
            item.entity_type,
            f"{item.confidence * 100:.0f}%",
            item.evidence,
        )
    console.print(c_table)

    # 3. Architectural Fingerprint
    fp = report.architectural_fingerprint
    fp_text = (
        f"Memory Types      : {', '.join(fp.memory_types)}\n"
        f"Mechanisms        : {', '.join(fp.mechanisms)}\n"
        f"Operations        : {', '.join(fp.operations)}\n"
        f"Stored Information: {fp.stored_information}\n"
        f"Retrieval Pattern : {fp.retrieval_pattern}\n"
        f"Update Pattern    : {fp.update_pattern}\n"
        f"Evaluation Tasks  : {', '.join(fp.evaluation_tasks)}"
    )
    console.print(Panel(fp_text, title="[bold yellow]3. Architectural Fingerprint[/]", border_style="yellow"))

    # 4. Closest Prior Art
    art_table = Table(title="4. Closest Prior Art (Reproducible Overlap Scoring)", border_style="blue", show_header=True)
    art_table.add_column("Paper & Mechanism", style="bold white", width=32)
    art_table.add_column("Year", style="dim", width=6)
    art_table.add_column("Overlap", style="bold green", width=9)
    art_table.add_column("Why It Matches", style="white", width=35)
    art_table.add_column("Evidence Excerpt", style="dim", width=42)

    for art in report.closest_prior_art:
        art_table.add_row(
            f"{art.paper_title}\n[cyan]({art.mechanism_name})[/]",
            str(art.year),
            f"{art.overlap_score * 100:.0f}%",
            art.why_it_matches,
            f"\"{art.evidence[:85]}...\"" if len(art.evidence) > 85 else f"\"{art.evidence}\"",
        )
    console.print(art_table)

    # 5. Historical Failure Modes
    fail_table = Table(title="5. Historical Failure Modes & Design Implications", border_style="red", show_header=True)
    fail_table.add_column("Failure Mode", style="bold red", width=22)
    fail_table.add_column("Affected Architecture", style="cyan", width=24)
    fail_table.add_column("Why It Affects Proposal", style="white", width=35)
    fail_table.add_column("Design Implication", style="yellow", width=35)
    fail_table.add_column("Strength", style="bold magenta", width=12)

    for f in report.historical_failure_modes:
        fail_table.add_row(
            f.failure_mode,
            f.affected_architecture,
            f.why_it_affects_proposal,
            f.design_implication,
            f.evidence_strength,
        )
    console.print(fail_table)

    # 6. Historical Lineage
    lineage_table = Table(title="6. Historical Research Lineage", border_style="yellow")
    lineage_table.add_column("Source Architecture", style="cyan")
    lineage_table.add_column("Relationship", style="bold yellow")
    lineage_table.add_column("Target Architecture", style="green")
    lineage_table.add_column("Evolutionary Rationale", style="white")

    for step in report.historical_lineage:
        lineage_table.add_row(
            step.source_mechanism,
            step.relationship,
            step.target_mechanism,
            step.transition_rationale,
        )
    console.print(lineage_table)

    # 7. Benchmark Recommendations
    bm_table = Table(title="7. Capability-Driven Benchmark Recommendations", border_style="magenta")
    bm_table.add_column("Benchmark", style="bold cyan", width=18)
    bm_table.add_column("Task Domain", style="yellow", width=20)
    bm_table.add_column("Required Capability", style="white", width=32)
    bm_table.add_column("Metric", style="bold green", width=20)
    bm_table.add_column("Evidence Citation", style="dim", width=32)

    for bm in report.benchmark_recommendations:
        bm_table.add_row(
            bm.benchmark_name,
            bm.task,
            bm.required_capability,
            bm.metric,
            bm.evidence[:60] + "...",
        )
    console.print(bm_table)

    # 8. Research Gaps
    gaps_text = "[bold green]Verified Literature Gaps:[reset]\n"
    for vg in report.verified_gaps:
        gaps_text += f"• {vg}\n"
    gaps_text += "\n[bold yellow]Inferred Gaps:[reset]\n"
    for ig in report.inferred_gaps:
        gaps_text += f"• {ig}\n"
    gaps_text += "\n[dim]Unknown / Insufficient Knowledge-State Coverage:[reset]\n"
    for ug in report.unknown_evidence:
        gaps_text += f"• {ug}\n"
    console.print(Panel(gaps_text.strip(), title="[bold white]8. Research Gaps Analysis[/]", border_style="white"))

    # 9. Novelty / Overlap Assessment
    novelty_text = (
        "[bold cyan]Established Prior Art Aspects:[reset]\n"
        + "\n".join([f"• {x}" for x in report.established_aspects])
        + "\n\n[bold yellow]Strong Overlap Aspects:[reset]\n"
        + "\n".join([f"• {x}" for x in report.strong_overlap_aspects])
        + "\n\n[bold green]Evolutionary Extension Aspects:[reset]\n"
        + "\n".join([f"• {x}" for x in report.extension_aspects])
        + "\n\n[bold magenta]Potentially Novel Combination:[reset]\n"
        + "\n".join([f"• {x}" for x in report.potentially_novel_combinations])
        + "\n\n[bold red]Unsupported Novelty Claims:[reset]\n"
        + "\n".join([f"• {x}" for x in report.unsupported_claims])
    )
    console.print(Panel(novelty_text, title="[bold magenta]9. Research Novelty & Overlap Assessment[/]", border_style="magenta"))

    # 10. Five-Stage Reading Path
    reading_text = ""
    for stage in report.five_stage_reading_path:
        reading_text += (
            f"[bold cyan]{stage.stage_name}:[/] [bold white]{stage.paper_title}[/]\n"
            f"  [dim]Authors & Year:[/] {stage.authors} ({stage.year}) [{stage.paper_id}]\n"
            f"  [yellow]Why Read Now:[/] {stage.why_read_now}\n"
            f"  [green]What to Learn:[/] {stage.what_to_learn}\n"
            f"  [dim]Connection to Previous:[/] {stage.connection_to_previous_stage}\n"
            f"  [bold cyan]Connection to Proposal:[/] {stage.connection_to_proposal}\n\n"
        )
    console.print(Panel(reading_text.strip(), title="[bold cyan]10. Five-Stage Pedagogical Reading Path[/]", border_style="cyan"))

    # 11. Evidence Audit
    audit = report.evidence_audit
    audit_text = (
        f"Total Major Claims Evaluated: [bold white]{audit.total_major_claims}[/]\n"
        f"Explicitly Evidence-Backed  : [bold green]{audit.evidence_backed}[/]\n"
        f"Inferred (Clearly Labeled)  : [bold yellow]{audit.inferred}[/]\n"
        f"Unsupported Claims          : [bold red]{audit.unsupported}[/]\n"
        f"Total Evidence Coverage     : [bold green]{audit.evidence_coverage * 100:.1f}%[/]"
    )
    console.print(Panel(audit_text, title="[bold green]11. Evidence Grounding Audit[/]", border_style="green"))

    # 12. Integrity Warnings
    if report.integrity_warnings:
        warn_text = "\n".join([f"• {w}" for w in report.integrity_warnings])
        console.print(Panel(warn_text, title="[bold red]12. Integrity Warnings[/]", border_style="red"))
    else:
        console.print(Panel("[bold green]Zero integrity violations detected. All concepts, relations, and benchmarks internally consistent.[/]", title="[bold green]12. Integrity Verification[/]", border_style="green"))

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
