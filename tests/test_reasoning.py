"""Unit tests for the Online Reasoning Engine and Proposal Decomposer."""

import json
from pathlib import Path
from src.models.schema import KnowledgeState
from src.graph.builder import AgentMemoryGraph
from src.reasoning.concept_resolver import ProposalDecomposer
from src.reasoning.engine import ReasoningEngine

def test_concept_resolution_basic():
    decomposer = ProposalDecomposer()
    query = "I want to design an agent that logs past interaction trajectories into episodic memory and retrieves them."
    decomp = decomposer.decompose(query)

    types = [item.resolved_entity_id for item in decomp.resolved_items if item.entity_type == "MemoryType"]
    ops = [item.resolved_entity_id for item in decomp.resolved_items if item.entity_type == "MemoryOperation"]

    assert "memtype_episodic" in types
    assert "op_store" in ops or "op_retrieve" in ops

def test_reasoning_engine_produces_structured_report():
    root = Path(__file__).resolve().parent.parent
    kg_path = root / "knowledge" / "knowledge_state.json"
    with open(kg_path, "r", encoding="utf-8") as f:
        k_data = json.load(f)

    graph = AgentMemoryGraph(KnowledgeState(**k_data))
    engine = ReasoningEngine(graph)

    query = "I want to build an operating system memory hierarchy for LLM agents that pages context between working memory and disk archival storage."
    report = engine.reason(query)

    assert report.proposal_summary == query
    assert len(report.closest_prior_art) > 0
    top_art = report.closest_prior_art[0]
    assert "paging" in top_art.mechanism_id.lower() or "context" in top_art.mechanism_id.lower()

    # Verify failure modes with evidence
    assert len(report.historical_failure_modes) > 0
    top_pitfall = report.historical_failure_modes[0]
    assert len(top_pitfall.evidence) > 0

    # Verify 5-stage reading path
    assert len(report.five_stage_reading_path) == 5
    assert report.five_stage_reading_path[0].stage_number == 1
    assert report.five_stage_reading_path[4].stage_number == 5

    # Verify Evidence Audit
    assert report.evidence_audit.total_major_claims > 0
    assert report.evidence_audit.evidence_coverage > 0.70
