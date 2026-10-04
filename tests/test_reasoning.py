"""Unit tests for the Online Reasoning Engine and Concept Resolver."""

import json
from pathlib import Path
from src.models.schema import KnowledgeState
from src.graph.builder import AgentMemoryGraph
from src.reasoning.concept_resolver import ConceptResolver
from src.reasoning.engine import ReasoningEngine

def test_concept_resolution_basic():
    resolver = ConceptResolver()
    query = "I want to design an agent that logs past interaction trajectories into episodic memory and retrieves them."
    concepts = resolver.resolve(query)

    assert "memtype_episodic" in concepts.identified_memory_types
    assert "op_store" in concepts.identified_operations or "op_retrieve" in concepts.identified_operations

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

    # Verify pitfall warnings with evidence
    assert len(report.known_pitfalls_and_failure_modes) > 0
    top_pitfall = report.known_pitfalls_and_failure_modes[0]
    assert len(top_pitfall.evidence_excerpt) > 0

    # Verify 5-stage reading path
    assert len(report.guided_reading_path) == 5
    assert report.guided_reading_path[0].stage_number == 1
    assert report.guided_reading_path[4].stage_number == 5
