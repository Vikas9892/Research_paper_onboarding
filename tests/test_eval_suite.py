"""Unit tests validating the 15-case unseen evaluation benchmark suite."""

import json
from pathlib import Path
from src.models.schema import KnowledgeState
from src.graph.builder import AgentMemoryGraph
from src.reasoning.engine import ReasoningEngine
from src.reasoning.evaluator import run_evaluation_suite

def test_evaluation_benchmark_suite_metrics():
    root = Path(__file__).resolve().parent.parent
    kg_path = root / "knowledge" / "knowledge_state.json"
    with open(kg_path, "r", encoding="utf-8") as f:
        k_data = json.load(f)

    graph = AgentMemoryGraph(KnowledgeState(**k_data))
    engine = ReasoningEngine(graph)

    metrics, results = run_evaluation_suite(engine)

    assert metrics.total_test_cases == 15
    assert metrics.concept_resolution_precision >= 0.80
    assert metrics.prior_art_recall_at_3 >= 0.80
    assert metrics.benchmark_alignment_rate >= 0.90
    assert metrics.reading_path_completeness == 1.00
    assert len(results) == 15
