"""Unseen Research Proposal Evaluation Suite (15 Test Cases).

Measures concept resolution precision, prior art retrieval recall,
pitfall identification, and structural output validity across diverse,
unseen agent memory proposals.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.schema import KnowledgeState
from src.graph.builder import AgentMemoryGraph
from src.reasoning.engine import ReasoningEngine, ComprehensiveResearchReport

class EvalTestCase(BaseModel):
    case_id: str
    proposal_text: str
    expected_memory_types: List[str]
    expected_primary_prior_art: str
    expected_limitation_warning: str
    expected_benchmark: str

UNSEEN_EVAL_CASES: List[EvalTestCase] = [
    EvalTestCase(
        case_id="case_01_reflection_buffer",
        proposal_text="I want an agent that logs verbal self-critiques after failing tasks in simulated household environments and retrieves past lessons before taking new actions.",
        expected_memory_types=["memtype_reflective"],
        expected_primary_prior_art="mech_verbal_reflection_buffer",
        expected_limitation_warning="limit_reflection_loops",
        expected_benchmark="bm_alfworld",
    ),
    EvalTestCase(
        case_id="case_02_context_paging",
        proposal_text="I want to build an operating system memory manager for LLMs that pages memory blocks between active context window RAM and disk storage to handle infinite sessions.",
        expected_memory_types=["memtype_working"],
        expected_primary_prior_art="mech_hierarchical_context_paging",
        expected_limitation_warning="limit_inference_latency",
        expected_benchmark="bm_hotpotqa",
    ),
    EvalTestCase(
        case_id="case_03_skill_library",
        proposal_text="I propose an embodied agent that writes executable JavaScript programs for completed tasks and saves them into an indexed skill library for open-ended exploration in Minecraft.",
        expected_memory_types=["memtype_procedural"],
        expected_primary_prior_art="mech_vectorized_skill_library",
        expected_limitation_warning="limit_skill_proliferation",
        expected_benchmark="bm_minecraft",
    ),
    EvalTestCase(
        case_id="case_04_hippocampal_graph",
        proposal_text="I want to build an associative memory network that builds an open knowledge graph from documents and executes Personalized PageRank over graph entities for multi-hop QA.",
        expected_memory_types=["memtype_semantic"],
        expected_primary_prior_art="mech_hippocampal_pagerank_memory",
        expected_limitation_warning="limit_relational_gap",
        expected_benchmark="bm_hotpotqa",
    ),
    EvalTestCase(
        case_id="case_05_forgetting_curve",
        proposal_text="I want an episodic companion agent that decays memory strength over time using the Ebbinghaus forgetting curve, pruning stale memories that haven't been accessed recently.",
        expected_memory_types=["memtype_episodic"],
        expected_primary_prior_art="mech_ebbinghaus_decay_memory",
        expected_limitation_warning="limit_spurious_rules",
        expected_benchmark="bm_hotpotqa",
    ),
    EvalTestCase(
        case_id="case_06_policy_reflection",
        proposal_text="I want to optimize self-reflection text generation using policy gradient reinforcement learning directly on downstream task completion rewards.",
        expected_memory_types=["memtype_reflective"],
        expected_primary_prior_art="mech_policy_gradient_reflection",
        expected_limitation_warning="limit_inference_latency",
        expected_benchmark="bm_alfworld",
    ),
    EvalTestCase(
        case_id="case_07_tree_search",
        proposal_text="I want a tree-structured working memory where the language model explores branching paths of thoughts using breadth-first search and backtracks upon error.",
        expected_memory_types=["memtype_working"],
        expected_primary_prior_art="mech_tree_search_thought_buffer",
        expected_limitation_warning="limit_inference_latency",
        expected_benchmark="bm_hotpotqa",
    ),
    EvalTestCase(
        case_id="case_08_mcts_reflection",
        proposal_text="I want an agent framework that unifies Monte Carlo Tree Search rollouts with reflective episodic memory to backpropagate failure critiques up decision trees.",
        expected_memory_types=["memtype_reflective"],
        expected_primary_prior_art="mech_monte_carlo_tree_memory",
        expected_limitation_warning="limit_inference_latency",
        expected_benchmark="bm_hotpotqa",
    ),
    EvalTestCase(
        case_id="case_09_code_editor_memory",
        proposal_text="I want an autonomous software engineering agent that maintains structured memory of open file windows, search caches, and execution histories to solve GitHub repository issues.",
        expected_memory_types=["memtype_procedural"],
        expected_primary_prior_art="mech_aci_windowed_memory",
        expected_limitation_warning="limit_inference_latency",
        expected_benchmark="bm_swe_bench",
    ),
    EvalTestCase(
        case_id="case_10_relational_triplets",
        proposal_text="I want an agent that extracts factual entity-relation triplets into an explicit relational knowledge graph to avoid hallucinated memory during web navigation.",
        expected_memory_types=["memtype_semantic"],
        expected_primary_prior_art="mech_relational_triplet_memory",
        expected_limitation_warning="limit_relational_gap",
        expected_benchmark="bm_webarena",
    ),
    EvalTestCase(
        case_id="case_11_cross_task_rules",
        proposal_text="I want an experienced learning agent that gathers pairs of successful and failed interaction trajectories across domains and extracts reusable operational rule pools.",
        expected_memory_types=["memtype_episodic"],
        expected_primary_prior_art="mech_cross_task_rule_pool",
        expected_limitation_warning="limit_spurious_rules",
        expected_benchmark="bm_alfworld",
    ),
    EvalTestCase(
        case_id="case_12_transactional_paging",
        proposal_text="I propose adding ACID database transaction guarantees to context paging in agent runtimes to roll back memory state when external tools fail.",
        expected_memory_types=["memtype_working"],
        expected_primary_prior_art="mech_transactional_context_paging",
        expected_limitation_warning="limit_inference_latency",
        expected_benchmark="bm_swe_bench",
    ),
    EvalTestCase(
        case_id="case_13_memory_stream",
        proposal_text="I want to build a sandbox society of autonomous agents where each agent records a timestamped memory stream of observations and periodically summarizes them into abstract reflections.",
        expected_memory_types=["memtype_episodic"],
        expected_primary_prior_art="mech_stream_reflection_architecture",
        expected_limitation_warning="limit_memory_drift",
        expected_benchmark="bm_alfworld",
    ),
    EvalTestCase(
        case_id="case_14_hybrid_vector_graph",
        proposal_text="I propose a production memory layer that combines dense vector embeddings for semantic user preference retrieval with a graph database to resolve contradictory statements.",
        expected_memory_types=["memtype_semantic"],
        expected_primary_prior_art="mech_hybrid_keyvalue_graph_memory",
        expected_limitation_warning="limit_spurious_rules",
        expected_benchmark="bm_hotpotqa",
    ),
    EvalTestCase(
        case_id="case_15_poisoning_defense",
        proposal_text="I want to study security vulnerabilities where untrusted web content gets injected into long-term memory and triggers delayed prompt injection during subsequent retrieval.",
        expected_memory_types=["memtype_episodic"],
        expected_primary_prior_art="mech_verbal_reflection_buffer",
        expected_limitation_warning="limit_prompt_injection",
        expected_benchmark="bm_webarena",
    ),
]

class EvaluationMetrics(BaseModel):
    total_test_cases: int
    concept_resolution_precision: float
    prior_art_recall_at_1: float
    prior_art_recall_at_3: float
    pitfall_detection_rate: float
    benchmark_alignment_rate: float
    reading_path_completeness: float

def run_evaluation_suite(engine: ReasoningEngine) -> Tuple[EvaluationMetrics, List[Dict[str, Any]]]:
    """Execute all 15 unseen test cases and compute quantitative performance metrics."""
    correct_concept = 0
    recall_at_1 = 0
    recall_at_3 = 0
    pitfall_hits = 0
    benchmark_hits = 0
    reading_path_complete = 0

    detailed_results = []

    for test in UNSEEN_EVAL_CASES:
        report = engine.reason(test.proposal_text)

        # Concept resolution check
        c_hit = any(item.resolved_entity_id in test.expected_memory_types for item in report.resolved_concepts)
        if c_hit:
            correct_concept += 1

        # Prior art recall check
        top_mech_ids = [m.mechanism_id for m in report.closest_prior_art]
        r1 = len(top_mech_ids) > 0 and top_mech_ids[0] == test.expected_primary_prior_art
        r3 = test.expected_primary_prior_art in top_mech_ids[:3]
        if r1:
            recall_at_1 += 1
        if r3:
            recall_at_3 += 1

        # Pitfall check
        pitfall_ids = [p.limitation_id for p in report.historical_failure_modes]
        p_hit = test.expected_limitation_warning in pitfall_ids
        if p_hit:
            pitfall_hits += 1

        # Benchmark check
        bm_ids = [b.benchmark_id for b in report.benchmark_recommendations]
        b_hit = test.expected_benchmark in bm_ids
        if b_hit:
            benchmark_hits += 1

        # Reading path check: Must have all 5 stages populated
        rp_hit = len(report.five_stage_reading_path) == 5
        if rp_hit:
            reading_path_complete += 1

        detailed_results.append({
            "case_id": test.case_id,
            "concept_match": c_hit,
            "recall@1": r1,
            "recall@3": r3,
            "pitfall_match": p_hit,
            "benchmark_match": b_hit,
            "reading_path_valid": rp_hit,
            "top_match": top_mech_ids[0] if top_mech_ids else None,
        })

    n = len(UNSEEN_EVAL_CASES)
    metrics = EvaluationMetrics(
        total_test_cases=n,
        concept_resolution_precision=round(correct_concept / n, 3),
        prior_art_recall_at_1=round(recall_at_1 / n, 3),
        prior_art_recall_at_3=round(recall_at_3 / n, 3),
        pitfall_detection_rate=round(pitfall_hits / n, 3),
        benchmark_alignment_rate=round(benchmark_hits / n, 3),
        reading_path_completeness=round(reading_path_complete / n, 3),
    )

    return metrics, detailed_results

if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent.parent
    kg_path = root / "knowledge" / "knowledge_state.json"
    with open(kg_path, "r", encoding="utf-8") as f:
        k_data = json.load(f)
    graph = AgentMemoryGraph(KnowledgeState(**k_data))
    engine = ReasoningEngine(graph)
    metrics, _ = run_evaluation_suite(engine)
    print("\n" + "=" * 55)
    print("      REASONING ENGINE UNSEEN EVALUATION BENCHMARK")
    print("=" * 55)
    print(f"  Total Test Cases               : {metrics.total_test_cases}")
    print(f"  Concept Resolution Precision   : {metrics.concept_resolution_precision * 100:.1f}%")
    print(f"  Prior Art Recall@1             : {metrics.prior_art_recall_at_1 * 100:.1f}%")
    print(f"  Prior Art Recall@3             : {metrics.prior_art_recall_at_3 * 100:.1f}%")
    print(f"  Pitfall Identification Rate    : {metrics.pitfall_detection_rate * 100:.1f}%")
    print(f"  Benchmark Alignment Rate       : {metrics.benchmark_alignment_rate * 100:.1f}%")
    print(f"  Reading Path Completeness      : {metrics.reading_path_completeness * 100:.1f}%")
    print("=" * 55 + "\n")
