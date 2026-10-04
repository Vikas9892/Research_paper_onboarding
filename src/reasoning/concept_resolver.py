"""Concept Resolution Engine for New Research Proposals.

Deconstructs an unseen, natural language research proposal into known ontological
concepts: Memory Types, Operations, Limitations, and target Benchmarks.
"""

from __future__ import annotations

import re
from typing import Dict, List, Set
from pydantic import BaseModel, Field

from src.models.schema import (
    EntityType,
    MemoryTypeEntity,
    MemoryOperationEntity,
    LimitationEntity,
    BenchmarkEntity,
)

class ResolvedConcepts(BaseModel):
    raw_query: str
    identified_memory_types: List[str] = Field(default_factory=list)
    identified_operations: List[str] = Field(default_factory=list)
    anticipated_limitations: List[str] = Field(default_factory=list)
    suggested_benchmarks: List[str] = Field(default_factory=list)
    salient_keywords: List[str] = Field(default_factory=list)

# Ontological keyword triggers
MEMORY_TYPE_KEYWORDS = {
    "memtype_working": ["working memory", "scratchpad", "context window", "paging", "sliding window", "in-context", "ram", "buffer"],
    "memtype_episodic": ["episodic", "trajectory", "past experience", "history", "trial", "trials", "runs", "sessions", "episodes", "experience"],
    "memtype_reflective": ["reflection", "self-reflect", "critique", "feedback", "post-mortem", "verbal reinforcement", "lesson", "self-correction"],
    "memtype_procedural": ["procedural", "skill", "skills", "code library", "action library", "tool", "tools", "api", "executable", "scripts"],
    "memtype_semantic": ["semantic", "knowledge graph", "triplet", "associative", "pagerank", "long-term facts", "kg", "ontology", "relational"],
}

OPERATION_KEYWORDS = {
    "op_store": ["store", "log", "record", "save", "write", "append", "cache", "persist"],
    "op_retrieve": ["retrieve", "recall", "search", "fetch", "query", "look up", "lookup", "similarity search"],
    "op_consolidate": ["consolidate", "merge", "summarize", "abstract", "distill", "cluster", "update"],
    "op_reflect": ["reflect", "diagnose", "critique", "evaluate error", "analyze failure", "self-correct"],
    "op_evict": ["evict", "prune", "forget", "decay", "delete", "expire", "compress"],
}

LIMITATION_KEYWORDS = {
    "limit_context_overflow": ["overflow", "token limit", "long horizon", "context exhaustion", "window saturation"],
    "limit_retrieval_distraction": ["distraction", "irrelevant memory", "noisy retrieval", "misleading memory"],
    "limit_reflection_loops": ["infinite loop", "hallucinated reflection", "cyclic failure", "looping", "repeating mistake"],
    "limit_memory_drift": ["drift", "hallucination accumulation", "distorted fact", "confabulation"],
    "limit_skill_proliferation": ["skill bloat", "redundant skills", "proliferation", "duplicate skills"],
    "limit_spurious_rules": ["spurious rule", "false induction", "overfitting to one trial"],
    "limit_relational_gap": ["triplet limitation", "expressivity gap", "binary relation loss"],
    "limit_inference_latency": ["latency", "slow", "expensive rollouts", "mcts cost", "token overhead"],
    "limit_prompt_injection": ["prompt injection", "memory poisoning", "security vulnerability", "adversarial memory"],
    "limit_dom_volatility": ["dom changes", "ui drift", "layout change", "website redesign"],
}

BENCHMARK_KEYWORDS = {
    "bm_alfworld": ["embodied", "household", "text world", "pick and place", "alfworld", "physical environment"],
    "bm_webarena": ["web", "browser", "website", "e-commerce", "multi-tab", "webarena", "online"],
    "bm_hotpotqa": ["multi-hop", "question answering", "wikipedia", "qa", "hotpotqa"],
    "bm_gaia": ["general assistant", "gaia", "multimodal assistant", "complex tool"],
    "bm_swe_bench": ["github", "software engineering", "coding", "swe-bench", "swebench", "bug fix", "unit test"],
    "bm_scienceworld": ["science", "scientific experiment", "scienceworld"],
    "bm_mind2web": ["web navigation", "mind2web", "dom element"],
    "bm_minecraft": ["minecraft", "voyager", "open-ended", "crafting", "embodied exploration"],
    "bm_humaneval_mbpp": ["python", "humaneval", "mbpp", "code generation"],
}

class ConceptResolver:
    def __init__(self):
        pass

    def resolve(self, query: str) -> ResolvedConcepts:
        """Parse natural language query into known ontological concepts."""
        text = query.lower()
        cleaned_words = re.findall(r"\b\w+\b", text)

        # Match memory types
        matched_memtypes = []
        for type_id, triggers in MEMORY_TYPE_KEYWORDS.items():
            if any(t in text for t in triggers):
                matched_memtypes.append(type_id)

        # Match operations
        matched_ops = []
        for op_id, triggers in OPERATION_KEYWORDS.items():
            if any(t in text for t in triggers):
                matched_ops.append(op_id)

        # Match limitations
        matched_limits = []
        for lim_id, triggers in LIMITATION_KEYWORDS.items():
            if any(t in text for t in triggers):
                matched_limits.append(lim_id)

        # Match benchmarks
        matched_bms = []
        for bm_id, triggers in BENCHMARK_KEYWORDS.items():
            if any(t in text for t in triggers):
                matched_bms.append(bm_id)

        # If no explicit operation matched, default to store and retrieve
        if not matched_ops:
            matched_ops = ["op_store", "op_retrieve"]

        # If no explicit memory type matched, default to episodic and working
        if not matched_memtypes:
            matched_memtypes = ["memtype_episodic"]

        # Collect salient keywords
        stop_words = {"i", "want", "to", "build", "an", "agent", "that", "and", "or", "in", "a", "the", "with", "for", "on", "from"}
        salient = [w for w in cleaned_words if w not in stop_words and len(w) > 3][:10]

        return ResolvedConcepts(
            raw_query=query,
            identified_memory_types=matched_memtypes,
            identified_operations=matched_ops,
            anticipated_limitations=matched_limits,
            suggested_benchmarks=matched_bms,
            salient_keywords=salient,
        )
