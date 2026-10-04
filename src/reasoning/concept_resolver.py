"""Concept Resolution and Proposal Decomposition Engine.

Deconstructs an unseen research proposal into:
- Structured Proposal Understanding
- Typed Ontological Concepts with confidence scoring
- Formal Architectural Fingerprint
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

class ConceptResolutionItem(BaseModel):
    input_concept: str
    resolved_entity_id: str
    resolved_entity_name: str
    entity_type: str
    matching_rationale: str
    confidence: float
    evidence: str

class ArchitecturalFingerprint(BaseModel):
    memory_types: List[str]
    mechanisms: List[str]
    operations: List[str]
    stored_information: str
    retrieval_pattern: str
    update_pattern: str
    evaluation_tasks: List[str]

class DecomposedProposal(BaseModel):
    raw_query: str
    summary: str
    stored_information: str
    retrieval_trigger: str
    update_trigger: str
    intended_benefit: str
    resolved_items: List[ConceptResolutionItem] = Field(default_factory=list)
    fingerprint: ArchitecturalFingerprint

# Ontological triggers with semantic weights
MEMORY_TYPE_PATTERNS = {
    "memtype_working": {
        "name": "Working & In-Context Memory",
        "patterns": ["working memory", "scratchpad", "context window", "paging", "sliding window", "in-context", "ram", "buffer", "kv cache"],
        "rationale": "Proposal manages active short-term context buffers or in-context scratchpads.",
    },
    "memtype_episodic": {
        "name": "Episodic & Experience Memory",
        "patterns": ["episodic", "trajectory", "trajectories", "past experience", "history", "trial", "trials", "runs", "sessions", "episodes", "experience"],
        "rationale": "Proposal records sequential, timestamped logs of past agent runs or interaction outcomes.",
    },
    "memtype_reflective": {
        "name": "Reflective & Evaluative Memory",
        "patterns": ["reflection", "self-reflect", "critique", "feedback", "post-mortem", "verbal reinforcement", "lesson", "lessons", "self-correction", "error diagnosis"],
        "rationale": "Proposal analyzes task feedback or failures to extract abstract verbal critiques and lessons.",
    },
    "memtype_procedural": {
        "name": "Procedural & Skill Memory",
        "patterns": ["procedural", "skill", "skills", "code library", "action library", "tool", "tools", "api", "executable", "scripts", "programs"],
        "rationale": "Proposal caches reusable executable action programs, tool invocation schemas, or workflows.",
    },
    "memtype_semantic": {
        "name": "Semantic & Relational Memory",
        "patterns": ["semantic", "knowledge graph", "triplet", "triplets", "associative", "pagerank", "long-term facts", "kg", "ontology", "relational"],
        "rationale": "Proposal stores persistent factual entities and relational assertions.",
    },
}

OPERATION_PATTERNS = {
    "op_store": {
        "name": "Store",
        "patterns": ["store", "stores", "log", "logs", "record", "records", "save", "saves", "write", "writes", "append", "cache", "persist"],
        "rationale": "Encodes new observations, reflections, or skills into memory storage.",
    },
    "op_retrieve": {
        "name": "Retrieve",
        "patterns": ["retrieve", "retrieves", "recall", "recalls", "search", "searches", "fetch", "fetches", "query", "look up", "lookup"],
        "rationale": "Fetches context-relevant historical memories during planning or decision making.",
    },
    "op_consolidate": {
        "name": "Consolidate",
        "patterns": ["consolidate", "consolidates", "merge", "merges", "summarize", "summarizes", "abstract", "abstracts", "distill", "cluster", "update"],
        "rationale": "Synthesizes, merges, or compresses existing memories over time.",
    },
    "op_reflect": {
        "name": "Reflect",
        "patterns": ["reflect", "reflects", "diagnose", "diagnoses", "critique", "critiques", "evaluate error", "analyze failure", "self-correct"],
        "rationale": "Analyzes execution traces to extract causes of failure and verbal guidance.",
    },
    "op_evict": {
        "name": "Evict",
        "patterns": ["evict", "evicts", "prune", "prunes", "forget", "forgets", "decay", "decays", "delete", "expire", "compress"],
        "rationale": "Discards low-utility, decaying, or conflicting memories to preserve budget.",
    },
}

BENCHMARK_PATTERNS = {
    "bm_alfworld": {
        "name": "ALFWorld",
        "patterns": ["embodied", "household", "text world", "pick and place", "alfworld", "physical environment", "interactive household"],
        "capability": "Embodied multi-step decision planning with trial-and-error memory.",
    },
    "bm_webarena": {
        "name": "WebArena",
        "patterns": ["web", "browser", "website", "e-commerce", "multi-tab", "webarena", "online", "dom"],
        "capability": "Long-horizon web navigation across realistic multi-page websites.",
    },
    "bm_hotpotqa": {
        "name": "HotpotQA",
        "patterns": ["multi-hop", "question answering", "wikipedia", "qa", "hotpotqa", "associative recall"],
        "capability": "Multi-hop cross-document associative retrieval and joint reasoning.",
    },
    "bm_swe_bench": {
        "name": "SWE-bench",
        "patterns": ["github", "software engineering", "coding", "swe-bench", "swebench", "bug fix", "unit test", "repository"],
        "capability": "Resolving real-world software engineering issues requiring long code history.",
    },
    "bm_minecraft": {
        "name": "Minecraft Voyager Testbed",
        "patterns": ["minecraft", "voyager", "open-ended", "crafting", "embodied exploration"],
        "capability": "Lifelong open-ended exploration and iterative program skill synthesis.",
    },
}

class ProposalDecomposer:
    def decompose(self, query: str) -> DecomposedProposal:
        """Deconstruct research proposal into formal architectural components and concept matches."""
        text_lower = query.lower()
        cleaned_tokens = re.findall(r"\b\w+\b", text_lower)

        resolved_items: List[ConceptResolutionItem] = []
        matched_memtype_ids = []
        matched_op_ids = []
        matched_bm_ids = []

        # 1. Resolve Memory Types
        for m_id, data in MEMORY_TYPE_PATTERNS.items():
            matched_kws = [p for p in data["patterns"] if p in text_lower]
            if matched_kws:
                matched_memtype_ids.append(m_id)
                confidence = min(0.70 + 0.10 * len(matched_kws), 0.95)
                resolved_items.append(
                    ConceptResolutionItem(
                        input_concept=f"Phrases: '{', '.join(matched_kws)}'",
                        resolved_entity_id=m_id,
                        resolved_entity_name=data["name"],
                        entity_type="MemoryType",
                        matching_rationale=data["rationale"],
                        confidence=round(confidence, 2),
                        evidence=f"Direct ontological alignment with {data['name']}",
                    )
                )

        # Fallback if no memory type matched
        if not matched_memtype_ids:
            matched_memtype_ids = ["memtype_episodic"]
            resolved_items.append(
                ConceptResolutionItem(
                    input_concept="General agent interaction traces",
                    resolved_entity_id="memtype_episodic",
                    resolved_entity_name="Episodic & Experience Memory",
                    entity_type="MemoryType",
                    matching_rationale="Defaulting to episodic memory baseline for agent observation logging.",
                    confidence=0.60,
                    evidence="Inferred from general interaction framing",
                )
            )

        # 2. Resolve Operations
        for op_id, data in OPERATION_PATTERNS.items():
            matched_kws = [p for p in data["patterns"] if p in text_lower]
            if matched_kws:
                matched_op_ids.append(op_id)
                confidence = min(0.75 + 0.08 * len(matched_kws), 0.96)
                resolved_items.append(
                    ConceptResolutionItem(
                        input_concept=f"Action: '{', '.join(matched_kws)}'",
                        resolved_entity_id=op_id,
                        resolved_entity_name=data["name"],
                        entity_type="MemoryOperation",
                        matching_rationale=data["rationale"],
                        confidence=round(confidence, 2),
                        evidence=f"Matches lifecycle operation '{data['name']}' in ontology",
                    )
                )

        if not matched_op_ids:
            matched_op_ids = ["op_store", "op_retrieve"]

        # 3. Resolve Benchmarks
        for bm_id, data in BENCHMARK_PATTERNS.items():
            matched_kws = [p for p in data["patterns"] if p in text_lower]
            if matched_kws:
                matched_bm_ids.append(bm_id)
                confidence = min(0.80 + 0.08 * len(matched_kws), 0.98)
                resolved_items.append(
                    ConceptResolutionItem(
                        input_concept=f"Domain cue: '{', '.join(matched_kws)}'",
                        resolved_entity_id=bm_id,
                        resolved_entity_name=data["name"],
                        entity_type="Benchmark",
                        matching_rationale=data["capability"],
                        confidence=round(confidence, 2),
                        evidence=f"Empirical benchmark standard for {data['name']}",
                    )
                )

        # 4. Infer Stored Information, Retrieval Pattern, and Update Pattern
        stored_info = "Past task trajectories, state observations, and execution logs"
        if "lesson" in text_lower or "reflection" in text_lower or "critique" in text_lower:
            stored_info = "Verbal self-reflections, error diagnostic critiques, and failure post-mortems"
        elif "skill" in text_lower or "code" in text_lower or "program" in text_lower:
            stored_info = "Verified executable code programs and tool invocation signatures"
        elif "paging" in text_lower or "window" in text_lower:
            stored_info = "Paged active context blocks and external archival conversation history"
        elif "triplet" in text_lower or "knowledge graph" in text_lower:
            stored_info = "Explicit relational entity triplets (head, relation, tail)"

        retrieval_pattern = "Similarity-based recall indexed on current goal description"
        if "plan" in text_lower or "future" in text_lower:
            retrieval_pattern = "Pre-planning associative recall to prime actor prompt with prior lessons"
        elif "associative" in text_lower or "pagerank" in text_lower:
            retrieval_pattern = "Personalized PageRank traversal over linked entity graph"

        update_pattern = "Post-task episodic append upon task completion"
        if "after each task" in text_lower or "post" in text_lower:
            update_pattern = "Synchronous post-execution reflection and buffer update"
        elif "continuous" in text_lower or "stream" in text_lower:
            update_pattern = "Continuous in-flight streaming buffer update"

        intended_benefit = "Prevent repeating identical planning mistakes and eliminate context saturation."

        fingerprint = ArchitecturalFingerprint(
            memory_types=[MEMORY_TYPE_PATTERNS[m]["name"] for m in matched_memtype_ids],
            mechanisms=["Candidate Memory Buffer", "Retrieval Index"],
            operations=[OPERATION_PATTERNS[o]["name"] for o in matched_op_ids],
            stored_information=stored_info,
            retrieval_pattern=retrieval_pattern,
            update_pattern=update_pattern,
            evaluation_tasks=[BENCHMARK_PATTERNS[b]["name"] for b in matched_bm_ids] if matched_bm_ids else ["Embodied Planning"],
        )

        return DecomposedProposal(
            raw_query=query,
            summary=query.strip(),
            stored_information=stored_info,
            retrieval_trigger="Initiation of new task planning phase",
            update_trigger=update_pattern,
            intended_benefit=intended_benefit,
            resolved_items=resolved_items,
            fingerprint=fingerprint,
        )
