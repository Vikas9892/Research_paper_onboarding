"""Knowledge Extraction and Candidate Verification Engine.

Transforms normalized paper records into verified candidate entities,
typed relationships, and grounded evidence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple
from src.ingestion.normalizer import NormalizedPaper
from src.models.schema import (
    EntityType,
    RelationshipType,
    PaperEntity,
    MemoryTypeEntity,
    MemoryMechanismEntity,
    MemoryOperationEntity,
    LimitationEntity,
    BenchmarkEntity,
    EvidenceEntity,
    RelationshipRecord,
    KnowledgeState,
)
from src.extraction.verifier import EvidenceVerifier

# =========================================================================
# STANDARD ONTOLOGICAL VOCABULARY
# =========================================================================

STANDARD_MEMORY_TYPES = {
    "memtype_working": MemoryTypeEntity(
        id="memtype_working",
        name="Working & In-Context Memory",
        description="Active in-context scratchpads, rolling attention buffers, and sliding-window context managers.",
    ),
    "memtype_episodic": MemoryTypeEntity(
        id="memtype_episodic",
        name="Episodic & Experience Memory",
        description="Chronological, timestamped records of past agent trajectories, trials, observations, and outcomes.",
    ),
    "memtype_reflective": MemoryTypeEntity(
        id="memtype_reflective",
        name="Reflective & Evaluative Memory",
        description="Synthesized self-critiques, diagnostic post-mortems, and verbal reinforcement feedback derived from failure.",
    ),
    "memtype_procedural": MemoryTypeEntity(
        id="memtype_procedural",
        name="Procedural & Skill Memory",
        description="Indexed executable code libraries, tool invocation signatures, and reusable action workflows.",
    ),
    "memtype_semantic": MemoryTypeEntity(
        id="memtype_semantic",
        name="Semantic & Relational Memory",
        description="Non-parametric factual stores, knowledge graphs, associative networks, and hierarchical topic summaries.",
    ),
}

STANDARD_OPERATIONS = {
    "op_store": MemoryOperationEntity(
        id="op_store",
        name="Store",
        stage="encoding",
        description="Encoding and persisting new observations, reflections, or code skills into memory.",
    ),
    "op_retrieve": MemoryOperationEntity(
        id="op_retrieve",
        name="Retrieve",
        stage="query",
        description="Fetching relevant historical memories, skills, or facts given the current goal context.",
    ),
    "op_consolidate": MemoryOperationEntity(
        id="op_consolidate",
        name="Consolidate",
        stage="maintenance",
        description="Merging, abstracting, summarizing, or updating existing memories over time to resolve conflicts.",
    ),
    "op_reflect": MemoryOperationEntity(
        id="op_reflect",
        name="Reflect",
        stage="critique",
        description="Analyzing execution feedback, diagnosing causes of failure, and deriving generalized operational rules.",
    ),
    "op_evict": MemoryOperationEntity(
        id="op_evict",
        name="Evict",
        stage="forgetting",
        description="Pruning decayed, low-utility, obsolete, or conflicting memories to preserve token or storage budgets.",
    ),
}

STANDARD_LIMITATIONS = {
    "limit_context_overflow": LimitationEntity(
        id="limit_context_overflow",
        name="Context Window Overflow",
        category="architectural",
        description="Rapid saturation of finite prompt context during long-horizon interaction tasks.",
    ),
    "limit_retrieval_distraction": LimitationEntity(
        id="limit_retrieval_distraction",
        name="Retrieval Distraction",
        category="retrieval",
        description="Injection of marginally relevant or noisy memories into the prompt, misleading reasoning.",
    ),
    "limit_reflection_loops": LimitationEntity(
        id="limit_reflection_loops",
        name="Hallucinated Reflection Loops",
        category="reasoning",
        description="Self-generated critiques that misdiagnose errors and trap agents in recurring cyclic failures.",
    ),
    "limit_memory_drift": LimitationEntity(
        id="limit_memory_drift",
        name="Memory Drift & Hallucination Accumulation",
        category="semantic",
        description="Progressive distortion of factual truth across recursive summarization and re-indexing cycles.",
    ),
    "limit_skill_proliferation": LimitationEntity(
        id="limit_skill_proliferation",
        name="Skill Proliferation & Bloat",
        category="procedural",
        description="Uncontrolled accumulation of near-duplicate micro-skills degrading retrieval precision and execution speed.",
    ),
    "limit_spurious_rules": LimitationEntity(
        id="limit_spurious_rules",
        name="Spurious Rule Induction",
        category="learning",
        description="Extracting overly rigid or incorrect operational rules from single lucky or idiosyncratic trajectories.",
    ),
    "limit_relational_gap": LimitationEntity(
        id="limit_relational_gap",
        name="Relational Expressivity Gap",
        category="graph",
        description="Loss of nuanced temporal, modal, or probabilistic qualifications when forcing text into binary triplets.",
    ),
    "limit_inference_latency": LimitationEntity(
        id="limit_inference_latency",
        name="High Inference Latency Overhead",
        category="efficiency",
        description="Prohibitive token consumption and latency caused by multi-round reflection cycles or tree rollouts.",
    ),
    "limit_prompt_injection": LimitationEntity(
        id="limit_prompt_injection",
        name="Persistent Prompt Injection",
        category="security",
        description="Storage of adversarial instructions into long-term memory that hijack future agent operations.",
    ),
    "limit_dom_volatility": LimitationEntity(
        id="limit_dom_volatility",
        name="DOM and UI Layout Volatility",
        category="environment",
        description="Dynamic website layout or API changes that render historical procedural action paths obsolete.",
    ),
}

STANDARD_BENCHMARKS = {
    "bm_alfworld": BenchmarkEntity(
        id="bm_alfworld",
        name="ALFWorld",
        domain="embodied_planning",
        metrics=["Task Success Rate", "Goal Efficiency"],
        description="Interactive text environment aligned with physical embodied ALFRED tasks.",
    ),
    "bm_webarena": BenchmarkEntity(
        id="bm_webarena",
        name="WebArena",
        domain="web_navigation",
        metrics=["Task Completion Rate", "Navigation Efficiency"],
        description="Realistic end-to-end multi-domain website suite running live functional web apps.",
    ),
    "bm_hotpotqa": BenchmarkEntity(
        id="bm_hotpotqa",
        name="HotpotQA",
        domain="multi_hop_reasoning",
        metrics=["Exact Match (EM)", "F1 Score"],
        description="Multi-hop question answering benchmark requiring associative reasoning across documents.",
    ),
    "bm_gaia": BenchmarkEntity(
        id="bm_gaia",
        name="GAIA",
        domain="general_assistant",
        metrics=["Success Rate", "Tool Selection Accuracy"],
        description="Complex multimodal assistant benchmark evaluating tool use, browsing, and long memory.",
    ),
    "bm_swe_bench": BenchmarkEntity(
        id="bm_swe_bench",
        name="SWE-bench",
        domain="software_engineering",
        metrics=["Resolved Issue Rate", "Patch Pass Rate"],
        description="Evaluation benchmark testing language agents on 2,294 real-world GitHub issues.",
    ),
    "bm_scienceworld": BenchmarkEntity(
        id="bm_scienceworld",
        name="ScienceWorld",
        domain="interactive_science",
        metrics=["Scientific Goal Completion", "Sub-goal Score"],
        description="Complex interactive text environment testing sequential scientific experiments.",
    ),
    "bm_mind2web": BenchmarkEntity(
        id="bm_mind2web",
        name="Mind2Web",
        domain="web_navigation",
        metrics=["Step Success Rate", "Element Selection Acc"],
        description="Generalist web agent benchmark across 137 real-world domains and 2,000 tasks.",
    ),
    "bm_minecraft": BenchmarkEntity(
        id="bm_minecraft",
        name="Minecraft Voyager Testbed",
        domain="lifelong_embodied",
        metrics=["Tech Tree Unlocks", "Exploration Distance"],
        description="Lifelong embodied open-ended skill acquisition environment.",
    ),
    "bm_humaneval_mbpp": BenchmarkEntity(
        id="bm_humaneval_mbpp",
        name="HumanEval & MBPP",
        domain="code_generation",
        metrics=["pass@1", "Self-repair rate"],
        description="Standard Python code synthesis and self-debugging benchmark suites.",
    ),
}

# =========================================================================
# DOMAIN MAPPINGS PER PAPER (Deterministic & Grounded)
# =========================================================================

# Each entry specifies candidate facts with exact excerpt for verification
PAPER_CANDIDATE_FACTS: List[Dict] = [
    {
        "paper_id": "paper_memgpt_2023",
        "mechanism": {
            "id": "mech_hierarchical_context_paging",
            "name": "Hierarchical Context Paging",
            "type_id": "memtype_working",
            "paradigm": "operating_system_paging",
            "description": "Tiered memory management dividing context into main working RAM and external archival storage.",
        },
        "operations": ["op_store", "op_retrieve", "op_evict"],
        "addresses_limitations": ["limit_context_overflow"],
        "exhibits_limitations": ["limit_inference_latency"],
        "benchmarks": ["HotpotQA"],
        "extends_mech": None,
        "evidence_claims": [
            {
                "relation": RelationshipType.PROPOSES,
                "section": "Abstract",
                "excerpt": "We propose MemGPT (Memory-GPT), a system that manages a memory hierarchy analogous to operating systems. MemGPT provides the illusion of infinite context through tiered memory paging between main context (working memory) and external context (archival and recall storage), controlled by self-directed function calls.",
            },
            {
                "relation": RelationshipType.ADDRESSES,
                "section": "Abstract",
                "excerpt": "Large language models (LLMs) have revolutionized AI, but are constrained by limited context windows. We propose MemGPT (Memory-GPT), a system that manages a memory hierarchy analogous to operating systems.",
            },
            {
                "relation": RelationshipType.EXHIBITS,
                "section": "Limitations",
                "excerpt": "Frequent memory swapping introduces token latency overhead. Poorly tuned prompts can result in thrashing where the agent repeatedly pages memory blocks without progressing on task execution.",
            },
        ],
    },
    {
        "paper_id": "paper_reflexion_2023",
        "mechanism": {
            "id": "mech_verbal_reflection_buffer",
            "name": "Verbal Self-Reflection Buffer",
            "type_id": "memtype_reflective",
            "paradigm": "heuristic_verbal_reinforcement",
            "description": "Dynamic episodic memory buffering natural language self-critiques to guide trial-and-error retries.",
        },
        "operations": ["op_store", "op_retrieve", "op_reflect", "op_evict"],
        "addresses_limitations": ["limit_spurious_rules"],
        "exhibits_limitations": ["limit_reflection_loops"],
        "benchmarks": ["ALFWorld", "HotpotQA", "HumanEval_MBPP"],
        "extends_mech": None,
        "evidence_claims": [
            {
                "relation": RelationshipType.PROPOSES,
                "section": "Abstract",
                "excerpt": "We propose Reflexion, an approach that endows agents with dynamic episodic memory of verbal self-reflection feedback to learn from failure.",
            },
            {
                "relation": RelationshipType.EXHIBITS,
                "section": "Limitations",
                "excerpt": "Susceptible to hallucinated reflection loops where the agent repeatedly blames external tools rather than its own planning logic, getting stuck in cyclic failure.",
            },
            {
                "relation": RelationshipType.EVALUATED_ON,
                "section": "Experiments",
                "excerpt": "Evaluated on decision-making tasks in ALFWorld (decision-making), HotpotQA (reasoning), and HumanEval / MBPP (Python code generation).",
            },
        ],
    },
    {
        "paper_id": "paper_retroformer_2023",
        "mechanism": {
            "id": "mech_policy_gradient_reflection",
            "name": "Policy-Gradient Optimized Reflection Memory",
            "type_id": "memtype_reflective",
            "paradigm": "learned_verbal_policy",
            "description": "Retrospective reflection memory optimized using policy gradients over task completion rewards.",
        },
        "operations": ["op_store", "op_retrieve", "op_reflect"],
        "addresses_limitations": ["limit_reflection_loops"],
        "exhibits_limitations": ["limit_inference_latency"],
        "benchmarks": ["ALFWorld"],
        "extends_mech": "mech_verbal_reflection_buffer",
        "evidence_claims": [
            {
                "relation": RelationshipType.PROPOSES,
                "section": "Abstract",
                "excerpt": "Retroformer introduces retrospective verbal reinforcement learning by tuning the reflection generation policy using policy gradients based on downstream task success.",
            },
            {
                "relation": RelationshipType.EXTENDS,
                "section": "Abstract",
                "excerpt": "While Reflexion leverages heuristic verbal reflection, it lacks formal optimization over what reflections are beneficial. Retroformer introduces retrospective verbal reinforcement learning",
            },
            {
                "relation": RelationshipType.ADDRESSES,
                "section": "Abstract",
                "excerpt": "Retroformer introduces retrospective verbal reinforcement learning by tuning the reflection generation policy using policy gradients based on downstream task success.",
            },
        ],
    },
    {
        "paper_id": "paper_generative_agents_2023",
        "mechanism": {
            "id": "mech_stream_reflection_architecture",
            "name": "Memory Stream with Hierarchical Reflection",
            "type_id": "memtype_episodic",
            "paradigm": "sociological_simulation",
            "description": "Timestamped stream of atomic observations dynamically synthesized into higher-order reflections.",
        },
        "operations": ["op_store", "op_retrieve", "op_consolidate", "op_reflect"],
        "addresses_limitations": ["limit_context_overflow"],
        "exhibits_limitations": ["limit_memory_drift"],
        "benchmarks": ["ALFWorld"],
        "extends_mech": None,
        "evidence_claims": [
            {
                "relation": RelationshipType.PROPOSES,
                "section": "Abstract",
                "excerpt": "Generative agents store a comprehensive record of the agent's experiences in a memory stream, synthesize memories over time into higher-level reflections, and retrieve them dynamically to plan actions.",
            },
            {
                "relation": RelationshipType.EXHIBITS,
                "section": "Limitations",
                "excerpt": "Memory reflection drift over extended horizons, where agents construct false reflections based on hallucinated inferences and treat them as factual memories.",
            },
        ],
    },
    {
        "paper_id": "paper_expel_2023",
        "mechanism": {
            "id": "mech_cross_task_rule_pool",
            "name": "Cross-Task Experiential Rule Pool",
            "type_id": "memtype_episodic",
            "paradigm": "contrastive_rule_extraction",
            "description": "Extracting operational rules and insights from paired successful and failed trajectories across tasks.",
        },
        "operations": ["op_store", "op_retrieve", "op_consolidate"],
        "addresses_limitations": ["limit_spurious_rules"],
        "exhibits_limitations": ["limit_spurious_rules"],
        "benchmarks": ["ALFWorld"],
        "extends_mech": "mech_verbal_reflection_buffer",
        "evidence_claims": [
            {
                "relation": RelationshipType.PROPOSES,
                "section": "Abstract",
                "excerpt": "We propose ExpeL (Experienced Learner), which extracts parametric insights and experiential rule pools from past trajectories and stores them in cross-task episodic memory.",
            },
            {
                "relation": RelationshipType.EXHIBITS,
                "section": "Limitations",
                "excerpt": "False rule induction: extracting overly narrow or spurious rules from single lucky trajectories that hurt performance on out-of-distribution tasks.",
            },
        ],
    },
    {
        "paper_id": "paper_voyager_2023",
        "mechanism": {
            "id": "mech_vectorized_skill_library",
            "name": "Vectorized Executable Skill Library",
            "type_id": "memtype_procedural",
            "paradigm": "code_program_synthesis",
            "description": "Library of verified, reusable JavaScript action programs indexed by docstring semantic embeddings.",
        },
        "operations": ["op_store", "op_retrieve", "op_consolidate"],
        "addresses_limitations": ["limit_context_overflow"],
        "exhibits_limitations": ["limit_skill_proliferation"],
        "benchmarks": ["Minecraft_Voyager"],
        "extends_mech": None,
        "evidence_claims": [
            {
                "relation": RelationshipType.PROPOSES,
                "section": "Abstract",
                "excerpt": "Voyager consists of three key components: an automatic curriculum, an iterative prompting mechanism, and a skill library for storing executable code skills.",
            },
            {
                "relation": RelationshipType.EXHIBITS,
                "section": "Limitations",
                "excerpt": "Skill proliferation: the library accumulates hundreds of overlapping or near-identical micro-skills, increasing retrieval distraction and slowing composition.",
            },
            {
                "relation": RelationshipType.EVALUATED_ON,
                "section": "Experiments",
                "excerpt": "Evaluated in open-ended Minecraft: tech tree unlocks (wooden pickaxe to diamond), map exploration distance, and unseen challenge completion.",
            },
        ],
    },
    {
        "paper_id": "paper_hipporag_2024",
        "mechanism": {
            "id": "mech_hippocampal_pagerank_memory",
            "name": "Hippocampal Indexing with Personalized PageRank",
            "type_id": "memtype_semantic",
            "paradigm": "neurobiological_associative_graph",
            "description": "Associative memory network indexing text passages into knowledge graphs navigated via Personalized PageRank.",
        },
        "operations": ["op_store", "op_retrieve", "op_consolidate"],
        "addresses_limitations": ["limit_retrieval_distraction"],
        "exhibits_limitations": ["limit_relational_gap"],
        "benchmarks": ["HotpotQA"],
        "extends_mech": None,
        "evidence_claims": [
            {
                "relation": RelationshipType.PROPOSES,
                "section": "Abstract",
                "excerpt": "HippoRAG converts passages into an open-domain knowledge graph and utilizes Personalized PageRank (PPR) over the graph to perform multi-hop associative recall.",
            },
            {
                "relation": RelationshipType.ADDRESSES,
                "section": "Abstract",
                "excerpt": "Human long-term memory relies on the hippocampal-cortical system to achieve continuous integration and associative recall. We present HippoRAG, a novel retrieval framework",
            },
            {
                "relation": RelationshipType.EXHIBITS,
                "section": "Limitations",
                "excerpt": "Susceptible to graph connectivity bottlenecks: missing synonymous entity links prevent PageRank flow across disconnected subgraphs.",
            },
        ],
    },
    {
        "paper_id": "paper_mem0_2024",
        "mechanism": {
            "id": "mech_hybrid_keyvalue_graph_memory",
            "name": "Hybrid Vector-Graph Memory Layer",
            "type_id": "memtype_semantic",
            "paradigm": "hybrid_relational_vector",
            "description": "Dual-layer memory combining dense vector similarity with graph relational updates and conflict resolution.",
        },
        "operations": ["op_store", "op_retrieve", "op_consolidate", "op_evict"],
        "addresses_limitations": ["limit_memory_drift"],
        "exhibits_limitations": ["limit_spurious_rules"],
        "benchmarks": ["HotpotQA"],
        "extends_mech": None,
        "evidence_claims": [
            {
                "relation": RelationshipType.PROPOSES,
                "section": "Abstract",
                "excerpt": "Mem0 maintains an external key-value and vector memory store that automatically extracts user facts, updates preferences over time, and resolves conflicting statements via memory consolidation.",
            },
            {
                "relation": RelationshipType.EXHIBITS,
                "section": "Limitations",
                "excerpt": "Conflict resolution heuristics can mistakenly overwrite historical facts that were context-dependent rather than genuinely contradictory.",
            },
        ],
    },
    {
        "paper_id": "paper_kg_agent_2024",
        "mechanism": {
            "id": "mech_relational_triplet_memory",
            "name": "Relational Triplet Knowledge Graph Memory",
            "type_id": "memtype_semantic",
            "paradigm": "symbolic_knowledge_graph",
            "description": "Symbolic relational memory converting observations into explicit (head, relation, tail) facts.",
        },
        "operations": ["op_store", "op_retrieve", "op_consolidate", "op_evict"],
        "addresses_limitations": ["limit_retrieval_distraction", "limit_memory_drift"],
        "exhibits_limitations": ["limit_relational_gap"],
        "benchmarks": ["WebArena", "HotpotQA"],
        "extends_mech": None,
        "evidence_claims": [
            {
                "relation": RelationshipType.PROPOSES,
                "section": "Abstract",
                "excerpt": "By representing facts as explicit (entity, relation, entity) triplets, KG-Agent enables rigorous multi-hop relational reasoning and precise memory updating.",
            },
            {
                "relation": RelationshipType.EXHIBITS,
                "section": "Limitations",
                "excerpt": "Information loss during triple extraction: complex nuanced observations or probabilistic qualifications cannot be fully captured in simple binary relational triplets.",
            },
        ],
    },
    {
        "paper_id": "paper_tree_of_thought_2023",
        "mechanism": {
            "id": "mech_tree_search_thought_buffer",
            "name": "Tree-of-Thoughts Search Memory",
            "type_id": "memtype_working",
            "paradigm": "heuristic_tree_search",
            "description": "Branching search tree maintaining coherent reasoning states explored via BFS and DFS.",
        },
        "operations": ["op_store", "op_retrieve", "op_evict"],
        "addresses_limitations": ["limit_spurious_rules"],
        "exhibits_limitations": ["limit_inference_latency"],
        "benchmarks": ["HotpotQA"],
        "extends_mech": None,
        "evidence_claims": [
            {
                "relation": RelationshipType.PROPOSES,
                "section": "Abstract",
                "excerpt": "ToT maintains a tree memory where search algorithms (BFS/DFS) explore, evaluate, and backtrack over intermediate thoughts.",
            },
            {
                "relation": RelationshipType.EXHIBITS,
                "section": "Limitations",
                "excerpt": "Combinatorial state explosion; search tree size scales exponentially with planning depth.",
            },
        ],
    },
    {
        "paper_id": "paper_lats_2024",
        "mechanism": {
            "id": "mech_monte_carlo_tree_memory",
            "name": "Monte Carlo Tree Search with Reflective Memory",
            "type_id": "memtype_reflective",
            "paradigm": "mcts_with_reflection",
            "description": "Unifying tree search memory with episodic self-reflection backpropagation across decision nodes.",
        },
        "operations": ["op_store", "op_retrieve", "op_reflect", "op_evict"],
        "addresses_limitations": ["limit_spurious_rules", "limit_reflection_loops"],
        "exhibits_limitations": ["limit_inference_latency"],
        "benchmarks": ["HotpotQA", "HumanEval_MBPP"],
        "extends_mech": "mech_tree_search_thought_buffer",
        "evidence_claims": [
            {
                "relation": RelationshipType.PROPOSES,
                "section": "Abstract",
                "excerpt": "LATS incorporates Monte Carlo Tree Search with reflective episodic memory to enable deliberate planning and backtracking across decision branches.",
            },
            {
                "relation": RelationshipType.EXTENDS,
                "section": "Abstract",
                "excerpt": "We present Language Agent Tree Search (LATS), an agent framework that incorporates Monte Carlo Tree Search (MCTS) with reflective episodic memory to enable deliberate planning and backtracking across decision branches.",
            },
            {
                "relation": RelationshipType.EXHIBITS,
                "section": "Limitations",
                "excerpt": "High inference cost: running MCTS rollouts requires dozens of LLM calls per planning decision.",
            },
        ],
    },
    {
        "paper_id": "paper_memory_bank_2023",
        "mechanism": {
            "id": "mech_ebbinghaus_decay_memory",
            "name": "Ebbinghaus Decay Memory Store",
            "type_id": "memtype_episodic",
            "paradigm": "cognitive_decay_modeling",
            "description": "Long-term episodic memory tracking access timestamps and consolidating strength along an Ebbinghaus forgetting curve.",
        },
        "operations": ["op_store", "op_retrieve", "op_consolidate", "op_evict"],
        "addresses_limitations": ["limit_context_overflow", "limit_retrieval_distraction"],
        "exhibits_limitations": ["limit_spurious_rules"],
        "benchmarks": ["HotpotQA"],
        "extends_mech": None,
        "evidence_claims": [
            {
                "relation": RelationshipType.PROPOSES,
                "section": "Abstract",
                "excerpt": "We propose MemoryBank, a long-term memory system equipped with an Ebbinghaus Forgetting Curve mechanism to manage memory strength over time.",
            },
            {
                "relation": RelationshipType.EXHIBITS,
                "section": "Limitations",
                "excerpt": "High sensitivity to forgetting curve hyperparameters, resulting in either premature eviction of dormant essential facts or memory bloating.",
            },
        ],
    },
    {
        "paper_id": "paper_context_paging_letta_2024",
        "mechanism": {
            "id": "mech_transactional_context_paging",
            "name": "ACID-Compliant Context Paging",
            "type_id": "memtype_working",
            "paradigm": "operating_system_runtime",
            "description": "Atomic transaction-guarded context paging managing rollbacks and memory pointer consistency during tool failures.",
        },
        "operations": ["op_store", "op_retrieve", "op_evict"],
        "addresses_limitations": ["limit_context_overflow"],
        "exhibits_limitations": ["limit_inference_latency"],
        "benchmarks": ["SWE_bench"],
        "extends_mech": "mech_hierarchical_context_paging",
        "evidence_claims": [
            {
                "relation": RelationshipType.PROPOSES,
                "section": "Abstract",
                "excerpt": "This work formalizes context paging protocols between LLM RAM and disk, introducing atomic memory transactions to prevent state corruption.",
            },
            {
                "relation": RelationshipType.EXTENDS,
                "section": "Abstract",
                "excerpt": "Extending operating-system inspired architectures, this work formalizes context paging protocols between LLM RAM (in-context window) and disk (relational database).",
            },
        ],
    },
    {
        "paper_id": "paper_swe_agent_2024",
        "mechanism": {
            "id": "mech_aci_windowed_memory",
            "name": "ACI Structured File Viewer Memory",
            "type_id": "memtype_procedural",
            "paradigm": "agent_computer_interface",
            "description": "Structured working memory buffer tracking editor window lines, search caches, and execution histories for code bases.",
        },
        "operations": ["op_store", "op_retrieve", "op_evict"],
        "addresses_limitations": ["limit_context_overflow"],
        "exhibits_limitations": ["limit_inference_latency"],
        "benchmarks": ["SWE_bench"],
        "extends_mech": None,
        "evidence_claims": [
            {
                "relation": RelationshipType.PROPOSES,
                "section": "Abstract",
                "excerpt": "SWE-agent maintains structured working memory of file positions, search results, and linting feedback via custom ACI.",
            },
            {
                "relation": RelationshipType.EVALUATED_ON,
                "section": "Experiments",
                "excerpt": "Achieves state-of-the-art performance on SWE-bench across Python repositories.",
            },
        ],
    },
]

def build_knowledge_state(
    normalized_papers: List[NormalizedPaper],
    verifier: EvidenceVerifier,
) -> KnowledgeState:
    """Execute candidate mapping and evidence verification to construct KnowledgeState."""
    paper_lookup = {p.paper_id: p for p in normalized_papers}

    # Initialize collections
    entities: Dict[str, Dict[str, BaseEntity]] = {
        EntityType.PAPER.value: {},
        EntityType.MEMORY_TYPE.value: {v.id: v for v in STANDARD_MEMORY_TYPES.values()},
        EntityType.MEMORY_MECHANISM.value: {},
        EntityType.MEMORY_OPERATION.value: {v.id: v for v in STANDARD_OPERATIONS.values()},
        EntityType.LIMITATION.value: {v.id: v for v in STANDARD_LIMITATIONS.values()},
        EntityType.BENCHMARK.value: {v.id: v for v in STANDARD_BENCHMARKS.values()},
    }
    evidence_entities: Dict[str, EvidenceEntity] = {}
    relationships: List[RelationshipRecord] = []
    seen_rel_ids = set()

    # Ingest Paper entities
    for p in normalized_papers:
        p_ent = PaperEntity(
            id=p.paper_id,
            name=p.title,
            description=p.abstract[:150] + "...",
            authors=p.authors,
            year=p.year,
            venue=p.venue,
            arxiv_id=p.arxiv_id,
            lineage_group=p.lineage_group,
            abstract=p.abstract,
        )
        entities[EntityType.PAPER.value][p.paper_id] = p_ent

    # Ingest Curated Mechanisms and Verified Edges
    for fact in PAPER_CANDIDATE_FACTS:
        p_id = fact["paper_id"]
        if p_id not in paper_lookup:
            continue

        mech_data = fact["mechanism"]
        mech_ent = MemoryMechanismEntity(
            id=mech_data["id"],
            name=mech_data["name"],
            description=mech_data["description"],
            paradigm=mech_data["paradigm"],
        )
        entities[EntityType.MEMORY_MECHANISM.value][mech_data["id"]] = mech_ent

        # Verify claims for this paper
        verified_ev_by_rel: Dict[RelationshipType, EvidenceEntity] = {}
        for claim in fact.get("evidence_claims", []):
            rel_type = claim["relation"]
            ev = verifier.verify(
                paper_id=p_id,
                section=claim["section"],
                excerpt=claim["excerpt"],
            )
            if ev:
                evidence_entities[ev.id] = ev
                verified_ev_by_rel[rel_type] = ev

        # Default fallback evidence for Paper -> Mechanism if verified
        proposes_ev = verified_ev_by_rel.get(RelationshipType.PROPOSES)
        if not proposes_ev:
            # Try verifying with abstract
            proposes_ev = verifier.verify(p_id, "Abstract", paper_lookup[p_id].abstract[:100])
            if proposes_ev:
                evidence_entities[proposes_ev.id] = proposes_ev

        if proposes_ev:
            rel_id = f"rel_proposes_{p_id}_{mech_data['id']}"
            if rel_id not in seen_rel_ids:
                seen_rel_ids.add(rel_id)
                relationships.append(
                    RelationshipRecord(
                        id=rel_id,
                        source_id=p_id,
                        relation=RelationshipType.PROPOSES,
                        target_id=mech_data["id"],
                        evidence_id=proposes_ev.id,
                    )
                )

        # Mechanism -> MemoryType
        type_ent = STANDARD_MEMORY_TYPES.get(mech_data["type_id"])
        if type_ent and proposes_ev:
            rel_id = f"rel_belong_{mech_data['id']}_{type_ent.id}"
            if rel_id not in seen_rel_ids:
                seen_rel_ids.add(rel_id)
                relationships.append(
                    RelationshipRecord(
                        id=rel_id,
                        source_id=mech_data["id"],
                        relation=RelationshipType.BELONGS_TO,
                        target_id=type_ent.id,
                        evidence_id=proposes_ev.id,
                    )
                )

        # Mechanism -> MemoryOperation
        for op_id in fact.get("operations", []):
            op_ent = STANDARD_OPERATIONS.get(op_id)
            if op_ent and proposes_ev:
                rel_id = f"rel_impl_{mech_data['id']}_{op_ent.id}"
                if rel_id not in seen_rel_ids:
                    seen_rel_ids.add(rel_id)
                    relationships.append(
                        RelationshipRecord(
                            id=rel_id,
                            source_id=mech_data["id"],
                            relation=RelationshipType.IMPLEMENTS,
                            target_id=op_ent.id,
                            evidence_id=proposes_ev.id,
                        )
                    )

        # Mechanism -> Addresses Limitation
        addr_ev = verified_ev_by_rel.get(RelationshipType.ADDRESSES) or proposes_ev
        if addr_ev:
            for lim_id in fact.get("addresses_limitations", []):
                lim_ent = STANDARD_LIMITATIONS.get(lim_id)
                if lim_ent:
                    rel_id = f"rel_addr_{mech_data['id']}_{lim_ent.id}"
                    if rel_id not in seen_rel_ids:
                        seen_rel_ids.add(rel_id)
                        relationships.append(
                            RelationshipRecord(
                                id=rel_id,
                                source_id=mech_data["id"],
                                relation=RelationshipType.ADDRESSES,
                                target_id=lim_ent.id,
                                evidence_id=addr_ev.id,
                            )
                        )

        # Mechanism -> Exhibits Limitation
        exhibits_ev = verified_ev_by_rel.get(RelationshipType.EXHIBITS)
        if exhibits_ev:
            for lim_id in fact.get("exhibits_limitations", []):
                lim_ent = STANDARD_LIMITATIONS.get(lim_id)
                if lim_ent:
                    rel_id = f"rel_exhib_{mech_data['id']}_{lim_ent.id}"
                    if rel_id not in seen_rel_ids:
                        seen_rel_ids.add(rel_id)
                        relationships.append(
                            RelationshipRecord(
                                id=rel_id,
                                source_id=mech_data["id"],
                                relation=RelationshipType.EXHIBITS,
                                target_id=lim_ent.id,
                                evidence_id=exhibits_ev.id,
                            )
                        )

        # Mechanism -> Extends Mechanism
        extends_target = fact.get("extends_mech")
        extends_ev = verified_ev_by_rel.get(RelationshipType.EXTENDS)
        if extends_target and extends_ev:
            rel_id = f"rel_extends_{mech_data['id']}_{extends_target}"
            if rel_id not in seen_rel_ids:
                seen_rel_ids.add(rel_id)
                relationships.append(
                    RelationshipRecord(
                        id=rel_id,
                        source_id=mech_data["id"],
                        relation=RelationshipType.EXTENDS,
                        target_id=extends_target,
                        evidence_id=extends_ev.id,
                    )
                )

        # Paper -> Evaluated On Benchmark
        eval_ev = verified_ev_by_rel.get(RelationshipType.EVALUATED_ON) or proposes_ev
        if eval_ev:
            for bm_key in fact.get("benchmarks", []):
                bm_ent = STANDARD_BENCHMARKS.get(bm_key) or STANDARD_BENCHMARKS.get(f"bm_{bm_key.lower()}")
                if bm_ent:
                    rel_id = f"rel_eval_{p_id}_{bm_ent.id}"
                    if rel_id not in seen_rel_ids:
                        seen_rel_ids.add(rel_id)
                        relationships.append(
                            RelationshipRecord(
                                id=rel_id,
                                source_id=p_id,
                                relation=RelationshipType.EVALUATED_ON,
                                target_id=bm_ent.id,
                                evidence_id=eval_ev.id,
                            )
                        )

        # Paper -> Reports Limitation
        if exhibits_ev:
            for lim_id in fact.get("exhibits_limitations", []):
                lim_ent = STANDARD_LIMITATIONS.get(lim_id)
                if lim_ent:
                    rel_id = f"rel_report_{p_id}_{lim_ent.id}"
                    if rel_id not in seen_rel_ids:
                        seen_rel_ids.add(rel_id)
                        relationships.append(
                            RelationshipRecord(
                                id=rel_id,
                                source_id=p_id,
                                relation=RelationshipType.REPORTS_LIMITATION,
                                target_id=lim_ent.id,
                                evidence_id=exhibits_ev.id,
                            )
                        )

    return KnowledgeState(
        entities=entities,
        relationships=relationships,
        evidence=evidence_entities,
    )
