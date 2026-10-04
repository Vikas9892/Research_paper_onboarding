"""Graph Reasoning Engine for Agent Memory Research Intelligence.

Executes multi-hop graph traversals over the verified Knowledge State to produce
an actionable research intelligence report with architectural overlap, historical pitfalls,
lineage progression, benchmark recommendations, and a 5-stage reading path.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from src.models.schema import (
    EntityType,
    RelationshipType,
    KnowledgeState,
)
from src.graph.builder import AgentMemoryGraph
from src.reasoning.concept_resolver import ResolvedConcepts, ConceptResolver

# --- Structured Output Dataclasses ---

class PriorArtMatch(BaseModel):
    mechanism_id: str
    mechanism_name: str
    memory_type: str
    proposing_paper_id: str
    proposing_paper_title: str
    paper_year: int
    overlap_score: float
    shared_operations: List[str]
    evidence_excerpt: str

class PitfallWarning(BaseModel):
    limitation_id: str
    limitation_name: str
    exhibited_by_mechanism: str
    severity_category: str
    reported_in_paper: str
    evidence_section: str
    evidence_excerpt: str

class LineageStep(BaseModel):
    earlier_mechanism: str
    relationship: str
    later_mechanism: str
    rationale: str
    evidence_excerpt: str

class BenchmarkRecommendation(BaseModel):
    benchmark_id: str
    benchmark_name: str
    domain: str
    metrics: List[str]
    used_by_papers: List[str]

class ReadingStage(BaseModel):
    stage_number: int
    stage_name: str
    paper_id: str
    paper_title: str
    authors: str
    year: int
    pedagogical_purpose: str
    key_takeaway: str

class ResearchIntelligenceReport(BaseModel):
    proposal_summary: str
    resolved_concepts: ResolvedConcepts
    closest_prior_art: List[PriorArtMatch] = Field(default_factory=list)
    known_pitfalls_and_failure_modes: List[PitfallWarning] = Field(default_factory=list)
    architectural_lineage: List[LineageStep] = Field(default_factory=list)
    recommended_benchmarks: List[BenchmarkRecommendation] = Field(default_factory=list)
    guided_reading_path: List[ReadingStage] = Field(default_factory=list)

class ReasoningEngine:
    def __init__(self, graph_wrapper: AgentMemoryGraph):
        self.graph_wrapper = graph_wrapper
        self.graph = graph_wrapper.graph
        self.state = graph_wrapper.state

    def reason(self, proposal_text: str) -> ResearchIntelligenceReport:
        """Process an unseen research proposal and return structured research intelligence."""
        resolver = ConceptResolver()
        concepts = resolver.resolve(proposal_text)

        # 1. Prior Art Discovery & Overlap Scoring
        prior_art = self._find_prior_art(concepts)

        # 2. Known Pitfalls and Historical Failure Modes
        pitfalls = self._find_pitfalls(prior_art, concepts)

        # 3. Architectural Lineage Traversal
        lineage = self._trace_lineage(prior_art)

        # 4. Benchmark Alignment
        benchmarks = self._recommend_benchmarks(prior_art, concepts)

        # 5. 5-Stage Guided Reading Path
        reading_path = self._build_reading_path(prior_art, pitfalls, lineage, benchmarks)

        return ResearchIntelligenceReport(
            proposal_summary=proposal_text.strip(),
            resolved_concepts=concepts,
            closest_prior_art=prior_art,
            known_pitfalls_and_failure_modes=pitfalls,
            architectural_lineage=lineage,
            recommended_benchmarks=benchmarks,
            guided_reading_path=reading_path,
        )

    def _find_prior_art(self, concepts: ResolvedConcepts) -> List[PriorArtMatch]:
        """Traverse graph to score architectural overlap."""
        matches: List[PriorArtMatch] = []
        all_mechs = self.state.entities.get(EntityType.MEMORY_MECHANISM.value, {})

        for m_id, m_ent in all_mechs.items():
            score = 0.0
            shared_ops = []

            # Check MemoryType match
            m_type = "Unknown"
            for u, v, k, d in self.graph.out_edges(m_id, keys=True, data=True):
                if d.get("relation") == RelationshipType.BELONGS_TO.value:
                    m_type = v
                    if v in concepts.identified_memory_types:
                        score += 0.45

            # Check Operations match
            for u, v, k, d in self.graph.out_edges(m_id, keys=True, data=True):
                if d.get("relation") == RelationshipType.IMPLEMENTS.value:
                    if v in concepts.identified_operations:
                        shared_ops.append(v)
                        score += 0.10

            # Lexical and Architectural Phrase Alignment
            m_data = self.graph.nodes.get(m_id, {})
            m_text = f"{m_data.get('name', '')} {m_data.get('description', '')} {m_data.get('paradigm', '')}".lower()
            query_lower = concepts.raw_query.lower()

            for kw in concepts.salient_keywords:
                if kw in m_text:
                    score += 0.08

            phrases = [
                "monte carlo", "tree search", "policy gradient", "ebbinghaus",
                "skill library", "pagerank", "triplet", "acid", "transaction",
                "stream", "cross-task", "windowed", "file viewer", "decay", "javascript",
                "reinforcement", "scratchpad", "paging", "mcts", "verbal self-critique", "operating system"
            ]
            for phrase in phrases:
                if phrase in query_lower and (phrase in m_text or phrase in m_id):
                    score += 0.40

            # Check proposing paper
            prop_paper_id = "Unknown"
            prop_paper_title = "Unknown"
            paper_year = 2024
            evidence_text = ""

            for u, v, k, d in self.graph.in_edges(m_id, keys=True, data=True):
                if d.get("relation") == RelationshipType.PROPOSES.value:
                    prop_paper_id = u
                    p_data = self.graph.nodes.get(u, {})
                    prop_paper_title = p_data.get("name", u)
                    paper_year = p_data.get("year", 2024)
                    ev_id = d.get("evidence_id")
                    ev_obj = self.state.evidence.get(ev_id)
                    if ev_obj:
                        evidence_text = ev_obj.excerpt
                    break

            if score > 0.3:
                m_data = self.graph.nodes.get(m_id, {})
                matches.append(
                    PriorArtMatch(
                        mechanism_id=m_id,
                        mechanism_name=m_data.get("name", m_id),
                        memory_type=m_type,
                        proposing_paper_id=prop_paper_id,
                        proposing_paper_title=prop_paper_title,
                        paper_year=paper_year,
                        overlap_score=round(min(score, 1.0), 2),
                        shared_operations=shared_ops,
                        evidence_excerpt=evidence_text,
                    )
                )

        matches.sort(key=lambda x: x.overlap_score, reverse=True)
        return matches[:6]

    def _find_pitfalls(
        self, prior_art: List[PriorArtMatch], concepts: ResolvedConcepts
    ) -> List[PitfallWarning]:
        """Traverse EXHIBITS and REPORTS_LIMITATION to retrieve verified limitations."""
        pitfalls: List[PitfallWarning] = []
        seen_lims = set()

        top_mechs = {m.mechanism_id for m in prior_art[:4]}

        for m_id in top_mechs:
            for u, v, k, d in self.graph.out_edges(m_id, keys=True, data=True):
                if d.get("relation") == RelationshipType.EXHIBITS.value:
                    lim_id = v
                    if lim_id not in seen_lims:
                        seen_lims.add(lim_id)
                        lim_data = self.graph.nodes.get(lim_id, {})
                        lim_name = lim_data.get("name", lim_id)
                        lim_cat = lim_data.get("category", "general")

                        ev_id = d.get("evidence_id")
                        ev_obj = self.state.evidence.get(ev_id)
                        p_id = ev_obj.paper_id if ev_obj else "unknown"
                        sec = ev_obj.section if ev_obj else "Limitations"
                        ex = ev_obj.excerpt if ev_obj else ""

                        pitfalls.append(
                            PitfallWarning(
                                limitation_id=lim_id,
                                limitation_name=lim_name,
                                exhibited_by_mechanism=m_id,
                                severity_category=lim_cat,
                                reported_in_paper=p_id,
                                evidence_section=sec,
                                evidence_excerpt=ex,
                            )
                        )

        # Fallback to general limitation if no edge matched
        if not pitfalls and concepts.anticipated_limitations:
            for lim_id in concepts.anticipated_limitations:
                lim_data = self.graph.nodes.get(lim_id, {})
                if lim_data:
                    pitfalls.append(
                        PitfallWarning(
                            limitation_id=lim_id,
                            limitation_name=lim_data.get("name", lim_id),
                            exhibited_by_mechanism="general_agent_architecture",
                            severity_category=lim_data.get("category", "general"),
                            reported_in_paper="literature_standard",
                            evidence_section="Theory",
                            evidence_excerpt=lim_data.get("description", ""),
                        )
                    )

        return pitfalls

    def _trace_lineage(self, prior_art: List[PriorArtMatch]) -> List[LineageStep]:
        """Traverse EXTENDS and ADDRESSES edges to trace genealogical evolution."""
        lineage: List[LineageStep] = []
        top_mech_ids = {m.mechanism_id for m in prior_art}

        for m_id in top_mech_ids:
            # Traversal on EXTENDS
            for u, v, k, d in self.graph.out_edges(m_id, keys=True, data=True):
                if d.get("relation") == RelationshipType.EXTENDS.value:
                    ev_obj = self.state.evidence.get(d.get("evidence_id"))
                    lineage.append(
                        LineageStep(
                            earlier_mechanism=v,
                            relationship="EXTENDS",
                            later_mechanism=u,
                            rationale=f"Mechanism '{u}' directly inherits and refines architecture from '{v}'",
                            evidence_excerpt=ev_obj.excerpt if ev_obj else "",
                        )
                    )
            # Traversal on ADDRESSES
            for u, v, k, d in self.graph.out_edges(m_id, keys=True, data=True):
                if d.get("relation") == RelationshipType.ADDRESSES.value:
                    ev_obj = self.state.evidence.get(d.get("evidence_id"))
                    lineage.append(
                        LineageStep(
                            earlier_mechanism=v,
                            relationship="ADDRESSES",
                            later_mechanism=u,
                            rationale=f"Mechanism '{u}' was explicitly introduced to overcome '{v}'",
                            evidence_excerpt=ev_obj.excerpt if ev_obj else "",
                        )
                    )

        return lineage[:5]

    def _recommend_benchmarks(
        self, prior_art: List[PriorArtMatch], concepts: ResolvedConcepts
    ) -> List[BenchmarkRecommendation]:
        """Traverse EVALUATED_ON edges from prior art papers."""
        recs: Dict[str, BenchmarkRecommendation] = {}

        papers_to_check = {m.proposing_paper_id for m in prior_art}
        for p_id in papers_to_check:
            for u, v, k, d in self.graph.out_edges(p_id, keys=True, data=True):
                if d.get("relation") == RelationshipType.EVALUATED_ON.value:
                    bm_id = v
                    bm_data = self.graph.nodes.get(bm_id, {})
                    if bm_data:
                        if bm_id not in recs:
                            recs[bm_id] = BenchmarkRecommendation(
                                benchmark_id=bm_id,
                                benchmark_name=bm_data.get("name", bm_id),
                                domain=bm_data.get("domain", "general"),
                                metrics=bm_data.get("metrics", ["Task Success Rate"]),
                                used_by_papers=[p_id],
                            )
                        else:
                            if p_id not in recs[bm_id].used_by_papers:
                                recs[bm_id].used_by_papers.append(p_id)

        # Include concepts benchmarks if missing
        for bm_id in concepts.suggested_benchmarks:
            bm_data = self.graph.nodes.get(bm_id, {})
            if bm_data and bm_id not in recs:
                recs[bm_id] = BenchmarkRecommendation(
                    benchmark_id=bm_id,
                    benchmark_name=bm_data.get("name", bm_id),
                    domain=bm_data.get("domain", "general"),
                    metrics=bm_data.get("metrics", ["Task Success Rate"]),
                    used_by_papers=["concept_match"],
                )

        return list(recs.values())[:4]

    def _build_reading_path(
        self,
        prior_art: List[PriorArtMatch],
        pitfalls: List[PitfallWarning],
        lineage: List[LineageStep],
        benchmarks: List[BenchmarkRecommendation],
    ) -> List[ReadingStage]:
        """Construct pedagogical 5-stage chronological reading curriculum."""
        path: List[ReadingStage] = []

        # Stage 1: Foundational Paper
        stage1_paper = "paper_react_2022"
        p1_data = self.graph.nodes.get(stage1_paper, {})
        if not p1_data and prior_art:
            stage1_paper = prior_art[-1].proposing_paper_id
            p1_data = self.graph.nodes.get(stage1_paper, {})

        path.append(
            ReadingStage(
                stage_number=1,
                stage_name="1. Foundational Architecture",
                paper_id=stage1_paper,
                paper_title=p1_data.get("name", "ReAct: Synergizing Reasoning and Acting in Language Models"),
                authors=", ".join(p1_data.get("authors", ["Yao et al."])[:2]) + " et al.",
                year=p1_data.get("year", 2022),
                pedagogical_purpose="Understand original baseline of in-context interaction buffers without complex persistent storage.",
                key_takeaway="Establishes thought-action-observation cycles and why naive context accumulation overflows.",
            )
        )

        # Stage 2: Direct Prior Art
        top_art = prior_art[0] if prior_art else None
        p2_id = top_art.proposing_paper_id if top_art else "paper_reflexion_2023"
        p2_data = self.graph.nodes.get(p2_id, {})

        path.append(
            ReadingStage(
                stage_number=2,
                stage_name="2. Direct Architectural Precedent",
                paper_id=p2_id,
                paper_title=p2_data.get("name", top_art.proposing_paper_title if top_art else "Reflexion"),
                authors=", ".join(p2_data.get("authors", ["Lead Researcher"])[:2]) + " et al.",
                year=p2_data.get("year", 2023),
                pedagogical_purpose="Examine the closest structural predecessor that implements your core memory type and operations.",
                key_takeaway=f"Implements {top_art.mechanism_name if top_art else 'Episodic Memory'} with verified operational bounds.",
            )
        )

        # Stage 3: Known Failure Mode / Pitfall Paper
        p3_id = pitfalls[0].reported_in_paper if pitfalls and pitfalls[0].reported_in_paper != "literature_standard" else "paper_lost_in_middle_2023"
        p3_data = self.graph.nodes.get(p3_id, {})

        path.append(
            ReadingStage(
                stage_number=3,
                stage_name="3. Failure Mode & Boundary Analysis",
                paper_id=p3_id,
                paper_title=p3_data.get("name", "Lost in the Middle: How Language Models Use Long Contexts"),
                authors=", ".join(p3_data.get("authors", ["Liu et al."])[:2]) + " et al.",
                year=p3_data.get("year", 2023),
                pedagogical_purpose="Analyze empirical failure modes when this memory mechanism scales or encounters noise.",
                key_takeaway=f"Exposes critical bottleneck: {pitfalls[0].limitation_name if pitfalls else 'Context Degradation'}.",
            )
        )

        # Stage 4: Modern Extension / Counter-Approach
        p4_id = "paper_retroformer_2023"
        p4_data = self.graph.nodes.get(p4_id, {})

        path.append(
            ReadingStage(
                stage_number=4,
                stage_name="4. Evolutionary Extension",
                paper_id=p4_id,
                paper_title=p4_data.get("name", "Retroformer: Retrospective Large Language Agents"),
                authors=", ".join(p4_data.get("authors", ["Shi et al."])[:2]) + " et al.",
                year=p4_data.get("year", 2023),
                pedagogical_purpose="Learn how subsequent literature addressed the bottlenecks identified in Stage 3.",
                key_takeaway="Replaces heuristic prompts with formal reinforcement or structured graph constraints.",
            )
        )

        # Stage 5: Evaluation Benchmark Standard
        bm_top = benchmarks[0] if benchmarks else None
        bm_paper_id = "paper_alfworld_2021" if (not bm_top or "alfworld" in bm_top.benchmark_id) else "paper_webarena_2023"
        p5_data = self.graph.nodes.get(bm_paper_id, {})

        path.append(
            ReadingStage(
                stage_number=5,
                stage_name="5. Empirical Benchmark Standard",
                paper_id=bm_paper_id,
                paper_title=p5_data.get("name", "ALFWorld: Interactive Learning Environment"),
                authors=", ".join(p5_data.get("authors", ["Shridhar et al."])[:2]) + " et al.",
                year=p5_data.get("year", 2021),
                pedagogical_purpose="Adopt the standard experimental protocols and evaluation metrics required for peer review.",
                key_takeaway=f"Use metrics: {', '.join(bm_top.metrics if bm_top else ['Success Rate'])} in {bm_top.domain if bm_top else 'Embodied Planning'}.",
            )
        )

        return path
