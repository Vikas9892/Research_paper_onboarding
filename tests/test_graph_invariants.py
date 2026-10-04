"""Unit tests for Graph Structural and Semantic Invariants."""

import json
from pathlib import Path
import pytest

from src.models.schema import (
    EntityType,
    RelationshipType,
    KnowledgeState,
    PaperEntity,
    MemoryMechanismEntity,
    RelationshipRecord,
    EvidenceEntity,
)
from src.graph.builder import AgentMemoryGraph, GraphInvariantViolation

def test_graph_invariants_pass_on_active_knowledge_state():
    root = Path(__file__).resolve().parent.parent
    kg_file = root / "knowledge" / "knowledge_state.json"
    assert kg_file.exists()

    with open(kg_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Reconstruct KnowledgeState from serialized file
    k_state = KnowledgeState(**data)
    graph_wrapper = AgentMemoryGraph(k_state)
    assert graph_wrapper.graph.number_of_nodes() > 100
    assert graph_wrapper.graph.number_of_edges() > 50

def test_detects_dangling_edge():
    k_state = KnowledgeState(
        entities={
            EntityType.PAPER.value: {
                "p1": PaperEntity(id="p1", name="P1", description="desc", year=2024, venue="ICLR", abstract="abs", lineage_group="g")
            }
        },
        relationships=[
            RelationshipRecord(
                id="r1",
                source_id="p1",
                relation=RelationshipType.PROPOSES,
                target_id="nonexistent_mech",
                evidence_id="ev1",
            )
        ],
        evidence={
            "ev1": EvidenceEntity(id="ev1", name="ev1", description="desc", paper_id="p1", section="Abstract", excerpt="excerpt text")
        }
    )
    with pytest.raises(GraphInvariantViolation, match="Dangling target ID"):
        AgentMemoryGraph(k_state)

def test_detects_domain_type_mismatch():
    k_state = KnowledgeState(
        entities={
            EntityType.PAPER.value: {
                "p1": PaperEntity(id="p1", name="P1", description="desc", year=2024, venue="ICLR", abstract="abs", lineage_group="g"),
                "p2": PaperEntity(id="p2", name="P2", description="desc", year=2024, venue="ICLR", abstract="abs", lineage_group="g")
            }
        },
        relationships=[
            # PROPOSES requires (Paper -> MemoryMechanism), but here target is Paper!
            RelationshipRecord(
                id="r1",
                source_id="p1",
                relation=RelationshipType.PROPOSES,
                target_id="p2",
                evidence_id="ev1",
            )
        ],
        evidence={
            "ev1": EvidenceEntity(id="ev1", name="ev1", description="desc", paper_id="p1", section="Abstract", excerpt="excerpt text")
        }
    )
    with pytest.raises(GraphInvariantViolation, match="Range mismatch"):
        AgentMemoryGraph(k_state)

def test_detects_circular_extends_inheritance():
    k_state = KnowledgeState(
        entities={
            EntityType.MEMORY_MECHANISM.value: {
                "m1": MemoryMechanismEntity(id="m1", name="M1", description="d1"),
                "m2": MemoryMechanismEntity(id="m2", name="M2", description="d2"),
            }
        },
        relationships=[
            RelationshipRecord(
                id="r1",
                source_id="m1",
                relation=RelationshipType.EXTENDS,
                target_id="m2",
                evidence_id="ev1",
            ),
            RelationshipRecord(
                id="r2",
                source_id="m2",
                relation=RelationshipType.EXTENDS,
                target_id="m1",
                evidence_id="ev1",
            ),
        ],
        evidence={
            "ev1": EvidenceEntity(id="ev1", name="ev1", description="desc", paper_id="p1", section="Abstract", excerpt="excerpt text")
        }
    )
    with pytest.raises(GraphInvariantViolation, match="Circular inheritance cycle detected in EXTENDS"):
        AgentMemoryGraph(k_state)
