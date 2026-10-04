"""Unit tests for Pydantic schema models and constraints."""

import pytest
from pydantic import ValidationError
from src.models.schema import (
    EntityType,
    RelationshipType,
    RELATIONSHIP_CONSTRAINTS,
    MemoryTypeEntity,
    MemoryMechanismEntity,
    RelationshipRecord,
    EvidenceEntity,
)

def test_entity_creation():
    ent = MemoryMechanismEntity(
        id="mech_test",
        name="Test Mechanism",
        description="A test description.",
        paradigm="test_paradigm",
    )
    assert ent.id == "mech_test"
    assert ent.entity_type == EntityType.MEMORY_MECHANISM
    assert ent.paradigm == "test_paradigm"

def test_relationship_constraints_definition():
    assert RelationshipType.PROPOSES in RELATIONSHIP_CONSTRAINTS
    assert RELATIONSHIP_CONSTRAINTS[RelationshipType.PROPOSES] == (
        EntityType.PAPER,
        EntityType.MEMORY_MECHANISM,
    )
    assert RELATIONSHIP_CONSTRAINTS[RelationshipType.EXTENDS] == (
        EntityType.MEMORY_MECHANISM,
        EntityType.MEMORY_MECHANISM,
    )
    assert RELATIONSHIP_CONSTRAINTS[RelationshipType.BELONGS_TO] == (
        EntityType.MEMORY_MECHANISM,
        EntityType.MEMORY_TYPE,
    )

def test_relationship_record_validation():
    rel = RelationshipRecord(
        id="rel_001",
        source_id="paper_001",
        relation=RelationshipType.PROPOSES,
        target_id="mech_001",
        evidence_id="ev_001",
    )
    assert rel.confidence == 1.0
    assert rel.relation == RelationshipType.PROPOSES
