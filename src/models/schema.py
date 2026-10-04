"""Core Domain Schema and Pydantic Models for Agent Memory Knowledge Graph.

Strictly defines the 7 core entities and typed relationships.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

class EntityType(str, Enum):
    PAPER = "paper"
    MEMORY_TYPE = "memory_type"
    MEMORY_MECHANISM = "memory_mechanism"
    MEMORY_OPERATION = "memory_operation"
    LIMITATION = "limitation"
    BENCHMARK = "benchmark"
    EVIDENCE = "evidence"

class RelationshipType(str, Enum):
    PROPOSES = "PROPOSES"                       # Paper -> MemoryMechanism
    EVALUATED_ON = "EVALUATED_ON"               # Paper -> Benchmark
    REPORTS_LIMITATION = "REPORTS_LIMITATION"   # Paper -> Limitation
    BELONGS_TO = "BELONGS_TO"                   # MemoryMechanism -> MemoryType
    IMPLEMENTS = "IMPLEMENTS"                   # MemoryMechanism -> MemoryOperation
    ADDRESSES = "ADDRESSES"                     # MemoryMechanism -> Limitation
    EXHIBITS = "EXHIBITS"                       # MemoryMechanism -> Limitation
    EXTENDS = "EXTENDS"                         # MemoryMechanism -> MemoryMechanism

# Valid domain and range constraints for all relationship types
RELATIONSHIP_CONSTRAINTS = {
    RelationshipType.PROPOSES: (EntityType.PAPER, EntityType.MEMORY_MECHANISM),
    RelationshipType.EVALUATED_ON: (EntityType.PAPER, EntityType.BENCHMARK),
    RelationshipType.REPORTS_LIMITATION: (EntityType.PAPER, EntityType.LIMITATION),
    RelationshipType.BELONGS_TO: (EntityType.MEMORY_MECHANISM, EntityType.MEMORY_TYPE),
    RelationshipType.IMPLEMENTS: (EntityType.MEMORY_MECHANISM, EntityType.MEMORY_OPERATION),
    RelationshipType.ADDRESSES: (EntityType.MEMORY_MECHANISM, EntityType.LIMITATION),
    RelationshipType.EXHIBITS: (EntityType.MEMORY_MECHANISM, EntityType.LIMITATION),
    RelationshipType.EXTENDS: (EntityType.MEMORY_MECHANISM, EntityType.MEMORY_MECHANISM),
}

# --- Entity Definitions ---

class BaseEntity(BaseModel):
    id: str
    entity_type: EntityType
    name: str
    description: str

class PaperEntity(BaseEntity):
    entity_type: EntityType = EntityType.PAPER
    authors: List[str] = Field(default_factory=list)
    year: int
    venue: str
    arxiv_id: Optional[str] = None
    lineage_group: str
    abstract: str

class MemoryTypeEntity(BaseEntity):
    entity_type: EntityType = EntityType.MEMORY_TYPE

class MemoryMechanismEntity(BaseEntity):
    entity_type: EntityType = EntityType.MEMORY_MECHANISM
    paradigm: str = "in_context"

class MemoryOperationEntity(BaseEntity):
    entity_type: EntityType = EntityType.MEMORY_OPERATION
    stage: str = "execution"

class LimitationEntity(BaseEntity):
    entity_type: EntityType = EntityType.LIMITATION
    category: str = "architectural"

class BenchmarkEntity(BaseEntity):
    entity_type: EntityType = EntityType.BENCHMARK
    domain: str = "embodied_planning"
    metrics: List[str] = Field(default_factory=list)

class EvidenceEntity(BaseEntity):
    entity_type: EntityType = EntityType.EVIDENCE
    paper_id: str
    section: str
    excerpt: str

# --- Relationship & Knowledge State ---

class RelationshipRecord(BaseModel):
    id: str
    source_id: str
    relation: RelationshipType
    target_id: str
    evidence_id: str
    confidence: float = 1.0

class KnowledgeState(BaseModel):
    schema_version: str = "1.0.0"
    created_at: str = "2026-10-04"
    domain: str = "Agent Memory Architecture and Evolution"
    entities: Dict[str, Dict[str, BaseEntity]] = Field(default_factory=dict)
    relationships: List[RelationshipRecord] = Field(default_factory=list)
    evidence: Dict[str, EvidenceEntity] = Field(default_factory=dict)
