"""Graph Builder and Invariant Checker for Agent Memory Knowledge State.

Constructs in-memory NetworkX MultiDiGraph, executes strict invariant checks,
and exports the validated standalone knowledge_state.json file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
import networkx as nx

from src.models.schema import (
    EntityType,
    RelationshipType,
    RELATIONSHIP_CONSTRAINTS,
    KnowledgeState,
)

class GraphInvariantViolation(Exception):
    """Raised when the knowledge graph violates structural or semantic constraints."""
    pass

class AgentMemoryGraph:
    def __init__(self, knowledge_state: KnowledgeState):
        self.state = knowledge_state
        self.graph = nx.MultiDiGraph()
        self.entity_index: Dict[str, Tuple[EntityType, str]] = {}  # id -> (type, name)
        self._populate_and_validate()

    def _populate_and_validate(self):
        # 1. Register all entities as graph nodes
        for cat_name, cat_entities in self.state.entities.items():
            for ent_key, ent_obj in cat_entities.items():
                if isinstance(ent_obj, dict):
                    e_id = ent_obj["id"]
                    e_type = EntityType(ent_obj["entity_type"])
                    e_name = ent_obj.get("name", e_id)
                    e_desc = ent_obj.get("description", "")
                    payload = dict(ent_obj)
                else:
                    e_id = ent_obj.id
                    e_type = ent_obj.entity_type
                    e_name = ent_obj.name
                    e_desc = ent_obj.description
                    payload = ent_obj.model_dump()

                self.entity_index[e_id] = (e_type, e_name)
                # Unpack all properties into node attributes
                node_attrs = {
                    "entity_type": e_type.value,
                    "name": e_name,
                    "description": e_desc,
                    "payload": payload,
                }
                node_attrs.update(payload)
                self.graph.add_node(e_id, **node_attrs)

        # 2. Add edges and validate invariants
        extends_edges: List[Tuple[str, str]] = []

        for rel in self.state.relationships:
            # Check 1: No dangling source
            if rel.source_id not in self.entity_index:
                raise GraphInvariantViolation(
                    f"Dangling source ID '{rel.source_id}' in relationship {rel.id}"
                )

            # Check 2: No dangling target
            if rel.target_id not in self.entity_index:
                raise GraphInvariantViolation(
                    f"Dangling target ID '{rel.target_id}' in relationship {rel.id}"
                )

            # Check 3: Prohibit self-referencing loops
            if rel.source_id == rel.target_id:
                raise GraphInvariantViolation(
                    f"Self-referencing relationship prohibited: ({rel.source_id}) -> {rel.relation} -> ({rel.target_id})"
                )

            # Check 4: Domain & Range constraint
            expected_src_type, expected_tgt_type = RELATIONSHIP_CONSTRAINTS[rel.relation]
            actual_src_type, _ = self.entity_index[rel.source_id]
            actual_tgt_type, _ = self.entity_index[rel.target_id]

            if actual_src_type != expected_src_type:
                raise GraphInvariantViolation(
                    f"Domain mismatch for {rel.relation}: expected {expected_src_type}, got {actual_src_type} (id: {rel.source_id})"
                )
            if actual_tgt_type != expected_tgt_type:
                raise GraphInvariantViolation(
                    f"Range mismatch for {rel.relation}: expected {expected_tgt_type}, got {actual_tgt_type} (id: {rel.target_id})"
                )

            # Check 5: Evidence grounding exists
            if not rel.evidence_id or rel.evidence_id not in self.state.evidence:
                raise GraphInvariantViolation(
                    f"Relationship {rel.id} lacks valid evidence grounding (evidence_id: {rel.evidence_id})"
                )

            # Record extends for DAG check
            if rel.relation == RelationshipType.EXTENDS:
                extends_edges.append((rel.source_id, rel.target_id))

            # Add to NetworkX MultiDiGraph
            self.graph.add_edge(
                rel.source_id,
                rel.target_id,
                key=rel.id,
                relation=rel.relation.value,
                evidence_id=rel.evidence_id,
            )

        # Check 6: EXTENDS relation must form an Acyclic DAG (no circular architectural inheritance)
        extends_subgraph = nx.DiGraph()
        extends_subgraph.add_edges_from(extends_edges)
        if not nx.is_directed_acyclic_graph(extends_subgraph):
            cycle = nx.find_cycle(extends_subgraph, orientation="original")
            raise GraphInvariantViolation(f"Circular inheritance cycle detected in EXTENDS relation: {cycle}")

    def export_json(self, output_path: Path):
        """Export serialized, human-inspectable knowledge state to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dump_data = self.state.model_dump()
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dump_data, f, indent=2, ensure_ascii=False)
        print(f"Exported Knowledge State to {output_path} with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")

    def summary_stats(self) -> Dict[str, int]:
        """Return counts of entities and relations."""
        counts = {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
        }
        for e_type in EntityType:
            sub = [n for n, d in self.graph.nodes(data=True) if d.get("entity_type") == e_type.value]
            counts[f"entities_{e_type.value}"] = len(sub)
        for r_type in RelationshipType:
            edges = [(u, v) for u, v, k, d in self.graph.edges(keys=True, data=True) if d.get("relation") == r_type.value]
            counts[f"relations_{r_type.value}"] = len(edges)
        return counts
