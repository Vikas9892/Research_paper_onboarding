"""Build and export the Agent Memory Knowledge State.

Executes the full pipeline:
1. Loads normalized papers
2. Verifies evidence text
3. Constructs verified entities and relations
4. Validates graph structural invariants
5. Serializes standalone knowledge/knowledge_state.json
"""

import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.normalizer import NormalizedPaper
from src.extraction.verifier import EvidenceVerifier
from src.extraction.extractor import build_knowledge_state
from src.graph.builder import AgentMemoryGraph

def main():
    root = Path(__file__).resolve().parent.parent
    proc_file = root / "data" / "processed" / "normalized_papers.json"
    output_kg = root / "knowledge" / "knowledge_state.json"

    if not proc_file.exists():
        raise FileNotFoundError(f"Normalized papers not found at {proc_file}. Run normalizer first.")

    with open(proc_file, "r", encoding="utf-8") as f:
        raw_list = json.load(f)

    papers = [NormalizedPaper(**p) for p in raw_list]
    corpus_lookup = {p.paper_id: p for p in papers}

    print(f"Loaded {len(papers)} normalized papers. Initializing evidence verifier...")
    verifier = EvidenceVerifier(corpus_lookup)

    print("Building knowledge state with candidate verification...")
    k_state = build_knowledge_state(papers, verifier)

    print("Validating graph structural invariants...")
    graph_wrapper = AgentMemoryGraph(k_state)

    print(f"Validation successful. Exporting knowledge state to {output_kg}...")
    graph_wrapper.export_json(output_kg)

    stats = graph_wrapper.summary_stats()
    print("\n" + "=" * 50)
    print("      KNOWLEDGE STATE GENERATION SUMMARY")
    print("=" * 50)
    for k, v in stats.items():
        print(f"  {k:<30}: {v}")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    main()
