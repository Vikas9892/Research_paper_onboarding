"""Tests for paper ingestion and normalization."""

import json
from pathlib import Path
from src.ingestion.normalizer import normalize_paper, process_corpus, NormalizedPaper

def test_normalize_single_paper():
    sample = {
        "paper_id": "test_001",
        "title": "  Sample Agent Memory  ",
        "authors": ["Alice Doe ", " Bob Smith"],
        "year": 2024,
        "venue": "NeurIPS",
        "lineage_group": "working_context_memory",
        "abstract": "  This is an abstract.  ",
        "key_sections": {
            "Architecture": " Core memory hierarchy. "
        }
    }
    norm = normalize_paper(sample)
    assert norm.paper_id == "test_001"
    assert norm.title == "Sample Agent Memory"
    assert norm.authors == ["Alice Doe", "Bob Smith"]
    assert norm.sections["Abstract"] == "This is an abstract."
    assert norm.sections["Architecture"] == "Core memory hierarchy."
    assert "Core memory hierarchy." in norm.full_text_corpus

def test_processed_corpus_file_exists():
    root = Path(__file__).resolve().parent.parent
    proc_file = root / "data" / "processed" / "normalized_papers.json"
    assert proc_file.exists()
    
    with open(proc_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    assert len(data) == 75
    assert all("paper_id" in p for p in data)
    assert all("abstract" in p and len(p["abstract"]) > 0 for p in data)
