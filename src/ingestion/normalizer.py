"""Document Normalizer for Agent Memory Research Papers.

Parses raw paper entries, validates schema integrity, normalizes text content,
and structures sections into a clean representation for evidence matching.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class NormalizedPaper(BaseModel):
    paper_id: str
    title: str
    authors: List[str]
    year: int
    venue: str
    arxiv_id: Optional[str] = None
    lineage_group: str
    abstract: str
    sections: Dict[str, str] = Field(default_factory=dict)
    full_text_corpus: str = ""

def clean_text(text: str) -> str:
    """Normalize whitespace and standard formatting artifacts."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def normalize_paper(raw: Dict[str, Any]) -> NormalizedPaper:
    """Validate and normalize a raw paper record."""
    paper_id = raw["paper_id"]
    title = clean_text(raw["title"])
    authors = [clean_text(a) for a in raw.get("authors", [])]
    year = int(raw["year"])
    venue = clean_text(raw.get("venue", "Unknown"))
    arxiv_id = raw.get("arxiv_id")
    lineage_group = raw.get("lineage_group", "general")
    abstract = clean_text(raw.get("abstract", ""))

    raw_sections = raw.get("key_sections", {})
    cleaned_sections: Dict[str, str] = {}
    for sec_name, sec_content in raw_sections.items():
        cleaned_sections[clean_text(sec_name)] = clean_text(sec_content)

    # Ensure abstract is present in sections
    if "Abstract" not in cleaned_sections and abstract:
        cleaned_sections["Abstract"] = abstract

    corpus_parts = [title, abstract]
    for s_name, s_content in cleaned_sections.items():
        if s_name != "Abstract":
            corpus_parts.append(f"{s_name}: {s_content}")
    full_text_corpus = "\n".join(corpus_parts)

    return NormalizedPaper(
        paper_id=paper_id,
        title=title,
        authors=authors,
        year=year,
        venue=venue,
        arxiv_id=arxiv_id,
        lineage_group=lineage_group,
        abstract=abstract,
        sections=cleaned_sections,
        full_text_corpus=full_text_corpus,
    )

def process_corpus(raw_path: Path, output_path: Path) -> List[NormalizedPaper]:
    """Ingest raw papers and produce normalized dataset."""
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw paper file not found at: {raw_path}")

    with open(raw_path, "r", encoding="utf-8") as f:
        raw_papers = json.load(f)

    normalized: List[NormalizedPaper] = []
    seen_ids = set()

    for item in raw_papers:
        p = normalize_paper(item)
        if p.paper_id in seen_ids:
            raise ValueError(f"Duplicate paper_id found in corpus: {p.paper_id}")
        seen_ids.add(p.paper_id)
        normalized.append(p)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([p.model_dump() for p in normalized], f, indent=2, ensure_ascii=False)

    return normalized

if __name__ == "__main__":
    base = Path(__file__).resolve().parent.parent.parent
    raw_file = base / "data" / "raw" / "papers.json"
    proc_file = base / "data" / "processed" / "normalized_papers.json"
    res = process_corpus(raw_file, proc_file)
    print(f"Normalized {len(res)} papers to {proc_file}")
