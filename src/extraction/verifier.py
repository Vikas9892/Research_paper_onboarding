"""Evidence Verification Engine.

Ensures that every candidate relationship edge is strictly grounded in verifiable
text from the original paper. Rejects any hallucinated or ungrounded assertions.
"""

from __future__ import annotations

import hashlib
import re
from typing import Dict, Optional
from src.ingestion.normalizer import NormalizedPaper
from src.models.schema import EvidenceEntity, EntityType

class EvidenceVerificationError(Exception):
    """Raised when candidate evidence fails grounding verification."""
    pass

def normalize_text_for_matching(text: str) -> str:
    """Normalize text by stripping whitespace and punctuation for robust matching."""
    text = re.sub(r"[^\w\s]", "", text.lower())
    text = re.sub(r"\s+", " ", text)
    return text.strip()

class EvidenceVerifier:
    def __init__(self, corpus: Dict[str, NormalizedPaper]):
        self.corpus = corpus

    def verify(
        self,
        paper_id: str,
        section: str,
        excerpt: str,
        min_match_len: int = 15,
    ) -> Optional[EvidenceEntity]:
        """Verify that excerpt is present in the specified paper and section.
        
        Returns an EvidenceEntity if valid, or None if rejected.
        """
        if paper_id not in self.corpus:
            return None

        paper = self.corpus[paper_id]
        clean_excerpt = excerpt.strip()
        if len(clean_excerpt) < min_match_len:
            return None

        norm_excerpt = normalize_text_for_matching(clean_excerpt)

        # Check in the requested section first
        section_text = paper.sections.get(section, "")
        norm_section = normalize_text_for_matching(section_text)

        found = norm_excerpt in norm_section

        # If not found in exact section, search entire paper text
        if not found:
            norm_full = normalize_text_for_matching(paper.full_text_corpus)
            if norm_excerpt in norm_full:
                found = True

        if not found:
            return None

        # Generate deterministic evidence ID
        hash_digest = hashlib.sha256(
            f"{paper_id}:{section}:{clean_excerpt[:60]}".encode("utf-8")
        ).hexdigest()[:12]
        evidence_id = f"ev_{paper_id}_{hash_digest}"

        return EvidenceEntity(
            id=evidence_id,
            name=f"Evidence from {paper_id} [{section}]",
            description=clean_excerpt[:100] + "...",
            paper_id=paper_id,
            section=section,
            excerpt=clean_excerpt,
        )
