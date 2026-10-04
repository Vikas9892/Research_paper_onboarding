"""Unit tests for Evidence Verifier."""

from src.ingestion.normalizer import NormalizedPaper
from src.extraction.verifier import EvidenceVerifier

def test_verifier_accepts_verbatim_text():
    sample_paper = NormalizedPaper(
        paper_id="paper_test_01",
        title="Testing Memory Systems",
        authors=["Author One"],
        year=2024,
        venue="ICLR",
        lineage_group="working_context_memory",
        abstract="We demonstrate that hierarchical memory paging eliminates context overflow.",
        sections={
            "Abstract": "We demonstrate that hierarchical memory paging eliminates context overflow.",
            "Architecture": "The system divides memory into main context and external archival context.",
        },
    )
    corpus = {"paper_test_01": sample_paper}
    verifier = EvidenceVerifier(corpus)

    # Valid excerpt in Abstract
    ev = verifier.verify("paper_test_01", "Abstract", "hierarchical memory paging eliminates context overflow")
    assert ev is not None
    assert ev.paper_id == "paper_test_01"
    assert ev.section == "Abstract"

    # Valid excerpt in Architecture
    ev2 = verifier.verify("paper_test_01", "Architecture", "main context and external archival context")
    assert ev2 is not None
    assert ev2.section == "Architecture"

def test_verifier_rejects_hallucinated_text():
    sample_paper = NormalizedPaper(
        paper_id="paper_test_01",
        title="Testing Memory Systems",
        authors=["Author One"],
        year=2024,
        venue="ICLR",
        lineage_group="working_context_memory",
        abstract="We demonstrate that hierarchical memory paging eliminates context overflow.",
        sections={
            "Abstract": "We demonstrate that hierarchical memory paging eliminates context overflow.",
        },
    )
    corpus = {"paper_test_01": sample_paper}
    verifier = EvidenceVerifier(corpus)

    # Fabricated / hallucinated claim
    ev = verifier.verify("paper_test_01", "Abstract", "This paper uses quantum neuromorphic quantum memristors")
    assert ev is None

def test_verifier_rejects_nonexistent_paper():
    corpus = {}
    verifier = EvidenceVerifier(corpus)
    ev = verifier.verify("nonexistent_paper", "Abstract", "Any text excerpt here")
    assert ev is None
