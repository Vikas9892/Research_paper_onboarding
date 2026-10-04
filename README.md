# Agent Memory Research Intelligence & Decision Reasoning Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/pytest-15%20passed-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An evidence-backed research intelligence engine that transforms unstructured literature on **LLM Agent Memory (2021–2026)** into an inspectable, structured knowledge graph. When an AI researcher enters an unseen memory architecture proposal, the system reasons over verified multi-hop graph paths to discover architectural overlap, surface known failure modes with verbatim evidence quotes, align with standard evaluation benchmarks, and generate a 5-stage pedagogical reading path.

Built for the **Calyb AI Engineering Intern Assignment (Domain B: Research Paper Onboarding)**.

---

## Key Features

- **Evidence-Backed Knowledge State**: Every relationship edge (`PROPOSES`, `ADDRESSES`, `EXHIBITS`, `EXTENDS`, `EVALUATED_ON`) is grounded in verifiable text excerpts from source papers.
- **Zero Black-Box Extraction**: No LangChain `GraphIndex`, LlamaIndex KG, or opaque auto-extractors. Entity definitions, relationship schemas, and candidate verification logic belong 100% to this repository.
- **Standalone Inspectable Artifact**: `knowledge/knowledge_state.json` is a serialized, self-contained graph representation that can be inspected without executing any code.
- **Reasoning Over Unseen Proposals**: Deconstructs novel user ideas into ontological concepts, traverses multi-hop graph paths, and outputs actionable intelligence.
- **Interactive CLI & Evaluator**: Rich terminal interface with interactive query mode, standalone inspection mode, and an automated 15-case unseen evaluation benchmark suite.

---

## Repository Structure

```
Research_paper_onboarding/
├── data/
│   ├── raw/
│   │   └── papers.json                  # Curated metadata & sections for 75 papers
│   └── processed/
│       └── normalized_papers.json       # Cleaned, validated paper corpus
├── knowledge/
│   └── knowledge_state.json             # Serialized, standalone knowledge graph (118 nodes, 135 edges)
├── src/
│   ├── models/
│   │   └── schema.py                    # Strict Pydantic models for 7 entities & 8 typed relations
│   ├── ingestion/
│   │   ├── normalizer.py                # Document cleaning & corpus normalization
│   │   └── seed_corpus.py               # Corpus definition across 6 evolutionary lineages
│   ├── extraction/
│   │   ├── verifier.py                  # Ground-truth evidence verification engine
│   │   └── extractor.py                 # Candidate mapping & verified knowledge builder
│   ├── graph/
│   │   └── builder.py                   # NetworkX MultiDiGraph builder & invariant validator
│   └── reasoning/
│       ├── concept_resolver.py          # Deconstructs unseen proposals into ontology concepts
│       ├── engine.py                    # Multi-hop graph traversal & 5-stage reading path planner
│       └── evaluator.py                 # 15-case unseen proposal evaluation benchmark
├── app/
│   └── cli.py                           # Interactive terminal interface & scorecard viewer
├── tests/
│   ├── test_schema.py                   # Schema models & domain/range constraint tests
│   ├── test_ingestion.py                # Normalization & ingestion tests
│   ├── test_verifier.py                 # Verbatim matching & hallucination rejection tests
│   ├── test_graph_invariants.py         # Dangling edges, type constraints & DAG cycle tests
│   ├── test_reasoning.py                # Online reasoning & concept resolution tests
│   └── test_eval_suite.py               # Quantitative benchmark suite test
├── approach.md                          # Mandatory deep-dive: design decisions & trade-offs
├── pyproject.toml                       # Package configuration (pip install -e .)
└── requirements.txt                     # Minimal, explicit dependencies
```

---

## Quickstart Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Vikas9892/Research_paper_onboarding.git
cd Research_paper_onboarding
```

### 2. Install Dependencies
This project uses pure Python standard libraries alongside minimal, robust dependencies (`pydantic`, `networkx`, `rich`, `pytest`).

```bash
pip install -e .
```
*(Or `pip install -r requirements.txt`)*

### 3. Environment Variables & Configuration
**No external API keys, tokens, or environment variables are required.** The entire reasoning engine, graph traversals, and evidence verification run completely locally and offline using the serialized `knowledge/knowledge_state.json` file.

---

## How to Run & Use the System

### 1. Run the Interactive CLI (New Research Input)
To submit an unseen proposal interactively:
```bash
python -m app.cli
```
You will be prompted:
```
Enter research idea / proposal:
> I want an agent that logs verbal self-reflections after failing tasks in simulated household environments and retrieves past lessons before taking new actions.
```

### 2. Run a One-Liner Query via CLI Flag
```bash
python -m app.cli --query "I want to build an operating system memory manager for LLMs that pages context between active RAM and disk archival storage."
```

### 3. Inspect the Active Knowledge State
View structural invariants, node counts, and relationship distributions:
```bash
python -m app.cli --inspect
```

### 4. Run the 15-Case Unseen Evaluation Benchmark
Run the automated evaluation suite to verify generalization performance:
```bash
python -m app.cli --eval
```

Expected Scorecard:
```
      Unseen Research Proposal Evaluation Scorecard      
+-------------------------------------------------------+
| Evaluation Metric                 | Performance Score |
|-----------------------------------+-------------------|
| Total Unseen Evaluation Proposals |                15 |
| Concept Resolution Precision      |             86.7% |
| Prior Art Recall@1                |             73.3% |
| Prior Art Recall@3                |             86.7% |
| Pitfall Identification Rate       |             86.7% |
| Benchmark Alignment Rate          |            100.0% |
| Reading Path Completeness         |            100.0% |
+-------------------------------------------------------+
```

### 5. Export Report to JSON
```bash
python -m app.cli --query "My research proposal..." --json-out proposal_report.json
```

---

## How to Regenerate the Knowledge State from Scratch

If you modify paper metadata or wish to execute the complete verification pipeline from scratch:

```bash
# Step 1: Normalize raw papers
python src/ingestion/normalizer.py

# Step 2: Extract, verify evidence, validate graph invariants, and serialize knowledge_state.json
python scripts/build_knowledge.py
```

---

## Running the Automated Test Suite

Run the full pytest suite (15 unit & integration tests):
```bash
pytest
```
All tests execute in under 1 second.

---

## Author
**Vikas Tiwari**  
Email: [vikast4843@gmail.com](mailto:vikast4843@gmail.com)  
GitHub: [@Vikas9892](https://github.com/Vikas9892)
