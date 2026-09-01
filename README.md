---

title: PharmaRAG MVP
emoji: 💊
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: "1.39.0"
app_file: app.py
pinned: false
-------------

# PharmaRAG MVP

PharmaRAG MVP is an early pharmaceutical and regulatory document-retrieval prototype built with local embeddings, ChromaDB, heuristic reranking, and extractive answer synthesis.

The repository is retained as an **evolution/reference project**. Its broader successor is **Pharma AI Platform**, which extends the retrieval concepts into a larger modular system with additional grounded-answer, governance, evaluation, and operational capabilities.

## Purpose

The project demonstrates a lightweight local workflow for asking questions over pharmaceutical and regulatory PDF documents without requiring a hosted LLM API.

The core pipeline is:

```text
PDF documents
      ↓
text extraction
      ↓
sentence-aware chunking
      ↓
SentenceTransformer embeddings
      ↓
ChromaDB vector retrieval
      ↓
heuristic reranking
      ↓
extractive answer selection
      ↓
source + chunk citations
```

## Implemented Capabilities

### PDF ingestion

The application can:

* discover local PDF documents
* accept uploaded PDF files through Streamlit
* extract text with PyPDF
* skip files without extractable text

Runtime uploads and local document collections are intentionally excluded from Git.

### Chunking

Extracted text is:

* whitespace-normalized
* split into sentences
* grouped into approximately 900-character chunks
* overlapped by two sentences between neighboring chunks

This preserves limited neighboring context without indexing entire documents as single records.

### Embeddings

The application uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Embeddings are generated locally through Sentence Transformers.

### Vector storage and retrieval

ChromaDB provides the persistent local vector store.

For each chunk, the index records:

* chunk text
* source filename
* chunk index
* embedding vector

Queries are embedded with the same model and retrieve a larger candidate set before reranking.

### Heuristic reranking

Initial semantic-retrieval results are reranked using a combination of:

* Chroma vector distance
* query-term overlap
* pharmaceutical-domain term signals
* sentence-level relevance

This is a deterministic heuristic reranker rather than a learned cross-encoder.

### Extractive answer synthesis

This MVP does **not** use an LLM to generate answers.

Instead, it selects highly relevant sentences from retrieved chunks and constructs an extractive answer with supporting context.

This keeps the system local and makes the answer traceable to retrieved document text.

### Citations and excerpts

Results include:

* an answer summary
* a primary source citation
* source filename
* chunk index
* supporting source references
* retrieved excerpts

These references identify retrieval evidence but are not a substitute for formal page-level citation verification.

## Application

The Streamlit interface supports two document-source modes:

```text
Bundled/local sample documents
or
Uploaded PDF documents
```

Users can ingest the selected documents, enter a question, configure the number of displayed retrieval results, and inspect the resulting evidence.

## Technology

```text
Python
Streamlit
PyPDF
Sentence Transformers
ChromaDB
PyTorch
Hugging Face ecosystem
```

## Local Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install runtime dependencies:

```bash
pip install -r requirements.txt
```

Create the local runtime directories:

```bash
mkdir -p \
  data/input/sample_docs \
  data/uploads \
  data/runtime/vector_store \
  logs
```

Place PDFs under:

```text
data/input/sample_docs/
```

or upload them through the Streamlit interface.

Start the application:

```bash
streamlit run app.py
```

## Tests

Development dependencies are installed with:

```bash
pip install -r requirements-dev.txt
```

Run:

```bash
pytest -q
```

The focused test suite validates deterministic pipeline behavior without requiring a live document index.

## Runtime Data Policy

The following directories are intentionally excluded from version control:

```text
data/input/sample_docs/
data/uploads/
data/runtime/vector_store/
logs/
```

This prevents uploaded documents, local regulatory corpora, generated vector indexes, and runtime logs from becoming repository artifacts.

## Repository Structure

```text
.
├── app.py
├── rag_pipeline.py
├── requirements.txt
├── requirements-dev.txt
├── docs/
│   ├── architecture.md
│   └── runbook.md
├── scripts/
│   ├── setup.sh
│   └── smoke_test.sh
├── tests/
│   └── unit/
└── .github/
    └── workflows/
        └── ci.yml
```

## Relationship to Pharma AI Platform

This repository represents an earlier stage in the evolution of the pharmaceutical retrieval work.

PharmaRAG MVP focuses narrowly on:

```text
local PDF retrieval
+
semantic embeddings
+
ChromaDB
+
heuristic reranking
+
extractive answers
```

The later Pharma AI Platform is the primary portfolio project for broader pharmaceutical AI engineering.

This repository is intentionally kept smaller rather than being expanded into a second competing platform.

## Scope Boundaries

PharmaRAG MVP does not claim to implement:

```text
production clinical decision support
validated regulatory decision-making
page-level citation verification
BM25 or hybrid retrieval
cross-encoder reranking
GraphRAG
agentic workflows
MCP
RBAC
enterprise governance
production-scale distributed retrieval
```

The application is an engineering and retrieval prototype.

## Status

**Reference / evolution MVP — feature-frozen after cleanup and verification.**
