# PharmaRAG MVP Architecture

## Purpose

PharmaRAG MVP is a local pharmaceutical-document retrieval prototype.

Its design intentionally remains small:

```text
Document
   ↓
Extraction
   ↓
Chunking
   ↓
Embedding
   ↓
Vector Index
   ↓
Retrieval
   ↓
Heuristic Reranking
   ↓
Extractive Answer
   ↓
Citation + Evidence
```

The repository is retained as an evolution/reference project rather than as the primary production-oriented pharma platform.

## Application Layer

`app.py` implements the Streamlit user interface.

It supports:

```text
local sample documents
uploaded documents
document ingestion
question entry
top-k result selection
answer display
source display
retrieved excerpts
```

The interface delegates ingestion and retrieval behavior to `rag_pipeline.py`.

## Document Storage

Three runtime areas are used:

```text
data/input/sample_docs/
data/uploads/
data/runtime/vector_store/
```

All are excluded from Git.

### Sample documents

`data/input/sample_docs/` is used for locally supplied pharmaceutical or regulatory PDFs.

### Uploaded documents

`data/uploads/` contains documents uploaded through Streamlit.

The application clears this directory before saving a new upload batch.

### Vector storage

`data/runtime/vector_store/chroma_db/` contains the Chroma persistent index.

The vector database is disposable derived state and can be rebuilt from source documents.

## Extraction

PDF text is extracted with PyPDF.

Pages with extractable text are concatenated into a single document string.

Extraction failures return an empty document and allow ingestion to continue with other files.

## Text normalization

Whitespace is normalized before chunking.

The pipeline splits text using sentence-ending punctuation and removes very short sentence fragments.

## Chunking

Default chunk configuration:

```text
target size: approximately 900 characters
overlap: 2 sentences
```

When a chunk reaches the configured size, the final sentences from the previous chunk seed the next chunk.

This provides limited contextual continuity between neighboring records.

## Embeddings

The embedding model is:

```text
all-MiniLM-L6-v2
```

It is loaded lazily using Sentence Transformers.

The same model is used for document chunks and user queries.

## ChromaDB

ChromaDB provides local persistent vector storage.

Collection name:

```text
pharmarag_docs
```

Each indexed record includes:

```text
document chunk
embedding
source filename
chunk index
```

Ingestion uses upsert semantics.

## Retrieval

A query is embedded with the same SentenceTransformer model.

The system retrieves more candidates than are ultimately displayed:

```text
retrieval_k = max(top_k * 4, 12)
```

This gives the reranker a larger candidate set.

The MVP can also apply an optional source metadata filter when calling the retrieval pipeline directly.

## Heuristic Reranking

The retrieved candidate chunks are scored using:

```text
vector relevance
+
keyword overlap
+
sentence-level relevance
+
pharmaceutical-domain signals
```

The algorithm is deterministic and hand-crafted.

It is not a learned reranker or cross-encoder.

## Answer Construction

No LLM is called by this MVP.

Retrieved chunks are split into sentences and scored for relevance.

The highest-ranking sentences are selected and combined into an extractive answer.

Definition-style questions receive limited rule-based treatment so definitional sentences can be preferred when available.

## Evidence Output

The pipeline returns:

```text
summary
primary citation
supporting sources
raw retrieval excerpts
```

The citation format identifies:

```text
source filename
chunk index
```

This provides evidence traceability to indexed chunks but does not represent page-level citation verification.

## Failure Behavior

The pipeline handles:

```text
missing document folders
empty indexes
PDF extraction failures
embedding failures
Chroma upsert failures
empty questions
empty retrieval results
unexpected query errors
```

The MVP favors user-readable status messages over a complex exception hierarchy.

## Security and Data Boundary

Runtime documents are not source-controlled.

Git ignores:

```text
.env
.env.local
data/input/sample_docs/
data/uploads/
data/runtime/vector_store/
logs/
```

The application does not require API credentials for its implemented retrieval path.

## Relationship to Pharma AI Platform

The architecture in this repository represents an earlier, narrower retrieval system.

The later Pharma AI Platform incorporates the pharmaceutical retrieval concept into a broader application architecture with richer answer handling and additional platform capabilities.

The MVP is therefore maintained as historical engineering evidence rather than independently expanded.

## Explicit Non-Goals

This repository does not attempt to provide:

```text
clinical recommendations
validated regulatory decisions
enterprise authorization
distributed vector infrastructure
hybrid BM25/vector retrieval
cross-encoder reranking
GraphRAG
agent orchestration
MCP
production high availability
```
