# Architecture

## Purpose
PharmaRAG MVP is a retrieval-augmented question answering application for pharmaceutical and regulatory documents.

## Core Entry Points
- `app.py` — main application entry point
- `rag_pipeline.py` — ingestion, retrieval, and question-answering pipeline

## Data Areas
- `data/input/sample_docs/` — bundled sample PDFs
- `data/uploads/` — uploaded PDFs
- `data/runtime/vector_store/chroma_db/` — runtime vector storage

## Notes
This project is being normalized to the enterprise workspace standard while preserving current behavior.
