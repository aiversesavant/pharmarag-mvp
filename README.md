# PharmaRAG MVP

PharmaRAG MVP is a low-cost AI-powered Retrieval-Augmented Generation (RAG) application for pharmaceutical and regulatory documents.

It allows users to:
- ingest pharma/regulatory PDFs
- perform semantic retrieval over document content
- ask natural-language questions
- receive grounded answers with citations and supporting excerpts

## Features

- PDF ingestion from `sample_docs/`
- sentence-based chunking
- sentence-transformer embeddings
- ChromaDB vector storage
- hybrid retrieval and reranking
- Gradio UI
- citation-backed responses

## Tech Stack

- Python
- Gradio
- PyPDF
- Sentence Transformers
- ChromaDB

## Run Locally

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py