---
title: PharmaRAG MVP
emoji: 💊
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: "1.39.0"
app_file: app.py
pinned: false
---

# PharmaRAG MVP

PharmaRAG MVP is a low-cost AI-powered Retrieval-Augmented Generation (RAG) application for pharmaceutical and regulatory documents.

## Features
- Upload pharmaceutical and regulatory PDFs
- Ingest document content into a vector database
- Perform semantic retrieval
- Ask natural-language questions
- Receive grounded answers with citations and supporting excerpts

## Tech Stack
- Python
- Streamlit
- PyPDF
- Sentence Transformers
- ChromaDB

## How to Use
1. Load bundled sample docs or upload your own PDFs
2. Click **Ingest Documents**
3. Ask a question
4. Review the answer summary, citation, supporting sources, and excerpts

## Project Purpose
This project is the general-purpose pharma and regulatory Q&A module in the broader PharmaAI platform.

## Notes
- This MVP is designed for grounded document Q&A
- It is separate from:
  - **PharmaSummarizer** → summary/highlights extraction
  - **CompliBot** → compliance/SOP/process-oriented Q&A