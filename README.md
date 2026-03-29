---
title: PharmaRAG MVP
emoji: 💊
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: "1.32.0"
app_file: app.py
pinned: false
---

# PharmaRAG MVP

PharmaRAG MVP is a low-cost AI-powered Retrieval-Augmented Generation (RAG) application for pharmaceutical and regulatory documents.

## Features
- Upload pharma/regulatory PDFs
- Ingest document content
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
4. Review the answer, citation, and excerpts