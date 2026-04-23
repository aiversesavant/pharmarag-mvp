# Runbook

## Local Setup
1. Activate the local virtual environment if present
2. Install dependencies from `requirements.txt`
3. Start the application from the project root

## Operational Checks
- `app.py` exists
- `rag_pipeline.py` exists
- `data/input/sample_docs/` exists
- `data/uploads/` exists
- vector store path exists

## Recovery Notes
- if the vector store is missing, rebuild the retrieval index
- if sample docs are missing, restore bundled PDFs under `data/input/sample_docs/`
- never commit runtime data or secrets
