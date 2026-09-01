# PharmaRAG MVP Runbook

## Local Requirements

Recommended:

```text
Python 3.11
```

Create a virtual environment from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install runtime dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

For testing:

```bash
pip install -r requirements-dev.txt
```

## Runtime Directories

Create the local directories if they do not exist:

```bash
mkdir -p \
  data/input/sample_docs \
  data/uploads \
  data/runtime/vector_store \
  logs
```

These directories contain runtime/local data and are intentionally excluded from Git.

## Add Documents

For local documents, place PDF files under:

```text
data/input/sample_docs/
```

Alternatively, start the application and use its PDF uploader.

Do not commit local documents or uploaded files.

## Start the Application

From the repository root:

```bash
streamlit run app.py
```

Streamlit will display the local application URL.

## Ingest Documents

In the UI:

```text
Step 1
→ select local/sample documents or upload PDFs
→ choose Ingest Documents
```

Successful ingestion reports the number of processed PDFs and indexed chunks.

## Query Documents

After ingestion:

```text
Step 2
→ enter a question
→ choose top-k display count
→ select Ask
```

Inspect:

```text
answer summary
primary citation
supporting sources
retrieved excerpts
```

## Run Unit Tests

```bash
pytest -q
```

## Run Smoke Checks

The smoke script expects runtime directories to exist.

Create them first if necessary:

```bash
mkdir -p \
  data/input/sample_docs \
  data/uploads \
  data/runtime/vector_store
```

Then:

```bash
bash scripts/smoke_test.sh
```

## Rebuild the Vector Index

The Chroma index is derived runtime state.

If it becomes corrupted or stale:

```bash
rm -rf data/runtime/vector_store/chroma_db
```

Then restart the application and ingest the desired source documents again.

## Clean Uploaded Documents

Uploaded PDFs are runtime data.

To remove them:

```bash
rm -rf data/uploads/*
```

## Verification

Useful repository checks:

```bash
python -m compileall app.py rag_pipeline.py tests
pytest -q
git status --short
git diff --check
```

## Data Safety

Never commit:

```text
local PDFs
uploaded documents
Chroma database files
environment files containing secrets
runtime logs
```

The repository `.gitignore` is configured to exclude these areas.

## Scope

This runbook applies only to the local retrieval MVP.

Production-oriented pharma platform operations belong to the successor Pharma AI Platform repository.
