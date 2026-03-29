import os
import shutil
from pathlib import Path

import streamlit as st

from rag_pipeline import (
    get_available_pdf_names,
    ingest_pdfs_from_folder,
    query_documents,
)

SAMPLE_DOCS_DIR = "sample_docs"
UPLOAD_DIR = "uploaded_docs"


def clear_upload_dir() -> None:
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
    os.makedirs(UPLOAD_DIR, exist_ok=True)


def save_uploaded_files(uploaded_files) -> str:
    if not uploaded_files:
        return "Please upload one or more PDF files."

    clear_upload_dir()

    saved_names = []
    for uploaded_file in uploaded_files:
        save_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.read())
        saved_names.append(uploaded_file.name)

    return f"Saved {len(saved_names)} PDF file(s): {', '.join(saved_names)}"


def get_sample_docs_status() -> str:
    files = get_available_pdf_names(SAMPLE_DOCS_DIR)
    if not files:
        return "No PDF files found in sample_docs."
    return f"Found {len(files)} PDF files in sample_docs."


def ingest_documents(mode: str, uploaded_files):
    if mode == "Use bundled sample_docs":
        return ingest_pdfs_from_folder(SAMPLE_DOCS_DIR)

    save_status = save_uploaded_files(uploaded_files)
    if save_status.startswith("Please upload"):
        return save_status

    ingest_status = ingest_pdfs_from_folder(UPLOAD_DIR)
    return f"{save_status}\n\n{ingest_status}"


st.set_page_config(page_title="PharmaRAG MVP", layout="wide")

st.title("PharmaRAG MVP")
st.write(
    "Use local bundled sample PDFs or upload your own regulatory/pharma PDFs, ingest them, and ask grounded questions."
)

if "ingested" not in st.session_state:
    st.session_state.ingested = False

with st.expander("Step 1: Load Documents", expanded=True):
    sample_docs_status = get_sample_docs_status()
    st.write(f"Bundled sample_docs status: {sample_docs_status}")

    mode_options = ["Use bundled sample_docs", "Upload PDFs"]
    default_mode = "Use bundled sample_docs" if get_available_pdf_names(SAMPLE_DOCS_DIR) else "Upload PDFs"

    mode = st.radio("Document Source Mode", mode_options, index=mode_options.index(default_mode))

    uploaded_files = None
    if mode == "Upload PDFs":
        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True,
        )

    if st.button("Ingest Documents"):
        status = ingest_documents(mode, uploaded_files)
        st.text_area("Ingestion Status", value=status, height=220)
        if "Ingestion complete." in status:
            st.session_state.ingested = True

with st.expander("Step 2: Ask Questions", expanded=True):
    top_k = st.slider("Number of top chunks to retrieve", 1, 10, 3)

    question = st.text_area(
        "Enter your question",
        placeholder="Example: What does ICH E2A say about serious adverse events?",
        height=100,
    )

    if st.button("Ask"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            result = query_documents(
                user_query=question,
                top_k=top_k,
                source_filter=None,
            )

            st.subheader("Answer Summary")
            st.write(result["summary"])

            st.subheader("Primary Citation")
            st.code(result["primary_citation"] or "No citation")

            st.subheader("Top Supporting Sources")
            st.text(result["supporting_sources"] or "No supporting sources")

            st.subheader("Relevant Excerpts")
            st.text(result["raw_results"] or "No excerpts")