import os
import shutil

import streamlit as st

from rag_pipeline import (
    get_available_pdf_names,
    ingest_pdfs_from_folder,
    query_documents,
)

SAMPLE_DOCS_DIR = "sample_docs"
UPLOAD_DIR = "uploaded_docs"


def clear_upload_dir() -> None:
    """Remove and recreate the upload directory."""
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
    os.makedirs(UPLOAD_DIR, exist_ok=True)


def save_uploaded_files(uploaded_files) -> str:
    """Save uploaded PDF files locally for ingestion."""
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
    return f"Found {len(files)} PDF file(s) in sample_docs."


def ingest_documents(mode: str, uploaded_files) -> str:
    """Ingest either bundled sample docs or uploaded PDFs."""
    if mode == "Use bundled sample_docs":
        return ingest_pdfs_from_folder(SAMPLE_DOCS_DIR)

    save_status = save_uploaded_files(uploaded_files)
    if save_status.startswith("Please upload"):
        return save_status

    ingest_status = ingest_pdfs_from_folder(UPLOAD_DIR)
    return f"{save_status}\n\n{ingest_status}"


st.set_page_config(page_title="PharmaRAG MVP", page_icon="💊", layout="wide")

st.title("💊 PharmaRAG MVP")
st.write(
    "Use bundled sample PDFs or upload your own pharmaceutical/regulatory PDFs, ingest them, and ask grounded questions."
)

if "ingested" not in st.session_state:
    st.session_state.ingested = False

if "ingestion_status" not in st.session_state:
    st.session_state.ingestion_status = ""

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
        with st.spinner("Ingesting documents..."):
            status = ingest_documents(mode, uploaded_files)
            st.session_state.ingestion_status = status
            st.session_state.ingested = "Ingestion complete." in status

    if st.session_state.ingestion_status:
        st.text_area("Ingestion Status", value=st.session_state.ingestion_status, height=220)

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
            with st.spinner("Querying documents..."):
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