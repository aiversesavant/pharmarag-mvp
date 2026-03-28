import os
import shutil
from pathlib import Path
from typing import List

import gradio as gr

from rag_pipeline import (
    get_available_pdf_names,
    ingest_pdfs_from_folder,
    query_documents,
)

SAMPLE_DOCS_DIR = "sample_docs"
UPLOAD_DIR = "uploaded_docs"


os.makedirs(UPLOAD_DIR, exist_ok=True)


def clear_upload_dir() -> None:
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
    os.makedirs(UPLOAD_DIR, exist_ok=True)


def get_local_sample_files() -> List[str]:
    return get_available_pdf_names(SAMPLE_DOCS_DIR)


def detect_default_mode() -> str:
    sample_files = get_local_sample_files()
    if sample_files:
        return "Use bundled sample_docs"
    return "Upload PDFs"


def refresh_sample_docs():
    files = get_local_sample_files()
    if not files:
        return "No PDF files found in sample_docs."
    return f"Found {len(files)} PDF files in sample_docs."


def save_uploaded_files(files) -> str:
    if not files:
        return "Please upload one or more PDF files."

    clear_upload_dir()

    saved_count = 0
    saved_names = []

    for file_obj in files:
        src_path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
        filename = Path(src_path).name
        dst_path = os.path.join(UPLOAD_DIR, filename)
        shutil.copy(src_path, dst_path)
        saved_count += 1
        saved_names.append(filename)

    return (
        f"Saved {saved_count} PDF file(s) to {UPLOAD_DIR}.\n"
        f"Files: {', '.join(saved_names)}"
    )


def ingest_documents(mode: str, uploaded_files):
    if mode == "Use bundled sample_docs":
        status = ingest_pdfs_from_folder(SAMPLE_DOCS_DIR)
        return f"Mode: bundled sample_docs\n\n{status}"

    save_status = save_uploaded_files(uploaded_files)
    if save_status.startswith("Please upload"):
        return save_status

    ingest_status = ingest_pdfs_from_folder(UPLOAD_DIR)
    return f"Mode: uploaded PDFs\n\n{save_status}\n\n{ingest_status}"


def ask_question(question: str, top_k: int):
    result = query_documents(
        user_query=question,
        top_k=top_k,
        source_filter=None,
    )

    return (
        result["summary"],
        result["primary_citation"],
        result["supporting_sources"],
        result["raw_results"],
    )


initial_mode = detect_default_mode()
initial_sample_status = refresh_sample_docs()

with gr.Blocks(title="PharmaRAG MVP") as demo:
    gr.Markdown("# PharmaRAG MVP")
    gr.Markdown(
        "Use local bundled sample PDFs or upload your own regulatory/pharma PDFs, ingest them, and ask grounded questions."
    )

    with gr.Tab("Load Documents"):
        gr.Markdown("### Step 1: Choose document source and ingest PDFs")

        mode_radio = gr.Radio(
            choices=["Use bundled sample_docs", "Upload PDFs"],
            value=initial_mode,
            label="Document Source Mode",
        )

        sample_docs_status = gr.Textbox(
            label="Bundled sample_docs status",
            value=initial_sample_status,
            lines=3,
        )

        refresh_button = gr.Button("Refresh sample_docs status")

        pdf_input = gr.File(
            file_count="multiple",
            file_types=[".pdf"],
            label="Upload PDF files (used only in Upload PDFs mode)",
        )

        ingest_button = gr.Button("Ingest Documents")
        ingest_output = gr.Textbox(
            label="Ingestion Status",
            lines=14,
        )

    with gr.Tab("Ask Questions"):
        gr.Markdown("### Step 2: Ask questions about the ingested PDFs")

        top_k_slider = gr.Slider(
            minimum=1,
            maximum=10,
            value=3,
            step=1,
            label="Number of top chunks to retrieve",
        )

        question_input = gr.Textbox(
            label="Enter your question",
            placeholder="Example: What does ICH E2A say about serious adverse events?",
            lines=3,
        )

        ask_button = gr.Button("Ask")

        answer_summary = gr.Textbox(
            label="Answer Summary",
            lines=10,
        )

        primary_citation = gr.Textbox(
            label="Primary Citation",
            lines=2,
        )

        supporting_sources = gr.Textbox(
            label="Top Supporting Sources",
            lines=6,
        )

        raw_results = gr.Textbox(
            label="Relevant Excerpts",
            lines=20,
        )

    refresh_button.click(
        fn=refresh_sample_docs,
        outputs=[sample_docs_status],
    )

    ingest_button.click(
        fn=ingest_documents,
        inputs=[mode_radio, pdf_input],
        outputs=[ingest_output],
    )

    ask_button.click(
        fn=ask_question,
        inputs=[question_input, top_k_slider],
        outputs=[answer_summary, primary_citation, supporting_sources, raw_results],
    )

demo.launch()