import gradio as gr

from rag_pipeline import (
    get_available_pdf_names,
    ingest_pdfs_from_folder,
    query_documents,
)


def refresh_sample_docs():
    files = get_available_pdf_names("sample_docs")
    if not files:
        choices = ["All Documents"]
        info = "No PDF files found in sample_docs."
    else:
        choices = ["All Documents"] + files
        info = f"Found {len(files)} PDF files in sample_docs."

    return gr.update(choices=choices, value="All Documents"), info


def run_ingestion():
    status = ingest_pdfs_from_folder("sample_docs")
    files = get_available_pdf_names("sample_docs")
    choices = ["All Documents"] + files if files else ["All Documents"]
    return (
        status,
        gr.update(choices=choices, value="All Documents"),
        f"Available PDFs: {len(files)}",
    )


def ask_question(question: str, selected_source: str, top_k: int):
    source_filter = None if selected_source == "All Documents" else selected_source
    result = query_documents(
        user_query=question,
        top_k=top_k,
        source_filter=source_filter,
    )

    return (
        result["summary"],
        result["primary_citation"],
        result["supporting_sources"],
        result["raw_results"],
    )


initial_files = get_available_pdf_names("sample_docs")
initial_choices = ["All Documents"] + initial_files if initial_files else ["All Documents"]

with gr.Blocks(title="PharmaRAG MVP") as demo:
    gr.Markdown("# PharmaRAG MVP")
    gr.Markdown(
        "Upload pharma/regulatory PDFs into `sample_docs/`, ingest them, select a document if needed, and ask questions."
    )

    with gr.Tab("Ingest Documents"):
        gr.Markdown("### Step 1: Check and ingest your sample documents")

        with gr.Row():
            refresh_button = gr.Button("Refresh Sample Files")
            ingest_button = gr.Button("Ingest PDFs from sample_docs")

        sample_docs_status = gr.Textbox(
            label="Sample Docs Status",
            value=f"Available PDFs: {len(initial_files)}",
            lines=2,
        )

        ingest_output = gr.Textbox(
            label="Ingestion Status",
            lines=10,
        )

    with gr.Tab("Ask Questions"):
        gr.Markdown("### Step 2: Ask regulatory or pharmacovigilance questions")

        with gr.Row():
            source_dropdown = gr.Dropdown(
                choices=initial_choices,
                value="All Documents",
                label="Select Document Source",
            )
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
            lines=8,
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
        outputs=[source_dropdown, sample_docs_status],
    )

    ingest_button.click(
        fn=run_ingestion,
        outputs=[ingest_output, source_dropdown, sample_docs_status],
    )

    ask_button.click(
        fn=ask_question,
        inputs=[question_input, source_dropdown, top_k_slider],
        outputs=[answer_summary, primary_citation, supporting_sources, raw_results],
    )

demo.launch()