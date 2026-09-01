from rag_pipeline import (
    chunk_text,
    normalize_whitespace,
    query_documents,
    rerank_retrieved_chunks,
    split_into_sentences,
)


def test_normalize_whitespace_collapses_spacing():
    text = "  Serious   adverse\n\nevent   reporting.  "

    assert normalize_whitespace(text) == "Serious adverse event reporting."


def test_split_into_sentences_returns_meaningful_sentences():
    text = (
        "A serious adverse event requires appropriate safety evaluation. "
        "Pharmacovigilance systems support ongoing safety monitoring."
    )

    sentences = split_into_sentences(text)

    assert len(sentences) == 2
    assert sentences[0].startswith("A serious adverse event")
    assert sentences[1].startswith("Pharmacovigilance systems")


def test_chunk_text_preserves_sentence_overlap():
    first = "The first regulatory sentence contains enough text for chunking."
    second = "The second regulatory sentence provides important safety context."
    third = "The third regulatory sentence continues the safety discussion."

    chunks = chunk_text(
        f"{first} {second} {third}",
        chunk_size=160,
        overlap_sentences=1,
    )

    assert len(chunks) >= 2
    assert second in chunks[0]
    assert second in chunks[1]


def test_reranker_prefers_query_specific_candidate():
    documents = [
        "This document contains general administrative information for staff.",
        (
            "A serious adverse event requires safety evaluation "
            "and appropriate regulatory reporting."
        ),
    ]

    metadatas = [
        {"source": "general.pdf", "chunk_index": 0},
        {"source": "safety.pdf", "chunk_index": 1},
    ]

    reranked_documents, reranked_metadatas = rerank_retrieved_chunks(
        documents=documents,
        metadatas=metadatas,
        distances=[0.3, 0.3],
        user_query="What is a serious adverse event?",
    )

    assert reranked_metadatas[0]["source"] == "safety.pdf"
    assert "serious adverse event" in reranked_documents[0].lower()


def test_empty_query_returns_validation_message_without_retrieval():
    result = query_documents("   ")

    assert result["summary"] == "Please enter a question."
    assert result["primary_citation"] == ""
    assert result["supporting_sources"] == ""
    assert result["raw_results"] == ""
