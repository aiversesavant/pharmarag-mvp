import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "pharmarag_docs"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


_embedding_model: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    """Lazy-load the embedding model."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def get_available_pdf_names(folder_path: str = "sample_docs") -> List[str]:
    folder = Path(folder_path)
    if not folder.exists():
        return []

    return sorted(
        [
            f.name
            for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() == ".pdf" and not f.name.startswith(".")
        ]
    )


def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        reader = PdfReader(pdf_path)
        text_parts: List[str] = []

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)

        return "\n".join(text_parts).strip()
    except Exception as exc:
        print(f"Error reading PDF {pdf_path}: {exc}")
        return ""


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_into_sentences(text: str) -> List[str]:
    text = normalize_whitespace(text)
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def chunk_text(text: str, chunk_size: int = 900, overlap_sentences: int = 2) -> List[str]:
    text = normalize_whitespace(text)
    if not text:
        return []

    sentences = split_into_sentences(text)
    if not sentences:
        return []

    chunks: List[str] = []
    current_chunk: List[str] = []
    current_length = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        sentence_length = len(sentence)

        if current_chunk and current_length + sentence_length > chunk_size:
            chunks.append(" ".join(current_chunk).strip())

            overlap = (
                current_chunk[-overlap_sentences:]
                if len(current_chunk) >= overlap_sentences
                else current_chunk[:]
            )
            current_chunk = overlap[:]
            current_length = sum(len(s) for s in current_chunk) + max(len(current_chunk) - 1, 0)

        current_chunk.append(sentence)
        current_length += sentence_length + 1

    if current_chunk:
        chunks.append(" ".join(current_chunk).strip())

    return chunks


def get_client():
    return chromadb.PersistentClient(path=CHROMA_PATH)


def get_collection():
    client = get_client()
    return client.get_or_create_collection(name=COLLECTION_NAME)


def reset_collection():
    client = get_client()
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
    return client.get_or_create_collection(name=COLLECTION_NAME)


def ingest_pdfs_from_folder(folder_path: str = "sample_docs") -> str:
    folder = Path(folder_path)

    if not folder.exists():
        return f"Folder not found: {folder_path}"

    pdf_files = sorted(
        [
            f
            for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() == ".pdf" and not f.name.startswith(".")
        ]
    )

    if not pdf_files:
        return f"No PDF files found in {folder_path}."

    collection = reset_collection()

    total_chunks = 0
    processed_files = 0
    skipped_files: List[str] = []

    model = get_embedding_model()

    for pdf_file in pdf_files:
        print(f"Processing: {pdf_file.name}")

        text = extract_text_from_pdf(str(pdf_file))
        if not text:
            skipped_files.append(f"{pdf_file.name} (no extractable text)")
            continue

        chunks = chunk_text(text)
        if not chunks:
            skipped_files.append(f"{pdf_file.name} (no chunks created)")
            continue

        try:
            embeddings = model.encode(chunks).tolist()
        except Exception as exc:
            skipped_files.append(f"{pdf_file.name} (embedding failed: {exc})")
            continue

        ids = [f"{pdf_file.stem}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": pdf_file.name, "chunk_index": i} for i in range(len(chunks))]

        try:
            collection.upsert(
                ids=ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            total_chunks += len(chunks)
            processed_files += 1
        except Exception as exc:
            skipped_files.append(f"{pdf_file.name} (upsert failed: {exc})")

    message = [
        "Ingestion complete.",
        f"Processed PDFs: {processed_files}",
        f"Total chunks added: {total_chunks}",
    ]

    if skipped_files:
        message.append("\nSkipped files:")
        message.extend(skipped_files)

    return "\n".join(message)


def is_definition_question(query: str) -> bool:
    query = query.lower()
    return any(
        phrase in query
        for phrase in [
            "what is",
            "define",
            "definition",
            "meaning of",
            "what does",
        ]
    )


def is_definition_sentence(sentence: str) -> bool:
    s = sentence.lower()
    return (
        " is " in s
        or " refers to " in s
        or " is defined as " in s
        or " means " in s
    )


def extract_query_terms(user_query: str) -> List[str]:
    query_lower = user_query.lower()

    phrase_candidates = [
        "pharmacovigilance system master file",
        "serious adverse event",
        "adverse event",
        "pharmacovigilance",
        "master file",
        "sop",
        "capa",
        "deviation",
        "compliance",
        "quality system",
        "safety communication",
        "good clinical practice",
        "qppv",
        "regulatory",
        "inspection",
    ]

    matched_phrases = [p for p in phrase_candidates if p in query_lower]
    query_words = [w for w in re.findall(r"\b\w+\b", query_lower) if len(w) > 3]

    combined = matched_phrases + query_words

    deduped: List[str] = []
    seen = set()
    for item in combined:
        if item not in seen:
            deduped.append(item)
            seen.add(item)

    return deduped


def keyword_overlap_score(text: str, user_query: str) -> int:
    text_lower = text.lower()
    query_terms = extract_query_terms(user_query)

    score = 0

    for term in query_terms:
        if term in text_lower:
            score += 8 if " " in term else 3

    if "pharmacovigilance system master file" in user_query.lower():
        if "pharmacovigilance system master file" in text_lower or "psmf" in text_lower:
            score += 20

    if "serious adverse event" in user_query.lower():
        if "serious adverse event" in text_lower:
            score += 20

    if is_definition_question(user_query) and is_definition_sentence(text):
        score += 10

    return score


def sentence_relevance_score(sentence: str, user_query: str) -> int:
    s = sentence.lower()
    score = keyword_overlap_score(sentence, user_query)

    pharma_keywords = [
        "pharmacovigilance",
        "adverse event",
        "system",
        "safety",
        "guideline",
        "compliance",
        "risk",
        "inspection",
        "quality",
        "master file",
        "serious adverse event",
        "capa",
        "deviation",
        "sop",
        "qppv",
        "regulatory",
    ]

    for keyword in pharma_keywords:
        if keyword in s:
            score += 1

    return score


def rerank_retrieved_chunks(
    documents: List[str],
    metadatas: List[dict],
    distances: Optional[List[float]],
    user_query: str,
) -> Tuple[List[str], List[dict]]:
    candidates = []

    for idx, (doc, meta) in enumerate(zip(documents, metadatas)):
        distance = distances[idx] if distances and idx < len(distances) else None

        vector_score = 0.0
        if distance is not None:
            vector_score = max(0.0, 10.0 - float(distance))

        keyword_score = float(keyword_overlap_score(doc, user_query))
        sentence_scores = [sentence_relevance_score(s, user_query) for s in split_into_sentences(doc)]
        best_sentence_score = float(max(sentence_scores)) if sentence_scores else 0.0

        total_score = (vector_score * 1.5) + (keyword_score * 2.0) + (best_sentence_score * 2.5)
        candidates.append((total_score, doc, meta))

    candidates.sort(key=lambda x: x[0], reverse=True)

    reranked_documents = [item[1] for item in candidates]
    reranked_metadatas = [item[2] for item in candidates]

    return reranked_documents, reranked_metadatas


def select_best_sentences(
    documents: List[str],
    user_query: str,
    max_sentences: int = 8,
) -> List[str]:
    scored_sentences: List[Tuple[int, str]] = []

    for doc in documents:
        for sentence in split_into_sentences(doc):
            score = sentence_relevance_score(sentence, user_query)
            if score > 0:
                scored_sentences.append((score, sentence))

    scored_sentences.sort(key=lambda x: x[0], reverse=True)

    selected: List[str] = []
    seen = set()

    for _, sentence in scored_sentences:
        norm = sentence.lower()
        if norm not in seen:
            selected.append(sentence)
            seen.add(norm)

        if len(selected) >= max_sentences:
            break

    return selected


def choose_main_answer(best_sentences: List[str], user_query: str) -> Optional[str]:
    if not best_sentences:
        return None

    query_lower = user_query.lower()

    if "pharmacovigilance system master file" in query_lower:
        for sentence in best_sentences:
            s = sentence.lower()
            if ("pharmacovigilance system master file" in s or "psmf" in s) and is_definition_sentence(sentence):
                return sentence

    if "serious adverse event" in query_lower:
        for sentence in best_sentences:
            s = sentence.lower()
            if "serious adverse event" in s and is_definition_sentence(sentence):
                return sentence

    if is_definition_question(user_query):
        for sentence in best_sentences:
            if is_definition_sentence(sentence):
                return sentence

    return best_sentences[0]


def compress_context_sentences(best_sentences: List[str], main_answer: str, max_context: int = 2) -> List[str]:
    context: List[str] = []

    for sentence in best_sentences:
        if sentence == main_answer:
            continue
        if sentence not in context:
            context.append(sentence)
        if len(context) >= max_context:
            break

    return context


def synthesize_definition_answer(user_query: str, best_sentences: List[str]) -> str:
    main_answer = choose_main_answer(best_sentences, user_query)
    if not main_answer:
        return "No clear answer found in the documents."

    context = compress_context_sentences(best_sentences, main_answer, max_context=2)

    answer_lines = [f"Answer:\n{main_answer}"]

    if context:
        answer_lines.append(f"Explanation:\n{' '.join(context)}")

    return "\n\n".join(answer_lines)


def synthesize_general_answer(user_query: str, best_sentences: List[str]) -> str:
    if not best_sentences:
        return "No clear answer found in the documents."

    main_answer = best_sentences[0]
    context = compress_context_sentences(best_sentences, main_answer, max_context=2)

    answer_lines = [f"Answer:\n{main_answer}"]

    if context:
        answer_lines.append(f"Supporting context:\n{' '.join(context)}")

    return "\n\n".join(answer_lines)


def build_clean_summary(user_query: str, documents: List[str]) -> str:
    best_sentences = select_best_sentences(documents, user_query, max_sentences=8)

    if not best_sentences:
        fallback = normalize_whitespace(documents[0]) if documents else ""
        if len(fallback) > 500:
            fallback = fallback[:500].rstrip() + "..."
        return fallback or "No clear answer found in the documents."

    if is_definition_question(user_query):
        return synthesize_definition_answer(user_query, best_sentences)

    return synthesize_general_answer(user_query, best_sentences)


def _format_supporting_sources(metadatas: List[dict]) -> str:
    unique_lines = []
    seen = set()

    for meta in metadatas:
        source = meta.get("source", "Unknown")
        chunk_index = meta.get("chunk_index", "N/A")
        line = f"- [{source} | chunk {chunk_index}]"

        if line not in seen:
            unique_lines.append(line)
            seen.add(line)

    return "\n".join(unique_lines)


def _format_raw_results(documents: List[str], metadatas: List[dict]) -> str:
    parts: List[str] = []

    for i, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
        source = meta.get("source", "Unknown")
        chunk_index = meta.get("chunk_index", "N/A")
        clean_doc = normalize_whitespace(doc)

        parts.append(
            f"Excerpt {i}\n"
            f"Source: {source}\n"
            f"Chunk: {chunk_index}\n"
            f"Text: {clean_doc}\n"
        )

    return "\n" + ("\n" + "-" * 80 + "\n\n").join(parts)


def _build_summary_from_results(
    documents: List[str],
    metadatas: List[dict],
    user_query: str,
    output_top_k: int = 3,
) -> Dict[str, str]:
    if not documents or not metadatas:
        return {
            "summary": "No relevant results found.",
            "primary_citation": "",
            "supporting_sources": "",
            "raw_results": "",
        }

    summary = build_clean_summary(user_query, documents)

    visible_documents = documents[:output_top_k]
    visible_metadatas = metadatas[:output_top_k]

    best_meta = visible_metadatas[0]
    source = best_meta.get("source", "Unknown")
    chunk_index = best_meta.get("chunk_index", "N/A")

    primary_citation = f"[{source} | chunk {chunk_index}]"
    supporting_sources = _format_supporting_sources(visible_metadatas)
    raw_results = _format_raw_results(visible_documents, visible_metadatas)

    return {
        "summary": summary,
        "primary_citation": primary_citation,
        "supporting_sources": supporting_sources,
        "raw_results": raw_results,
    }


def query_documents(
    user_query: str,
    top_k: int = 3,
    source_filter: Optional[str] = None,
) -> Dict[str, str]:
    if not user_query.strip():
        return {
            "summary": "Please enter a question.",
            "primary_citation": "",
            "supporting_sources": "",
            "raw_results": "",
        }

    try:
        collection = get_collection()

        if collection.count() == 0:
            return {
                "summary": "No documents are ingested yet. Please ingest PDFs first.",
                "primary_citation": "",
                "supporting_sources": "",
                "raw_results": "",
            }

        model = get_embedding_model()
        query_embedding = model.encode([user_query]).tolist()[0]

        retrieval_k = max(top_k * 4, 12)

        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": retrieval_k,
            "include": ["documents", "metadatas", "distances"],
        }

        if source_filter:
            query_kwargs["where"] = {"source": source_filter}

        results = collection.query(**query_kwargs)

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        if not documents:
            if source_filter:
                return {
                    "summary": f"No relevant results found in {source_filter}. Try all documents.",
                    "primary_citation": "",
                    "supporting_sources": "",
                    "raw_results": "",
                }

            return {
                "summary": "No relevant results found.",
                "primary_citation": "",
                "supporting_sources": "",
                "raw_results": "",
            }

        reranked_documents, reranked_metadatas = rerank_retrieved_chunks(
            documents=documents,
            metadatas=metadatas,
            distances=distances,
            user_query=user_query,
        )

        return _build_summary_from_results(
            documents=reranked_documents,
            metadatas=reranked_metadatas,
            user_query=user_query,
            output_top_k=top_k,
        )

    except Exception as exc:
        return {
            "summary": f"An error occurred while querying the documents: {exc}",
            "primary_citation": "",
            "supporting_sources": "",
            "raw_results": "",
        }