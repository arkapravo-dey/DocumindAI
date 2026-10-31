import os
from typing import List, Dict, Any

import faiss
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

load_dotenv()

APP_TITLE = "Chat with Multiple PDFs"
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 180
TOP_K = 5

st.set_page_config(page_title=APP_TITLE, page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Loading local embedding model (first run may take a few minutes)...")
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


def extract_pdf_pages(uploaded_files) -> List[Dict[str, Any]]:
    pages = []
    for uploaded_file in uploaded_files:
        try:
            reader = PdfReader(uploaded_file)
            if reader.is_encrypted:
                try:
                    reader.decrypt("")
                except Exception:
                    st.warning(f"Skipped encrypted PDF: {uploaded_file.name}")
                    continue
            for page_number, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:
                    pages.append({
                        "text": text,
                        "source": uploaded_file.name,
                        "page": page_number,
                    })
        except Exception as exc:
            st.warning(f"Could not read {uploaded_file.name}: {exc}")
    return pages


def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = " ".join(text.split())
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + chunk_size // 2, end)
            if boundary > start:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def build_chunks(page_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    chunks = []
    for record in page_records:
        for chunk in split_text(record["text"]):
            chunks.append({
                "text": chunk,
                "source": record["source"],
                "page": record["page"],
            })
    return chunks


def build_vector_index(chunks: List[Dict[str, Any]], embedding_model):
    texts = [item["text"] for item in chunks]
    vectors = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    vectors = np.asarray(vectors, dtype=np.float32)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def retrieve(question: str, index, chunks, embedding_model, top_k: int = TOP_K):
    query_vector = embedding_model.encode(
        [question], convert_to_numpy=True, normalize_embeddings=True
    )
    query_vector = np.asarray(query_vector, dtype=np.float32)
    scores, indices = index.search(query_vector, min(top_k, len(chunks)))
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0:
            item = dict(chunks[int(idx)])
            item["score"] = float(score)
            results.append(item)
    return results


def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing. Add it to your .env file and restart Streamlit.")
    return Groq(api_key=api_key, timeout=60.0, max_retries=2)


def answer_question(question: str, client, model_name: str, retrieved_chunks, chat_history):
    context_parts = []
    for i, item in enumerate(retrieved_chunks, start=1):
        context_parts.append(
            f"[Source {i}: {item['source']}, page {item['page']}]\n{item['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    messages = [{
        "role": "system",
        "content": (
            "You are a helpful assistant that answers questions about uploaded PDF documents. "
            "Use the supplied context as the primary source. If the answer is not present in the "
            "context, clearly say that you could not find it in the uploaded documents; do not invent facts. "
            "When making factual claims from the context, cite the source using [Source N]. "
            "Be clear and concise."
        ),
    }]
    # Include a small recent conversational window for follow-up questions.
    for turn in chat_history[-6:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({
        "role": "user",
        "content": f"DOCUMENT CONTEXT:\n{context}\n\nQUESTION:\n{question}",
    })

    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.2,
        max_tokens=1200,
    )
    return completion.choices[0].message.content or "I couldn't generate an answer."


def reset_chat():
    st.session_state.messages = []


def main():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pdf_index" not in st.session_state:
        st.session_state.pdf_index = None
    if "pdf_chunks" not in st.session_state:
        st.session_state.pdf_chunks = []
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []

    with st.sidebar:
        st.header("Your documents")
        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more text-based PDF documents.",
        )
        process_clicked = st.button(
            "Process PDFs", type="primary", use_container_width=True,
            disabled=not uploaded_files,
        )
        if process_clicked:
            with st.spinner("Extracting text and building the local search index..."):
                page_records = extract_pdf_pages(uploaded_files)
                chunks = build_chunks(page_records)
                if not chunks:
                    st.error("No extractable text was found. If your PDFs are scanned images, OCR is required.")
                else:
                    try:
                        embedding_model = load_embedding_model()
                        index = build_vector_index(chunks, embedding_model)
                        st.session_state.pdf_index = index
                        st.session_state.pdf_chunks = chunks
                        st.session_state.processed_files = [f.name for f in uploaded_files]
                        st.session_state.messages = []
                        st.success(f"Processed {len(uploaded_files)} PDF(s) into {len(chunks)} searchable chunks.")
                    except Exception as exc:
                        st.error(f"Could not build the local search index: {type(exc).__name__}: {exc}")

        if st.session_state.processed_files:
            st.divider()
            st.caption("Currently indexed")
            for filename in st.session_state.processed_files:
                st.write(f"📄 {filename}")
            if st.button("Clear documents and chat", use_container_width=True):
                st.session_state.pdf_index = None
                st.session_state.pdf_chunks = []
                st.session_state.processed_files = []
                st.session_state.messages = []
                st.rerun()

        st.divider()
        st.caption(f"LLM: {DEFAULT_MODEL}")
        st.caption(f"Embeddings: {EMBEDDING_MODEL} (local)")

    st.title("📚 Chat with Multiple PDFs")
    st.caption("Upload documents, build a local search index, and ask questions grounded in their content.")

    if st.session_state.pdf_index is None:
        st.info("Upload one or more PDFs in the sidebar and select **Process PDFs** to get started.")
        return

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("Sources"):
                    for source in message["sources"]:
                        st.markdown(f"- **{source['source']}**, page {source['page']}")

    question = st.chat_input("Ask a question about your documents...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        try:
            with st.chat_message("assistant"):
                with st.spinner("Searching your PDFs and asking Groq..."):
                    embedding_model = load_embedding_model()
                    retrieved = retrieve(
                        question,
                        st.session_state.pdf_index,
                        st.session_state.pdf_chunks,
                        embedding_model,
                    )
                    client = get_groq_client()
                    # Exclude the just-added user message from history; it is added separately.
                    history = st.session_state.messages[:-1]
                    answer = answer_question(
                        question, client, DEFAULT_MODEL, retrieved, history
                    )
                    st.markdown(answer)
                    sources = []
                    seen = set()
                    for item in retrieved:
                        key = (item["source"], item["page"])
                        if key not in seen:
                            sources.append({"source": item["source"], "page": item["page"]})
                            seen.add(key)
                    if sources:
                        with st.expander("Sources"):
                            for source in sources:
                                st.markdown(f"- **{source['source']}**, page {source['page']}")
            st.session_state.messages.append({
                "role": "assistant", "content": answer, "sources": sources
            })
        except Exception as exc:
            st.error(f"Request failed: {type(exc).__name__}: {exc}")
            if "user" in st.session_state.messages[-1]["role"]:
                st.session_state.messages.pop()


if __name__ == "__main__":
    main()
