"""
RAG-Based AI Search System — programming docs and Git documentation search.

Run with:
    streamlit run app.py

Document loading, sentence-transformer embeddings, cosine-similarity retrieval,
and LLM generation (local via Ollama, or cloud via Claude/OpenAI/Gemini) wired
into a Streamlit interface.
"""
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
from rag.ingest import load_documents, build_chunk_records
from rag.embed_store import VectorStore
from rag.generate import generate_answer, PROVIDERS, AVAILABLE_MODELS

DATASETS = {
    "Programming language docs": "data/lang_docs",
    "Git documentation": "data/git_docs",
}

st.set_page_config(page_title="Dev Search", page_icon="", layout="wide")


@st.cache_resource(show_spinner="Loading and indexing documents...")
def load_store(data_folder):
    docs = load_documents(data_folder)
    chunks = build_chunk_records(docs)
    store = VectorStore()
    store.build(chunks)
    return store, docs, chunks


with st.sidebar:
    st.header("Settings")
    dataset_name = st.selectbox("Dataset", list(DATASETS.keys()))
    store, docs, chunks = load_store(DATASETS[dataset_name])

    top_k = st.slider("Number of chunks to retrieve", min_value=1, max_value=10, value=3)
    mode = st.radio("Answer mode", ["extractive", "llm"], index=0,
                     help="Extractive works with no setup. LLM mode calls the provider below.")

    provider = None
    model = None
    if mode == "llm":
        provider = st.selectbox(
            "Model provider", PROVIDERS, index=0,
            help="gemini/claude/openai need the matching API key set. ollama runs locally for free.",
        )
        model = st.selectbox("Model", AVAILABLE_MODELS[provider])

    st.divider()
    st.caption(f"Indexed **{len(docs)}** documents \u2192 **{len(chunks)}** chunks")
    with st.expander("Documents in this index"):
        for d in docs:
            st.write(f"- {d['title']}")

st.title("Dev Search")
st.caption("Ask a question about the indexed documents below.")

with st.form("search_form"):
    col_input, col_button = st.columns([5, 1])
    query = col_input.text_input(
        "Your question", placeholder="e.g. How does content-based filtering rank items?",
        label_visibility="collapsed",
    )
    search_clicked = col_button.form_submit_button("Search", type="primary", use_container_width=True)

if search_clicked and query.strip():
    spinner_msg = "Searching documents..." if mode == "extractive" else "Searching documents and generating answer..."
    with st.spinner(spinner_msg):
        retrieved = store.query(query, top_k=top_k)
        answer = generate_answer(query, retrieved, mode=mode, provider=provider, model=model)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Sources")
    for chunk, score in retrieved:
        with st.expander(f"{chunk.doc_title}  \u00b7  similarity {score:.2f}"):
            st.write(chunk.text)
elif search_clicked:
    st.warning("Type a question first.")
