"""
RAG-Based AI Search System — programming docs and Git documentation search.

Run with:
    streamlit run app.py

Document loading, sentence-transformer embeddings, cosine-similarity retrieval,
and LLM generation (local via Ollama, or cloud via Gemini) wired
into a Streamlit interface.
"""
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
from rag.ingest import load_documents, build_chunk_records
from rag.embed_store import VectorStore
from rag.generate import generate_answer, PROVIDERS, AVAILABLE_MODELS

DATA_FOLDER = "data/lang_docs"

st.set_page_config(page_title="Dev Search", page_icon="", layout="wide")

st.markdown(
    """
    <style>
    .block-container { padding-top: 2.5rem; max-width: 900px; }
    .answer-box {
        background-color: rgba(120, 120, 120, 0.08);
        border: 1px solid rgba(120, 120, 120, 0.2);
        border-radius: 10px;
        padding: 1.25rem 1.5rem;
        line-height: 1.6;
    }
    .source-title { font-weight: 600; }
    .score-track {
        background-color: rgba(120, 120, 120, 0.2);
        border-radius: 4px;
        height: 6px;
        width: 100%;
        margin-top: 4px;
        overflow: hidden;
    }
    .score-fill {
        background-color: #4c8bf5;
        height: 100%;
        border-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading and indexing documents...")
def load_store(data_folder):
    docs = load_documents(data_folder)
    chunks = build_chunk_records(docs)
    store = VectorStore()
    store.build(chunks)
    return store, docs, chunks


with st.sidebar:
    st.header("Settings")
    store, docs, chunks = load_store(DATA_FOLDER)

    top_k = st.slider("Number of chunks to retrieve", min_value=1, max_value=10, value=3)
    similarity_threshold = st.slider("Minimum match threshold", min_value=0.0, max_value=1.0, value=0.25, step=0.05, help="Increase this to strictly require exact matches. Decrease to allow looser matches.")
    mode = st.radio("Answer mode", ["extractive", "llm"], index=0,
                     help="Extractive works with no setup. LLM mode calls the provider below.")

    provider = None
    model = None
    if mode == "llm":
        provider = st.selectbox(
            "Model provider", PROVIDERS, index=0,
            help="gemini needs the matching API key set. ollama runs locally for free.",
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
        
        # Filter out irrelevant chunks to prevent hallucination
        retrieved = [(chunk, score) for chunk, score in retrieved if score >= similarity_threshold]

        answer = generate_answer(query, retrieved, mode=mode, provider=provider, model=model)

    st.subheader("Answer")
    if not retrieved:
        st.info(answer)
    elif mode == "extractive":
        for chunk, score in retrieved:
            st.markdown(
                f"""
                <div class="answer-box" style="margin-bottom: 0.75rem;">
                    <div class="source-title">{chunk.doc_title}
                        <span style="float:right; color:#4c8bf5; font-weight:500;">{score:.0%} match</span>
                    </div>
                    <div style="margin-top:0.5rem;">{chunk.text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        badge = f"`{provider} / {model}`" if provider else ""
        st.caption(badge)
        st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

    if retrieved:
        st.subheader("Sources")
        for i, (chunk, score) in enumerate(retrieved, start=1):
            pct = max(0.0, min(1.0, score))
            with st.expander(f"{i}. {chunk.doc_title}  \u00b7  {score:.0%} match"):
                st.markdown(
                    f'<div class="score-track"><div class="score-fill" style="width:{pct*100:.0f}%;"></div></div>',
                    unsafe_allow_html=True,
                )
                st.write(chunk.text)
elif search_clicked:
    st.warning("Type a question first.")
