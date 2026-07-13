
# RAG-Based AI Search System

A Retrieval-Augmented Generation (RAG) search engine that combines semantic document retrieval with large language model-powered answer generation. Query documentation across two datasets (Git and programming frameworks) and get AI-grounded answers with citations.

## Features

- **Semantic Search**: Sentence-transformer embeddings (all-MiniLM-L6-v2) with cosine similarity retrieval via Chroma vector database
- **Multi-Dataset Support**: Switch between Git documentation (37 docs) and programming language docs (25 docs across 10 frameworks)
- **Flexible LLM Backend**: Local (Ollama) or cloud providers (Gemini, Claude, OpenAI)
- **Extractive or Generative**: View retrieved chunks directly, or ask the LLM to synthesize answers
- **Web UI**: Streamlit interface with form-based query submission and Enter-key support
- **Docker-Ready**: Containerized for easy server deployment

## Quick Start

### Local Development (without Docker)

1. **Clone and set up**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Set up API keys** (optional, for LLM modes):
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Run the app**:
   ```bash
   streamlit run app.py
   ```
   Open http://localhost:8501

### Docker (Recommended for Deployment)

#### With Docker Compose

```bash
cp .env.example .env
# Edit .env with your API keys

docker-compose up -d --build
docker-compose logs -f rag-search
```

App runs at http://localhost:8501.

#### Manual Docker Build

```bash
docker build -t rag-search:latest .
docker run -d \
  --name rag-search-app \
  -p 8501:8501 \
  --env-file .env \
  rag-search:latest
```

## Usage

### Sidebar Settings

- **Dataset**: Switch between "Git documentation" (37 docs) and "Programming language docs" (25 docs)
- **Number of chunks**: Retrieve top-k chunks (1–10)
- **Answer mode**: "Extractive" or "LLM"
- **Model provider**: Gemini (default), Ollama, Claude, or OpenAI
- **Model**: Dropdown selection (no free-text input)

### Example Queries

**Git documentation**:
- "How do I rebase a branch?"
- "What's the difference between git stash and git branch?"

**Programming language docs**:
- "How do I create middleware in Gin?"
- "How do I define relationships in GORM?"
- "How do I handle exceptions in Java?"

## Architecture

### Pipeline

```
Query → Embedding → Semantic Search (Chroma) → Retrieved Chunks
                                                    ↓
                                        ├─ Extractive: Display chunks
                                        └─ LLM: Generate grounded answer
```

### Key Components

- **`rag/ingest.py`**: Load .txt/.pdf/.md files, split into sentence-aware chunks (80 words, 20-word overlap)
- **`rag/embed_store.py`**: Vectorize with sentence-transformers, retrieve via Chroma (cosine similarity)
- **`rag/generate.py`**: Call LLM providers (Ollama, Claude, OpenAI, Gemini)
- **`app.py`**: Streamlit UI with dataset switching and form-based query

### Directory Structure

```
final/
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .env.example
├── app.py
├── requirements.txt
├── README.md
├── EVALUATION.md
├── data/
│   ├── git_docs/              # 37 Git docs (~2500 chunks)
│   └── lang_docs/             # 25 framework docs (~975 chunks)
└── rag/
    ├── ingest.py
    ├── embed_store.py
    └── generate.py
```

### Datasets

**Git Documentation** (37 docs, ~2500 chunks)
- git-add, git-commit, git-rebase, git-stash, git-branch, and 32 other subcommands

**Programming Language Docs** (25 docs, ~975 chunks)
- Go (3): Language basics, concurrency, error handling & testing
- Gin (2): Routing & middleware, handling requests
- GORM (4): Models & queries, associations, transactions, hooks
- Laravel (4): Routing, Eloquent, auth, services & views
- Fiber (3): Routing, config & validation, request handling
- Docker (4): Fundamentals, storage & networking, production
- Tailwind CSS (2): Layout & responsive, styling & theming
- Java (1): Classes, interfaces, generics, collections, exceptions
- PHP (1): Syntax, arrays, functions, OOP, exceptions
- Spring Boot (1): Getting started, REST, JPA, configuration

## LLM Providers

### Gemini (Default)
- Models: `gemini-flash-lite-latest`, `gemini-2.0-flash`, `gemini-2.5-pro`
- Get key: [Google AI Studio](https://aistudio.google.com/app/apikeys)
- Cost: Free tier available

### Ollama (Local, Free)
- Models: `phi3:mini`, `llama3.1`
- Setup: Install [Ollama](https://ollama.ai), run `ollama serve`, pull model
- Cost: Free; requires 3–4 GB RAM

### Claude (Anthropic)
- Models: `claude-sonnet-4-6`, `claude-haiku-4-5`
- Get key: [Anthropic Console](https://console.anthropic.com)
- Cost: Pay-as-you-go

### OpenAI
- Models: `gpt-4o-mini`, `gpt-4o`
- Get key: [OpenAI Platform](https://platform.openai.com/api-keys)
- Cost: Pay-as-you-go

## Troubleshooting

### "No such file or directory: 'data/lang_docs'"
- Check that data/lang_docs exists and contains .txt files
- Update DATASETS in app.py if files were moved

### "ModuleNotFoundError: No module named 'streamlit'"
- Activate venv: `source .venv/bin/activate`
- Re-run: `pip install -r requirements.txt`

### "GOOGLE_API_KEY not found" (using Gemini)
- Set in .env: `GOOGLE_API_KEY=your_key_here`
- Or export: `export GOOGLE_API_KEY=...`

### LLM calls timeout or fail
- Verify API key is valid
- Check internet connection (cloud providers)
- For Ollama: ensure `ollama serve` is running
- Check logs: `docker logs rag-search-app`

### Retrieval returns irrelevant chunks
- Rephrase your query (semantic search is sensitive to wording)
- Increase `top_k` in sidebar
- Verify correct dataset is selected

### Docker build fails
- Check Docker is installed: `docker --version`
- Verify port 8501 is free: `lsof -i :8501`
- Try: `docker-compose up -d --build --no-cache`

## Development

### Add New Documents

1. Place .txt, .pdf, or .md files in `data/lang_docs/` or `data/git_docs/`
2. App automatically re-indexes when dataset is switched
3. No restart needed

### Customize Models

Edit `rag/generate.py`:
```python
AVAILABLE_MODELS = {
    "gemini": ["gemini-flash-lite-latest", "gemini-2.0-flash", ...],
    "ollama": ["phi3:mini", "llama3.1", ...],
    ...
}
```

### Adjust Chunking

Edit `rag/ingest.py`:
```python
chunks = chunk_text(doc_text, chunk_size=80, overlap=20)  # 80 words, 20-word overlap
```

## Deployment Checklist

- [ ] Docker and Docker Compose installed
- [ ] API keys in `.env` (or environment variables)
- [ ] Run `docker-compose up -d --build`
- [ ] Test at http://localhost:8501
- [ ] Set up reverse proxy (nginx) if needed for SSL
- [ ] Monitor logs: `docker-compose logs -f`
- [ ] Backup `data/` folder if adding custom documents

## Performance Notes

- **First query**: ~2–5 seconds (indexing + LLM call)
- **Subsequent queries**: ~0.5–1.5 seconds (cached)
- **Memory**: ~500 MB base (Streamlit + embeddings)

## License

Educational project for CS 382 (Search Engines and Information Retrieval).

## Credits

- Embeddings: [sentence-transformers](https://www.sbert.net/)
- Vector DB: [Chroma](https://www.trychroma.com/)
- Web UI: [Streamlit](https://streamlit.io/)
- LLMs: Gemini, Claude, OpenAI, Ollama

