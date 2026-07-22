# RAG-Based AI Search System

This project is a Retrieval-Augmented Generation (RAG) search system for backend development knowledge. It covers Go, Gin, GORM, PHP, Laravel, Java, Spring Boot, and common backend topics such as databases, Docker, software architecture, and testing. The system first searches for the most relevant documents using semantic search, then either shows the retrieved text directly or generates an answer with an LLM while citing the source documents.

## Features

- **Semantic Search** – Uses the `all-MiniLM-L6-v2` embedding model with Chroma vector database and cosine similarity to find related documents.
- **Focused Knowledge Base** – Contains 61 documents with around 1,436 text chunks. The dataset is focused on backend development so the answers stay relevant.
- **Multiple LLM Providers** – Supports Ollama, and Gemini.
- **Two Answer Modes**
  - **Extractive Mode** – Shows the retrieved document chunks directly.
  - **Generative Mode** – Lets an LLM create an answer using only the retrieved information.
- **Simple Web Interface** – Built with Streamlit and supports Enter key submission.
- **Docker Support** – Can be deployed easily using Docker or Docker Compose.

---

# Quick Start

## Run Locally

1. Create a virtual environment and install dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. (Optional) Add your API keys.

```bash
cp .env.example .env
```

Edit `.env` if you want to use cloud LLM providers.

3. Start the application.

```bash
streamlit run app.py
```

Open:

```
http://localhost:8501
```

---

## Run with Docker

### Docker Compose

```bash
cp .env.example .env
docker-compose up -d --build
docker-compose logs -f rag-search
```

Open:

```
http://localhost:8501
```

### Manual Docker

```bash
docker build -t rag-search:latest .

docker run -d \
  --name rag-search-app \
  -p 8501:8501 \
  --env-file .env \
  rag-search:latest
```

---

# System Architecture

## Pipeline

```
User Query -> Create Query Embedding -> Search Chroma Vector Database -> Retrieve Relevant Chunks
```

---

## Main Components

### `rag/ingest.py`

- Reads `.txt`, `.pdf`, and `.md` files.
- Splits documents into sentence-based chunks with about 80 words and a 20-word overlap.

### `rag/embed_store.py`

- Creates embeddings using Sentence Transformers.
- Stores vectors in Chroma.
- Performs semantic search using cosine similarity.

### `rag/generate.py`

- Connects to Gemini, or Ollama.
- Generates answers using only the retrieved context.

### `app.py`

- Streamlit user interface.
- Handles user input and displays answers and sources.

---

# Project Structure

```
dev-search/
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .env.example
├── app.py
├── requirements.txt
├── README.md
├── evaluation.pdf
├── architecture.png
├── data/
│   └── lang_docs/
└── rag/
    ├── ingest.py
    ├── embed_store.py
    └── generate.py
```

---

# Dataset

The dataset contains **61 documents** with about **1,436 chunks**.

It includes:

- Go
- Gin
- GORM
- Laravel
- PHP
- Java
- Spring Boot

It also covers common backend topics such as:

- Authentication (JWT and OAuth2)
- Databases
- SQL
- ORM concepts
- Docker
- REST APIs
- Software architecture
- Validation
- Logging
- Testing
- CI/CD

The dataset is focused on backend development instead of general programming so the retrieved results are more accurate.

---

# Design Decisions

## Embedding Model

I used **all-MiniLM-L6-v2** because it is small and fast. Larger models usually give slightly better results, but they also need more memory and take longer to run. Since this project runs on a normal CPU inside Docker because the server spec is small
.

---

## Vector Database

I chose **Chroma** because the dataset is small. There are only around 1,400 chunks, so searching through them is already very fast.

I used an **EphemeralClient**, which means the vectors are recreated whenever the application starts. Although this adds a few seconds during startup, it keeps the project simple because there is no need to manage a saved database.

---

## Chunking Method

The documents are split into **80-word chunks** with a **20-word overlap**.

Instead of cutting text at fixed character lengths, the splitter keeps sentences together. This makes each chunk easier for the embedding model to understand.

The overlap helps prevent important information from being split between two chunks.

---

## Two Answer Modes

The application has two modes.

### Extractive Mode

This mode shows the retrieved document chunks directly. It helps users see exactly what was found in the knowledge base.

### Generative Mode

This mode sends the retrieved chunks to an LLM, which writes a complete answer.

Having both modes makes it easier to compare retrieval quality with generated answers.

---

## Grounded LLM Responses

The system prompt tells the LLM to answer only using the retrieved context.

If there is not enough information, the model should say that instead of making up an answer.

The LLM also cites the document sources used in the answer.

---

## Multiple LLM Providers

The project supports:

- Gemini
- Ollama
---
# Performance

The following results were measured during local testing.

| Task | Time |
|------|------|
| At start | 15–25 seconds |
| Warm query (Extractive) | 200–400 ms |
| Warm query (LLM) | 1–4 seconds (cloud providers) |
| Ollama (CPU) | 20-30 seconds |
| Memory usage | Around 500 MB – 1 GB |

The first startup is slower because the system loads the embedding model and creates embeddings for all document chunks.

After that, searches are much faster because the model and vector database are already loaded.

---

# LLM Providers

## Gemini (Default)

**Models**

- gemini-flash-lite-latest
- gemini-2.0-flash
- gemini-2.5-pro

**API Key**

Google AI Studio

**Cost**

Free tier available.

---

## Ollama

**Models**

- phi3:mini

**Setup**

Install Ollama, start the server, and download a model.

```bash
ollama serve
ollama pull phi3:mini
```

**Cost**

Free, but requires around 3–4 GB of RAM.