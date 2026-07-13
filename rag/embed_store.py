"""
Vector store: turn chunks into vectors and support similarity search over them.

Uses sentence-transformers (all-MiniLM-L6-v2) for real embeddings, run locally
with no API key needed. Chroma stores the vectors and handles similarity search.
"""

import uuid
from typing import List, Tuple

import chromadb
from sentence_transformers import SentenceTransformer

from .ingest import Chunk

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class VectorStore:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.client = chromadb.EphemeralClient()
        # EphemeralClient shares process-wide storage, so each VectorStore
        # needs its own uniquely named collection to avoid colliding with
        # chunks from another instance (e.g. a different dataset).
        self.collection = self.client.create_collection(
            f"chunks-{uuid.uuid4().hex}", metadata={"hnsw:space": "cosine"}
        )
        self.chunks: List[Chunk] = []

    def build(self, chunks: List[Chunk]) -> None:
        """Embed all chunk text and store the resulting vectors in Chroma."""
        self.chunks = chunks
        texts = [c.text for c in chunks]
        ids = [c.chunk_id for c in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        self.collection.add(ids=ids, embeddings=embeddings.tolist(), documents=texts)

    def query(self, query_text: str, top_k: int = 3) -> List[Tuple[Chunk, float]]:
        """Return the top_k (chunk, similarity_score) pairs for a query string."""
        query_vec = self.model.encode([query_text], normalize_embeddings=True)
        result = self.collection.query(query_embeddings=query_vec.tolist(), n_results=top_k)

        chunks_by_id = {c.chunk_id: c for c in self.chunks}
        matches = []
        for chunk_id, distance in zip(result["ids"][0], result["distances"][0]):
            similarity = 1 - distance  # cosine space: distance = 1 - cosine_similarity
            matches.append((chunks_by_id[chunk_id], similarity))
        return matches
