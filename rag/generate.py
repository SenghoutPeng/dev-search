"""
Generation: turn retrieved chunks + a query into a final answer.

Two answer modes:
- "extractive": no API key or model needed, just stitches the retrieved chunks
  together. Useful for checking retrieval quality on its own.
- "llm": sends the query and retrieved chunks to a language model, which must
  answer using only that context and cite which source(s) it used.

- gemini:  Google API (needs GOOGLE_API_KEY) — default provider
- ollama:  local, free, no API key
"""

import os
from typing import List, Tuple

from .ingest import Chunk

PROVIDERS = ["gemini", "ollama"]

# Models users can pick from in the UI (dropdown-only, no free text input).
AVAILABLE_MODELS = {
    "gemini": ["gemini-flash-lite-latest", "gemini-2.0-flash", "gemini-2.5-pro"],
    "ollama": ["phi3:mini"],
}

DEFAULT_MODELS = {provider: models[0] for provider, models in AVAILABLE_MODELS.items()}

REQUIRED_ENV_VAR = {
    "gemini": "GOOGLE_API_KEY",
}

# Keeps the model acting as a document search assistant, not a general chatbot:
# it must refuse anything the retrieved context doesn't cover.
SYSTEM_PROMPT = (
    "You are a strict, robotic search assistant. Your ONLY job is to answer questions based "
    "EXCLUSIVELY on the provided <context> blocks below.\n\n"
    "CRITICAL RULES:\n"
    "1. If the answer is not explicitly stated in the <context>, you MUST reply exactly with: "
    "\"I could not find any relevant information in the documents to answer your question.\"\n"
    "2. Never guess, infer, or use outside knowledge.\n"
    "3. Always cite the source title if you provide an answer."
)


def extractive_answer(query: str, retrieved: List[Tuple[Chunk, float]]) -> str:
    if not retrieved:
        return "No relevant passages were found for that query."
    lines = [f"Top passages related to: “{query}”\n"]
    for chunk, score in retrieved:
        lines.append(f"[{chunk.doc_title}, score={score:.2f}] {chunk.text}\n")
    return "\n".join(lines)


def build_user_prompt(query: str, retrieved: List[Tuple[Chunk, float]]) -> str:
    context = "\n\n".join(f"<context>\nSource: {c.doc_title}\n{c.text}\n</context>" for c, _ in retrieved)
    return f"{context}\n\nQuestion: {query}\nAnswer:"


def call_ollama(system: str, user: str, model: str) -> str:
    import ollama

    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return ollama.chat(model=model, messages=messages, options={"temperature": 0.0})["message"]["content"]


def call_gemini(system: str, user: str, model: str) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    config = types.GenerateContentConfig(system_instruction=system)
    return client.models.generate_content(model=model, contents=user, config=config).text


PROVIDER_CALLS = {
    "ollama": call_ollama,
    "gemini": call_gemini,
}


def llm_answer(query: str, retrieved: List[Tuple[Chunk, float]], provider: str = "gemini", model: str = None) -> str:
    if not retrieved:
        return "No relevant passages were found for that query."

    env_var = REQUIRED_ENV_VAR.get(provider)
    if env_var and not os.environ.get(env_var):
        return f"[{provider} not configured] Set {env_var} to use this provider."

    model = model or DEFAULT_MODELS[provider]
    user_prompt = build_user_prompt(query, retrieved)

    try:
        return PROVIDER_CALLS[provider](SYSTEM_PROMPT, user_prompt, model)
    except Exception as e:
        return f"[{provider} call failed: {e}]"


def generate_answer(
    query: str,
    retrieved: List[Tuple[Chunk, float]],
    mode: str = "extractive",
    provider: str = "gemini",
    model: str = None,
) -> str:
    if mode == "llm":
        return llm_answer(query, retrieved, provider=provider, model=model)
    return extractive_answer(query, retrieved)
