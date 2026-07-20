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
import time
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


# Sentence-transformer cosine similarity has a noise floor around 0.15-0.26 for
# *any* query against this corpus, even off-topic ones ("what is messi" scores
# ~0.26), while genuine on-topic matches score 0.5+. The sidebar's retrieval
# filter (default 0.25) sits too close to that floor to catch borderline noise,
# so this threshold gates whether a result is confident enough to present as an
# answer rather than just being retrieved.
CONFIDENT_MATCH_THRESHOLD = 0.35


def is_confident_match(retrieved: List[Tuple[Chunk, float]]) -> bool:
    return bool(retrieved) and retrieved[0][1] >= CONFIDENT_MATCH_THRESHOLD


def no_confident_match_message(retrieved: List[Tuple[Chunk, float]]) -> str:
    best_score = retrieved[0][1]
    return (
        f"No passage closely matches that question (best match was only {best_score:.0%} "
        "similar). This is likely outside the indexed documents."
    )


def extractive_answer(query: str, retrieved: List[Tuple[Chunk, float]]) -> str:
    if not retrieved:
        return "No relevant passages were found for that query."
    if not is_confident_match(retrieved):
        return no_confident_match_message(retrieved)
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


class RateLimitError(Exception):
    """Raised when a provider's rate limit is hit and retries are exhausted."""


GEMINI_RATE_LIMIT_MESSAGE = (
    "Gemini is rate-limited right now (too many requests). Please wait a moment "
    "and try again, or switch providers/models in the sidebar."
)

# Retry attempts + backoff (seconds) for transient Gemini rate-limit (429) responses.
RATE_LIMIT_RETRY_DELAYS = [1, 2, 4]


def call_gemini(system: str, user: str, model: str) -> str:
    from google import genai
    from google.genai import errors, types

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    config = types.GenerateContentConfig(system_instruction=system)

    for delay in [*RATE_LIMIT_RETRY_DELAYS, None]:
        try:
            return client.models.generate_content(model=model, contents=user, config=config).text
        except errors.ClientError as e:
            if e.code == 429 and delay is not None:
                time.sleep(delay)
                continue
            if e.code == 429:
                raise RateLimitError(GEMINI_RATE_LIMIT_MESSAGE) from e
            raise


PROVIDER_CALLS = {
    "ollama": call_ollama,
    "gemini": call_gemini,
}


def llm_answer(query: str, retrieved: List[Tuple[Chunk, float]], provider: str = "gemini", model: str = None) -> str:
    if not retrieved:
        return "No relevant passages were found for that query."
    if not is_confident_match(retrieved):
        return no_confident_match_message(retrieved)

    env_var = REQUIRED_ENV_VAR.get(provider)
    if env_var and not os.environ.get(env_var):
        return f"[{provider} not configured] Set {env_var} to use this provider."

    model = model or DEFAULT_MODELS[provider]
    user_prompt = build_user_prompt(query, retrieved)

    try:
        return PROVIDER_CALLS[provider](SYSTEM_PROMPT, user_prompt, model)
    except RateLimitError as e:
        return str(e)
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
