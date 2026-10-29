"""
rag.py - retrieval + answering logic (and a few helpers shared by the other scripts).

How a question is answered:
  1. The question is turned into an embedding (a list of numbers).
  2. It is compared with every chunk in the index (cosine similarity).
  3. The best `top_k` chunks scoring above `similarity_threshold` are kept.
     If none pass, we return a polite "cannot answer" message WITHOUT calling the model.
  4. The kept chunks are sent to Gemini, labeled with their category and source title,
     together with the system prompt and a short conversation history.
  5. We return the answer and the list of sources used.
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

# Folder that contains this file - all paths are relative to it.
ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.yaml"
PROMPT_PATH = ROOT / "prompts" / "system_prompt.txt"
INDEX_DIR = ROOT / "index"
EMBEDDINGS_FILE = INDEX_DIR / "embeddings.npy"   # one row of numbers per chunk
CHUNKS_FILE = INDEX_DIR / "chunks.json"          # text + metadata for each chunk
INFO_FILE = INDEX_DIR / "index_info.json"        # which model built the index, counts, date

# Read .env if it exists (on your computer). On Hugging Face the same names come from
# Space secrets. Existing environment variables are never overwritten.
load_dotenv(ROOT / ".env", override=False)


class MissingKeyError(Exception):
    """Raised when GEMINI_API_KEY is not set."""


class IndexMissingError(Exception):
    """Raised when the index has not been built or downloaded yet."""


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def load_config():
    """Read config.yaml into a Python dictionary."""
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def fill_owner_names(text, config):
    """Replace {owner_full_name}, {owner_first_name}, {owner_title_or_field} in a text."""
    for key in ("owner_full_name", "owner_first_name", "owner_title_or_field"):
        text = text.replace("{" + key + "}", str(config.get(key, "")))
    return text


def load_system_prompt(config):
    """Read prompts/system_prompt.txt, drop the '#' comment lines, and fill in the names."""
    lines = PROMPT_PATH.read_text(encoding="utf-8").splitlines()
    text = "\n".join(line for line in lines if not line.lstrip().startswith("#"))
    return fill_owner_names(text, config).strip()


def get_client():
    """Create the Gemini client using GEMINI_API_KEY (never printed)."""
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key or api_key.startswith("paste-your"):
        raise MissingKeyError("GEMINI_API_KEY is not set.")
    return genai.Client(api_key=api_key)


def is_retryable(error):
    """True for errors worth retrying: rate limits (429) and temporary server errors (5xx)."""
    code = getattr(error, "code", None)
    return isinstance(error, errors.APIError) and (code == 429 or (code or 0) >= 500)


def with_retry(func, max_attempts=6, first_wait=2.0):
    """Call func(); on a rate limit or server error wait and try again (2s, 4s, 8s, ...)."""
    wait = first_wait
    for attempt in range(1, max_attempts + 1):
        try:
            return func()
        except Exception as error:
            if attempt == max_attempts or not is_retryable(error):
                raise
            print(f"  API busy or rate-limited (attempt {attempt}); retrying in {wait:.0f}s...")
            time.sleep(wait)
            wait *= 2


def normalize(vectors):
    """Scale each row to length 1, so a dot product equals cosine similarity."""
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def embed_texts(client, texts, config, task_type):
    """
    Embed a list of texts with the Gemini embedding model and return a normalized numpy array.
    task_type is "RETRIEVAL_DOCUMENT" (for chunks) or "RETRIEVAL_QUERY" (for questions).
    """
    settings = {"output_dimensionality": config["embedding_dimensions"]}
    if config.get("use_task_types", True):
        settings["task_type"] = task_type

    def call(contents):
        return with_retry(lambda: client.models.embed_content(
            model=config["embedding_model"],
            contents=contents,
            config=types.EmbedContentConfig(**settings),
        ))

    response = call(texts)
    vectors = [e.values for e in response.embeddings]

    # Some multimodal embedding models merge a list of inputs into ONE embedding.
    # If that happens, fall back to embedding the texts one by one.
    if len(vectors) != len(texts):
        vectors = [call(text).embeddings[0].values for text in texts]

    return normalize(vectors)


# ---------------------------------------------------------------------------
# The index
# ---------------------------------------------------------------------------

_index_cache = {}


def load_index():
    """Load embeddings + chunk metadata from index/ (cached after the first call)."""
    if "data" not in _index_cache:
        if not (EMBEDDINGS_FILE.exists() and CHUNKS_FILE.exists()):
            raise IndexMissingError("The index has not been built yet. Run: python build_index.py")
        embeddings = np.load(EMBEDDINGS_FILE)
        chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
        info = json.loads(INFO_FILE.read_text(encoding="utf-8")) if INFO_FILE.exists() else {}
        _index_cache["data"] = (embeddings, chunks, info)
    return _index_cache["data"]


def retrieve(client, question, config):
    """Return the best matching chunks (each with a 'score'), best first, above the threshold."""
    embeddings, chunks, info = load_index()
    if info.get("embedding_model") not in (None, config["embedding_model"]):
        raise IndexMissingError("The index was built with a different embedding model. Rebuild it.")

    query_vector = embed_texts(client, [question], config, "RETRIEVAL_QUERY")[0]
    scores = embeddings @ query_vector                 # cosine similarity with every chunk
    best = np.argsort(scores)[::-1][: config["top_k"]]  # positions of the top_k scores

    results = []
    for i in best:
        if scores[i] >= config["similarity_threshold"]:
            results.append({**chunks[i], "score": float(scores[i])})
    return results


# ---------------------------------------------------------------------------
# Answering
# ---------------------------------------------------------------------------

def source_label(chunk):
    """Human-friendly source name, e.g. 'CV' or 'Paper — Protein Folding'."""
    if chunk["category"] == "Paper":
        return f"Paper — {chunk['source']}"
    if chunk["category"] == "CV":
        return "CV"
    return f"Personal Bio — {chunk['source']}"


def build_prompt(question, chunks):
    """Put the labeled excerpts and the question into one message for the model."""
    parts = ["SOURCE EXCERPTS (the only information you may use):\n"]
    for n, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[Excerpt {n}] Category: {chunk['category']} | Source: {chunk['source']}\n"
            f"{chunk['text']}\n"
        )
    parts.append(f"VISITOR QUESTION:\n{question}")
    return "\n".join(parts)


def history_to_contents(history, config):
    """
    Convert recent chat history into Gemini's format.
    history is a list of {"role": "user"/"assistant", "content": "..."} dictionaries.
    Only the last `max_history_turns` question/answer pairs are kept.
    """
    keep = config.get("max_history_turns", 3) * 2
    contents = []
    for message in (history or [])[-keep:]:
        text = (message.get("content") or "").strip()
        if not text:
            continue
        role = "user" if message.get("role") == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part.from_text(text=text)]))
    return contents


def answer_question(question, history=None, config=None):
    """
    Answer one visitor question.
    Returns a dictionary: {"answer": str, "sources": [str, ...]}
    """
    config = config or load_config()
    client = get_client()
    question = question.strip()

    # Always retrieve fresh sources for the new question.
    chunks = retrieve(client, question, config)

    # Follow-up questions like "what about the second one?" may not match anything alone.
    # In that case, try again with the previous visitor question added for context.
    if not chunks and history:
        previous = [m.get("content", "") for m in history if m.get("role") == "user"]
        if previous:
            chunks = retrieve(client, previous[-1] + "\n" + question, config)

    if not chunks:
        return {"answer": fill_owner_names(config["no_answer_message"], config), "sources": []}

    contents = history_to_contents(history, config)
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=build_prompt(question, chunks))]))

    response = with_retry(lambda: client.models.generate_content(
        model=config["chat_model"],
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=load_system_prompt(config),
            temperature=config["temperature"],
        ),
    ), max_attempts=3)

    answer = (response.text or "").strip() or "I'm sorry, I'm not able to answer that question."

    # Unique source labels, in the order they were retrieved.
    sources = list(dict.fromkeys(source_label(c) for c in chunks))
    return {"answer": answer, "sources": sources}
