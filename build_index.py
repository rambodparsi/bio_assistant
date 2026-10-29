"""
build_index.py - turns your .txt documents into a searchable index.

Run it again whenever you add or change documents:
    python build_index.py

What it does:
  1. Reads every .txt file in data/Papers, data/CV and data/Personal_Bio
     (INSTRUCTIONS.txt files are skipped).
  2. Splits each file into overlapping chunks of paragraphs.
  3. Embeds every chunk with the Gemini embedding model.
  4. Saves index/embeddings.npy (the numbers) and index/chunks.json (text + metadata).
"""

import json
import shutil
import time
from collections import Counter
from datetime import datetime, timezone

import numpy as np

from rag import (CHUNKS_FILE, EMBEDDINGS_FILE, INDEX_DIR, INFO_FILE, ROOT,
                 embed_texts, get_client, load_config)

# Folder name -> category label used in answers and in the prompt.
CATEGORIES = {"Papers": "Paper", "CV": "CV", "Personal_Bio": "Personal_Bio"}

BATCH_SIZE = 50            # chunks per embedding request
PAUSE_BETWEEN_BATCHES = 1  # seconds; keeps us under free-tier rate limits


def read_text_file(path):
    """Read a text file, coping with the most common encodings."""
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue


def title_from_filename(path):
    """'2024_Paper_Title.txt' -> '2024 Paper Title'."""
    return path.stem.replace("_", " ").replace("-", " ").strip()


def split_long_paragraph(paragraph, size):
    """Split a paragraph longer than `size` characters into pieces at word boundaries."""
    pieces, current = [], ""
    for word in paragraph.split():
        if current and len(current) + 1 + len(word) > size:
            pieces.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        pieces.append(current)
    return pieces


def overlap_tail(text, overlap):
    """The last `overlap` characters of a text, starting at a word boundary."""
    if overlap <= 0 or len(text) <= overlap:
        return text if overlap > 0 else ""
    tail = text[-overlap:]
    return tail[tail.find(" ") + 1:] if " " in tail else tail


def chunk_text(text, size, overlap):
    """
    Group paragraphs (separated by empty lines) into chunks of at most ~`size` characters.
    Each new chunk starts with the last ~`overlap` characters of the previous one,
    so sentences near a boundary are not lost.
    """
    paragraphs = []
    for block in text.replace("\r\n", "\n").split("\n\n"):
        block = " ".join(block.split())   # tidy whitespace inside the paragraph
        if block:
            paragraphs.extend(split_long_paragraph(block, size))

    chunks, current = [], ""
    for paragraph in paragraphs:
        if current and len(current) + 2 + len(paragraph) > size:
            chunks.append(current)
            current = overlap_tail(current, overlap)
            current = f"{current}\n\n{paragraph}" if current else paragraph
        else:
            current = f"{current}\n\n{paragraph}" if current else paragraph
    if current:
        chunks.append(current)
    return chunks


def collect_chunks(config):
    """Read all documents and return (chunks, number of files per category)."""
    all_chunks, file_counts = [], Counter()
    for folder, category in CATEGORIES.items():
        folder_path = ROOT / "data" / folder
        folder_path.mkdir(parents=True, exist_ok=True)
        for path in sorted(folder_path.glob("*.txt")):
            if path.name.upper() == "INSTRUCTIONS.TXT":
                continue
            text = read_text_file(path)
            pieces = chunk_text(text, config["chunk_size"], config["chunk_overlap"])
            if not pieces:
                print(f"  (skipped empty file: {folder}/{path.name})")
                continue
            file_counts[category] += 1
            for number, piece in enumerate(pieces, start=1):
                all_chunks.append({
                    "category": category,
                    "source": title_from_filename(path),
                    "file": f"{folder}/{path.name}",
                    "chunk": number,
                    "text": piece,
                })
    return all_chunks, file_counts


def main():
    config = load_config()
    print("Reading documents...")
    chunks, file_counts = collect_chunks(config)
    if not chunks:
        print("No .txt documents found in data/Papers, data/CV or data/Personal_Bio. Nothing to do.")
        return

    print(f"Embedding {len(chunks)} chunks with {config['embedding_model']}...")
    client = get_client()
    vectors = []
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        # The label line helps the search match questions like "what does the CV say...".
        texts = [f"{c['category']} — {c['source']}\n{c['text']}" for c in batch]
        vectors.append(embed_texts(client, texts, config, "RETRIEVAL_DOCUMENT"))
        print(f"  {min(start + BATCH_SIZE, len(chunks))}/{len(chunks)} done")
        time.sleep(PAUSE_BETWEEN_BATCHES)
    embeddings = np.vstack(vectors)

    # Replace the old index only after everything succeeded.
    if INDEX_DIR.exists():
        shutil.rmtree(INDEX_DIR)
    INDEX_DIR.mkdir()
    np.save(EMBEDDINGS_FILE, embeddings)
    CHUNKS_FILE.write_text(json.dumps(chunks, ensure_ascii=False, indent=1), encoding="utf-8")

    chunk_counts = Counter(c["category"] for c in chunks)
    INFO_FILE.write_text(json.dumps({
        "embedding_model": config["embedding_model"],
        "embedding_dimensions": int(embeddings.shape[1]),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files_per_category": dict(file_counts),
        "chunks_per_category": dict(chunk_counts),
    }, indent=1), encoding="utf-8")

    print("\nIndex built:")
    for category in CATEGORIES.values():
        print(f"  {category:<13} {file_counts[category]:>3} files  {chunk_counts[category]:>4} chunks")
    print(f"  {'TOTAL':<13} {sum(file_counts.values()):>3} files  {len(chunks):>4} chunks")


if __name__ == "__main__":
    main()
