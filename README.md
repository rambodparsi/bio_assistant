---
title: Bio Assistant
emoji: 🎓
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.31.0
python_version: "3.13"
app_file: app.py
pinned: false
short_description: Answers questions about my background and research.
---

# Bio Assistant

A small chatbot that answers visitors' questions about me and my work, using **only**
my own documents (CV, papers, bio). It uses Google Gemini for search ("embeddings")
and answers, and runs as a Gradio app on a Hugging Face Space.

> **Privacy:** my documents, the search index and all API keys are **never** in this
> GitHub repository. Documents and the index live only on my computer and in a
> **private** Hugging Face Dataset. Keys live in a local `.env` file (git-ignored) and
> in Hugging Face Space secrets.

## How it works (in one paragraph)

`build_index.py` splits the `.txt` files in `data/` into small overlapping chunks and
turns each one into an embedding. When a visitor asks a question, `rag.py` finds the
most similar chunks; if none is similar enough, it politely declines without calling
the model. Otherwise it sends those excerpts (labeled e.g. "CV" or "Paper — <title>")
and the question to Gemini with the rules in `prompts/system_prompt.txt`, and shows the
answer with a short "Sources" line.

## Files

| File | What it does |
|---|---|
| `app.py` | The chat web app (Space entry point) |
| `rag.py` | Search + answering logic |
| `build_index.py` | Builds the index from `data/` |
| `upload_data.py` | Uploads `data/` and `index/` to the private HF Dataset |
| `download_data.py` | Downloads the index from the private Dataset (used by the app) |
| `test_questions.py` | Runs a set of test questions |
| `config.yaml` | Your settings (name, welcome text, models, search settings) |
| `prompts/system_prompt.txt` | The assistant's persona and rules |
| `.env.example` | Names of the secret keys (no values) |

## 1. Setup (once)

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Fill in `.env` (secrets)

Copy `.env.example` to `.env` (if it does not exist yet) and fill in:

- `GEMINI_API_KEY` — from <https://aistudio.google.com/apikey>
- `HF_TOKEN` — from <https://huggingface.co/settings/tokens> (a **Write** token on your computer)

`.env` is git-ignored. Never share it or paste it anywhere.

## 3. Fill in `config.yaml`

Replace every `[PLACEHOLDER]`: your name, title/field, welcome message, example
questions and the dataset name (`hf_dataset_repo`, e.g. `Rambodparsi/bio-assistant-data`).
The model names and search settings already have sensible defaults.

## 4. Add your documents

Put plain-text `.txt` files in:

- `data/Papers/` — one file per paper, e.g. `2024_Paper_Title.txt` (the file name becomes the title)
- `data/CV/` — your CV
- `data/Personal_Bio/` — bio, interests, projects

Separate paragraphs with empty lines. Each folder has an `INSTRUCTIONS.txt` (ignored by the index).

## 5. Build the index

```bash
python build_index.py
```

Run it again whenever you add or change documents.

## 6. Test

```bash
python test_questions.py
```

```bash
python app.py
```

Then open <http://127.0.0.1:7860>.

## 7. Upload documents + index to the private Hugging Face Dataset

```bash
python upload_data.py
```

It creates the dataset as **private** if needed, refuses to upload to a public one,
and lists the uploaded files.

## Editing the persona

Edit `prompts/system_prompt.txt` and restart the app. No rebuild needed. To make
answers stricter or looser, change `similarity_threshold` in `config.yaml`
(higher = stricter).

## Later: deploying to a Hugging Face Space

1. Create a new Space at <https://huggingface.co/new-space> → SDK **Gradio**.
2. Push this repository's code to the Space (the YAML header at the top of this README configures it).
3. In the Space → **Settings → Variables and secrets**, add two **secrets**:
   - `GEMINI_API_KEY`
   - `HF_TOKEN` — a **read-only** token is enough here
4. On startup the app downloads the index from your private Dataset.
5. Embed it on your website with an `<iframe>` pointing to `https://<username>-<space-name>.hf.space`.
