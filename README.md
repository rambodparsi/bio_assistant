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
and answers, and runs as a Gradio app on a free Render web service.

> **Privacy:** my documents, the search index and all API keys are **never** in this
> GitHub repository. Documents and the index live only on my computer and in a
> **private** Hugging Face Dataset. Keys live in a local `.env` file (git-ignored) and
> in Render's environment settings.

## How it works (in one paragraph)

`build_index.py` splits the `.txt` files in `data/` into small overlapping chunks and
turns each one into an embedding. When a visitor asks a question, `rag.py` finds the
most similar chunks; if none is similar enough, it politely declines without calling
the model. Otherwise it sends those excerpts (labeled e.g. "CV" or "Paper — <title>")
and the question to Gemini with the rules in `prompts/system_prompt.txt`, and shows the
short answer (without citations; set `show_sources: true` in `config.yaml` to add a
"Sources" line).

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
| `render.yaml` | Render deployment settings (no secrets) |

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

## Deploying to Render (free)

Hugging Face now requires a paid PRO plan for Gradio Spaces, so the app runs on
[Render](https://render.com)'s free web service instead (no credit card, no time limit).
`render.yaml` already holds all settings; only the two secrets are entered by hand.

1. Make sure this code is on the repository's default branch (`main`).
2. Create a **read-only** Hugging Face token at <https://huggingface.co/settings/tokens>
   (the app only needs to download the index).
3. Sign up at <https://dashboard.render.com> with **GitHub**.
4. **New + → Blueprint** → connect and pick the `bio_assistant` repository → **Apply**.
5. When asked, paste the two secrets:
   - `GEMINI_API_KEY`
   - `HF_TOKEN` — the read-only token from step 2
6. Wait for the first deploy (a few minutes). The app is then at
   `https://bio-assistant-XXXX.onrender.com` (shown at the top of the service page).
   On startup it downloads the index from your private Dataset.
7. Embed it on your website:

   ```html
   <iframe src="https://bio-assistant-XXXX.onrender.com" width="100%" height="700" style="border:0"></iframe>
   ```

Good to know:

- The free service **sleeps after 15 minutes without visitors**; the next visitor waits
  about 30–60 seconds while it wakes up.
- Every push to `main` redeploys automatically.
- After `python build_index.py` + `python upload_data.py`, use **Manual Deploy → Restart
  service** on Render so the app downloads the new index.
- To change a key later: service → **Environment**.
