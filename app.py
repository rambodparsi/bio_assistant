"""
app.py - the Gradio chat app (this is the Hugging Face Space entry point).

Run locally:
    python app.py
then open http://127.0.0.1:7860 in your browser.
"""

import base64
import html
import mimetypes

import gradio as gr
from google.genai import errors

import download_data
import rag

config = rag.load_config()

# Marker that separates an answer from its "Sources" line. It is used to strip the
# sources line again before the answer is sent back to the model as history.
SOURCES_MARKER = "\n\n---\n<sub>Sources: "


def ensure_index():
    """On a fresh Hugging Face Space there is no index yet: download it from the private Dataset."""
    if rag.EMBEDDINGS_FILE.exists() and rag.CHUNKS_FILE.exists():
        return
    print("Index not found locally - downloading it from the private Hugging Face Dataset...")
    try:
        download_data.download(include_documents=False)
        print("Index downloaded.")
    except Exception as error:
        # Only the error type is printed - never tokens or details that could contain them.
        print(f"Could not download the index ({type(error).__name__}). Check HF_TOKEN and hf_dataset_repo.")


def message_text(content):
    """Gradio may send a message as plain text or as a list of parts; return plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return content.get("text") or ""
    if isinstance(content, list):
        return " ".join(message_text(part) for part in content).strip()
    return ""


def clean_history(history):
    """Turn Gradio's history into simple {"role", "content"} dicts, without the Sources lines."""
    cleaned = []
    for message in history or []:
        text = message_text(message.get("content")).split(SOURCES_MARKER)[0].strip()
        if text:
            cleaned.append({"role": message.get("role"), "content": text})
    return cleaned


def friendly_error(error):
    """Turn an exception into a short, polite message with no technical details or secrets."""
    if isinstance(error, rag.MissingKeyError):
        return "The assistant is not configured yet (missing API key). Please try again later."
    if isinstance(error, rag.IndexMissingError):
        return "The assistant's knowledge base is not available right now. Please try again later."
    if isinstance(error, errors.APIError):
        if error.code == 429:
            return "The assistant is receiving many requests right now. Please wait a minute and try again."
        if error.code in (400, 401, 403) and "key" in str(error).lower():
            return "The assistant is not configured correctly (API key problem). Please try again later."
        return "Sorry, the AI service had a temporary problem. Please try again in a moment."
    return "Sorry, something went wrong. Please try again in a moment."


def respond(message, history):
    """Called by Gradio for every visitor message."""
    question = message_text(message).strip()
    if not question:
        return "Please type a question."
    try:
        result = rag.answer_question(question, clean_history(history), config)
    except Exception as error:
        print(f"Error while answering: {type(error).__name__}")
        return friendly_error(error)

    answer = result["answer"]
    if config.get("show_sources", False) and result["sources"]:
        answer += SOURCES_MARKER + "; ".join(result["sources"]) + "</sub>"
    return answer


ensure_index()

# Fill the whole window (so the chat fits the box it is embedded in on the website),
# with small even padding, no footer, and card colors that suit the website (light / dark).
CSS = """
.gradio-container { max-width: 100% !important; padding: 16px !important; margin: 0 !important; }
.gradio-container .main.fillable { padding: 0 !important; }
/* Never taller than the window: the chat area shrinks to whatever space is left */
.gradio-container { height: 100vh !important; min-height: 0 !important; overflow: hidden; }
.gradio-container .main, .gradio-container .wrap, .gradio-container main.contain,
.gradio-container .column, .gradio-container .block.flex, .gradio-container .wrapper { min-height: 0 !important; }
.gradio-container, .gradio-container * {
  font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Helvetica, Arial, sans-serif !important; }
/* Header: icon left, title + description beside it, left-aligned */
.gradio-container .html-container:has(.chat-header) { padding: 0 0 4px !important; }
.chat-header { display: flex; align-items: flex-start; gap: 14px; margin: 0 0 8px; }
.chat-header img { width: 56px; height: 56px; border-radius: 50%; flex-shrink: 0; margin-top: 2px; }
.chat-header .title { font-size: 18px; font-weight: 600; letter-spacing: -.01em; line-height: 1.3; }
.chat-header .desc { margin-top: 3px; font-size: 14px; line-height: 1.45; color: #6f6b66; }
.dark .chat-header .desc { color: #a29e98; }
body, gradio-app, .gradio-container { background: #f1efec !important; }
.dark body, .dark gradio-app, .dark .gradio-container { background: #252423 !important; }
.gradio-container .placeholder { padding: 0 16px !important; }
footer { display: none !important; }
"""

def header_html():
    """Icon on the left, title and description on the right (from config.yaml)."""
    icon = ""
    icon_path = rag.ROOT / (config.get("assistant_icon") or "")
    if config.get("assistant_icon") and icon_path.is_file():
        mime = mimetypes.guess_type(icon_path.name)[0] or "image/png"
        data = base64.b64encode(icon_path.read_bytes()).decode()
        icon = f'<img src="data:{mime};base64,{data}" alt="">'
    desc = config.get("assistant_description") or ""
    desc = f'<div class="desc">{html.escape(desc)}</div>' if desc else ""
    return (f'<div class="chat-header">{icon}<div><div class="title">'
            f'{html.escape(config["assistant_name"])}</div>{desc}</div></div>')


with gr.Blocks(title=config["assistant_name"], fill_height=True, fill_width=True) as demo:
    gr.HTML(header_html())
    gr.ChatInterface(
        fn=respond,
        chatbot=gr.Chatbot(placeholder=config.get("welcome_message") or None, show_label=False, scale=1, min_height=200),
        examples=config.get("example_questions") or None,
        cache_examples=False,     # don't call the API at startup
        flagging_mode="never",
        save_history=False,
        fill_height=True,
    )

if __name__ == "__main__":
    demo.launch(css=CSS, footer_links=[])
