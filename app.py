"""
app.py - the Gradio chat app (this is the Hugging Face Space entry point).

Run locally:
    python app.py
then open http://127.0.0.1:7860 in your browser.
"""

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
    if result["sources"]:
        answer += SOURCES_MARKER + "; ".join(result["sources"]) + "</sub>"
    return answer


ensure_index()

demo = gr.ChatInterface(
    fn=respond,
    title=config["assistant_name"],
    chatbot=gr.Chatbot(placeholder=config["welcome_message"], height=520),
    examples=config["example_questions"],
    cache_examples=False,     # don't call the API at startup
    flagging_mode="never",
    save_history=False,
)

if __name__ == "__main__":
    demo.launch()
