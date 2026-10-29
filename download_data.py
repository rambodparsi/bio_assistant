"""
download_data.py - fetches the index (and optionally the documents) from your
PRIVATE Hugging Face Dataset. app.py calls this automatically at startup when
index/ is missing (for example on the Hugging Face Space).

    python download_data.py            # index only (all the app needs)
    python download_data.py --all      # index + data/ documents

Uses HF_TOKEN (a READ token is enough).
"""

import os
import sys

from huggingface_hub import snapshot_download

from rag import ROOT, load_config


def download(include_documents=False):
    """Download index/ (and data/ if include_documents=True) into this project folder."""
    config = load_config()
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token or token.startswith("paste-your"):
        raise RuntimeError("HF_TOKEN is not set.")

    patterns = ["index/*"] + (["data/**"] if include_documents else [])
    snapshot_download(
        repo_id=config["hf_dataset_repo"],
        repo_type="dataset",
        local_dir=ROOT,
        allow_patterns=patterns,
        token=token,
    )


if __name__ == "__main__":
    download(include_documents="--all" in sys.argv)
    print("Download complete.")
