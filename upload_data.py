"""
upload_data.py - uploads data/ and index/ to your PRIVATE Hugging Face Dataset.

    python upload_data.py

- Uses HF_TOKEN from .env (must be a WRITE token).
- Creates the dataset as PRIVATE if it does not exist yet.
- Refuses to upload if the dataset exists but is PUBLIC.
- Checks again after uploading that the dataset is private, then lists its files.
"""

import os
import sys

from huggingface_hub import HfApi

from rag import INDEX_DIR, ROOT, load_config


def main():
    config = load_config()
    repo_id = config["hf_dataset_repo"]
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token or token.startswith("paste-your"):
        sys.exit("HF_TOKEN is not set. Add your Hugging Face WRITE token to .env.")
    if "[" in repo_id:
        sys.exit("Set hf_dataset_repo in config.yaml first (e.g. Rambodparsi/bio-assistant-data).")
    if not (INDEX_DIR / "embeddings.npy").exists():
        sys.exit("No index found. Run: python build_index.py")

    api = HfApi(token=token)

    # 1. Create the dataset as PRIVATE (does nothing if it already exists).
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)

    # 2. Safety check BEFORE uploading anything.
    if not api.repo_info(repo_id, repo_type="dataset").private:
        sys.exit(f"STOP: the dataset {repo_id} is PUBLIC. Make it private on Hugging Face first. Nothing was uploaded.")

    # 3. Upload. Files deleted locally are also removed from the dataset (delete_patterns).
    for folder in ("data", "index"):
        print(f"Uploading {folder}/ ...")
        api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=ROOT / folder,
            path_in_repo=folder,
            ignore_patterns=["**/INSTRUCTIONS.txt", "INSTRUCTIONS.txt", ".*", "**/.*"],
            delete_patterns=["*"],
            commit_message=f"Update {folder}/",
        )

    # 4. Verify it is still private and list what is there.
    is_private = api.repo_info(repo_id, repo_type="dataset").private
    files = sorted(api.list_repo_files(repo_id, repo_type="dataset"))
    print(f"\nDataset: https://huggingface.co/datasets/{repo_id}")
    print(f"Private: {'YES' if is_private else 'NO  <-- WARNING: make it private now!'}")
    print(f"Files ({len(files)}):")
    for name in files:
        print(f"  {name}")


if __name__ == "__main__":
    main()
