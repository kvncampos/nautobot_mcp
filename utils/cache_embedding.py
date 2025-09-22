from pathlib import Path

from huggingface_hub import snapshot_download
from path import get_chroma_db_path
from sentence_transformers import SentenceTransformer

# This is used to switch out or download models locally and cache them for local use

hf_models_to_cache = ["facebook/bart-base"]


def get_sentence_transformer_model() -> SentenceTransformer:
    """Get a SentenceTransformer model with caching enabled."""
    model = SentenceTransformer(
        model_name_or_path="all-MiniLM-L6-v2",
        cache_folder=str(Path(get_chroma_db_path()) / "models"),
    )
    return model


if __name__ == "__main__":
    cache_folder = Path("backend", "models")
    cache_folder.mkdir(parents=True, exist_ok=True)

    for model in hf_models_to_cache:
        # Download and cache the CodeBERT model
        local_path = snapshot_download(
            repo_id=model,
            cache_dir=str(cache_folder),
            resume_download=True,
        )

        print(f"Model cached at: {local_path}")
