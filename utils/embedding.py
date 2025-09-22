from pathlib import Path

from chromadb import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer

from utils.path import get_chroma_db_path


class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: str = str(Path(get_chroma_db_path()) / "models"),
        local_files_only: bool = True,
    ):
        self.model = SentenceTransformer(
            model_name, cache_folder=cache_dir, local_files_only=local_files_only
        )

    def __call__(self, input: Documents) -> Embeddings:
        return self.model.encode(input).tolist()
