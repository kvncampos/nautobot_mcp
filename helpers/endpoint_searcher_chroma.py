import logging
from pathlib import Path
from typing import List, Optional

import requests
import urllib3
from chromadb import PersistentClient
from chromadb.config import Settings

from utils import SentenceTransformerEmbeddingFunction, config, get_chroma_db_path

# Disable SSL warnings if SSL verification is disabled
if not config.SSL_VERIFY:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure logging
logger = logging.getLogger("endpoint_searcher_chroma")
logger.setLevel(logging.INFO)


class EndpointSearcherChroma:
    def __init__(self) -> None:
        self.base_url: str = config.GLOBAL_TOOL_PROMPT
        self.token: str = config.NAUTOBOT_TOKEN
        self.model_name: str = config.EMBEDDING_MODEL
        self.db_path: str = str(Path(get_chroma_db_path()) / "nautobot_api" / "db")

        # Use the existing model cache location - go up one level from helpers to root, then to backend/models
        self.model_cache_dir: str = str(
            Path(__file__).parent.parent / "backend" / "models"
        )

        # Setup Chroma client
        self.client = PersistentClient(
            path=self.db_path, settings=Settings(anonymized_telemetry=False)
        )

        # Get Model from local cache only.
        self.embedding_function = SentenceTransformerEmbeddingFunction(
            model_name=self.model_name,
            cache_dir=self.model_cache_dir,
            local_files_only=True,
        )

        self.collection = self.client.get_or_create_collection(
            name="endpoint_collection", embedding_function=self.embedding_function
        )

    def _clean_metadata(self, metadata: dict) -> dict:
        """
        Ensure metadata contains only str, int, float, or bool.
        If a list contains dicts (e.g., OpenAPI parameters), serialize them.
        """
        cleaned = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
            elif isinstance(value, list):
                if all(isinstance(i, (str, int, float, bool)) for i in value):
                    cleaned[key] = ", ".join(map(str, value))
                elif all(isinstance(i, dict) for i in value):
                    # Serialize list of dicts as JSON string
                    import json

                    cleaned[key] = json.dumps(value)
        return cleaned

    def initialize_collection(self) -> None:
        try:
            headers = {"Authorization": f"Token {self.token}"} if self.token else {}
            response = requests.get(
                self.base_url, headers=headers, verify=config.SSL_VERIFY
            )
            response.raise_for_status()
            data = response.json()
            logger.info("Fetched Swagger/OpenAPI schema.")

            documents = []
            metadatas = []
            ids = []
            for path, methods in data.get("paths", {}).items():
                for method, operation in methods.items():
                    combined_text = (
                        f"{method.upper()} {path} - {operation.get('description', '')}"
                    )
                    metadata = {
                        "path": path,
                        "method": method.upper(),
                        "operation_id": operation.get("operationId", ""),
                        "description": operation.get("description", ""),
                        "parameters": operation.get("parameters", []),
                    }
                    cleaned_metadata = self._clean_metadata(metadata)
                    documents.append(combined_text)
                    metadatas.append(cleaned_metadata)
                    ids.append(f"{method.upper()} {path}")

            self.collection.add(documents=documents, metadatas=metadatas, ids=ids)
            logger.info(f"Successfully indexed {len(documents)} endpoints.")
        except Exception as e:
            logger.error(f"Failed to initialize endpoints: {e}")

    def search(self, query: str, n_results: int = 5) -> Optional[List[dict]]:
        try:
            result = self.collection.query(query_texts=[query], n_results=n_results)
            documents = result.get("documents") if result else None
            metadatas = result.get("metadatas") if result else None
            if documents and isinstance(documents, list) and len(documents) > 0:
                docs_list = documents[0]
                metas_list = (
                    metadatas[0]
                    if metadatas and isinstance(metadatas, list) and len(metadatas) > 0
                    else []
                )
                combined = [
                    {"document": doc, "metadata": meta}
                    for doc, meta in zip(docs_list, metas_list)
                ]
                logger.debug(
                    f"Search returned {len(combined)} results for query '{query}'."
                )
                return combined
            else:
                logger.info(f"No results found for query '{query}'.")
                return None
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return None
