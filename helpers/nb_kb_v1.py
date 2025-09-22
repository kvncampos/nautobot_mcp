import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

import requests
from chromadb import PersistentClient
from chromadb.config import Settings

from utils import SentenceTransformerEmbeddingFunction, config, get_chroma_db_path

# Configure logging
logger = logging.getLogger("nautobot_knowledge")
logger.setLevel(logging.INFO)


class NautobotKnowledge:
    def __init__(self) -> None:
        self.base_url: str = config.GLOBAL_TOOL_PROMPT
        self.token: str = config.NAUTOBOT_TOKEN
        self.model_name: str = config.EMBEDDING_MODEL
        self.db_path: str = str(Path(get_chroma_db_path()) / "nautobot_kb" / "db")

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
            name="nb_kb_collection", embedding_function=self.embedding_function
        )

        # List of GitHub repos
        self.repo_list = ["nautobot/nautobot"]

    def setup_repo_list(self) -> List[str]:
        repo_list: List[str] = []
        official_repo = "nautobot"

        # List of Top Official Nautobot Repos
        nautobot_public_repos = [
            "nautobot",
            "nautobot-plugin-chatops",
            "nautobot-plugin-golden-config",
            "nautobot-plugin-nornir",
            "nautobot-plugin-telemetry",
            "nautobot-app-device-onboarding",
        ]
        for repo in nautobot_public_repos:
            repo_list.append(f"{official_repo}/{repo}")
        return repo_list

    def _clean_metadata(self, metadata: dict) -> dict:
        cleaned = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
            elif isinstance(value, list):
                if all(isinstance(i, (str, int, float, bool)) for i in value):
                    cleaned[key] = ", ".join(map(str, value))
                elif all(isinstance(i, dict) for i in value):
                    cleaned[key] = json.dumps(value)
        return cleaned

    def fetch_repo_list(
        self, github_user_or_org: str, is_org: bool = True
    ) -> List[str]:
        """
        Fetch public repos for a given org/user using GitHub API. Returns list of "owner/repo" strings.
        """
        if is_org:
            url = f"https://api.github.com/orgs/{github_user_or_org}/repos"
        else:
            url = f"https://api.github.com/users/{github_user_or_org}/repos"
        headers = {"Accept": "application/vnd.github+json"}
        try:
            resp = requests.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            repo_list = [f"{repo['owner']['login']}/{repo['name']}" for repo in data]
            logger.info(f"Fetched {len(repo_list)} repos from {github_user_or_org}.")
            return repo_list
        except Exception as e:
            logger.error(f"Failed to fetch repos for {github_user_or_org}: {e}")
            return []

    # TODO We need to clone initially, then only check for updates with git pull/maybe compare with the hash commit?
    def initialize_collection(self) -> None:
        """
        Clone/fetch each repo, index all code/text files into ChromaDB.
        """
        temp_dir = tempfile.mkdtemp()
        documents = []
        metadatas = []
        ids = []
        try:
            for repo_full in self.repo_list:
                repo_url = f"https://github.com/{repo_full}.git"
                repo_name = repo_full.split("/")[-1]
                repo_dir = os.path.join(temp_dir, repo_name)
                # Clone repo (shallow)
                os.system(f"git clone --depth 1 {repo_url} {repo_dir}")
                for root, _, files in os.walk(repo_dir):
                    for fname in files:
                        if fname.endswith(
                            (".py", ".md", ".txt", ".js", ".ts", ".json")
                        ):
                            fpath = os.path.join(root, fname)
                            try:
                                with open(
                                    fpath, "r", encoding="utf-8", errors="ignore"
                                ) as f:
                                    content = f.read()
                                doc_id = (
                                    f"{repo_full}:{os.path.relpath(fpath, repo_dir)}"
                                )
                                metadata = {
                                    "repo": repo_full,
                                    "file_path": os.path.relpath(fpath, repo_dir),
                                    "file_name": fname,
                                }
                                cleaned_metadata = self._clean_metadata(metadata)
                                documents.append(content)
                                metadatas.append(cleaned_metadata)
                                ids.append(doc_id)
                            except Exception as fe:
                                logger.warning(f"Failed to read {fpath}: {fe}")
            if documents:
                self.collection.add(documents=documents, metadatas=metadatas, ids=ids)
                logger.info(
                    f"Indexed {len(documents)} files from {len(self.repo_list)} repos."
                )
            else:
                logger.info("No files indexed.")
        except Exception as e:
            logger.error(f"Failed to initialize GitHub KB: {e}")
        finally:
            shutil.rmtree(temp_dir)

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


if __name__ == "__main__":
    nk = NautobotKnowledge()
    nk.repo_list = ["nautobot/nautobot"]
    # nk.initialize_collection()
    results = nk.search("nautobot jobs")
    print(results)
