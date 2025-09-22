"""
Enhanced Nautobot Knowledge Base with per-repository vector databases.

This module provides an improved knowledge base system that:
- Stores each repository in its own ChromaDB collection
- Tracks Git commit hashes to only update when necessary
- Supports extensible repository configuration
- Provides efficient search across all or specific repositories
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from chromadb import PersistentClient
from chromadb.config import Settings

from helpers.content_processor import HybridContentProcessor
from utils import SentenceTransformerEmbeddingFunction, config, get_chroma_db_path
from utils.git_manager import GitRepoManager
from utils.repo_config import RepositoryConfig, RepositoryConfigManager

# Configure logging
logger = logging.getLogger("nautobot_knowledge_v2")
logger.setLevel(logging.INFO)


class EnhancedNautobotKnowledge:
    """Enhanced Knowledge Base with per-repository vector databases."""

    def __init__(
        self, config_dir: Optional[str] = None, cache_dir: Optional[str] = None
    ):
        """Initialize the enhanced knowledge base.

        Args:
            config_dir: Directory containing repository configuration files
            cache_dir: Directory for caching cloned repositories
        """
        # Set environment variables to force offline mode
        import os

        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

        self.model_name: str = config.EMBEDDING_MODEL
        self.db_base_path: str = str(Path(get_chroma_db_path()) / "nautobot_kb_v2")

        # Use the existing model cache location - go up one level from helpers to root, then to backend/models
        self.model_cache_dir: str = str(
            Path(__file__).parent.parent / "backend" / "models"
        )

        # Initialize managers
        self.repo_config_manager = RepositoryConfigManager(config_dir)
        self.git_manager = GitRepoManager(cache_dir)

        # Initialize intelligent content processor
        self.content_processor = HybridContentProcessor(self.model_name)

        # Setup embedding function
        self.embedding_function = SentenceTransformerEmbeddingFunction(
            model_name=self.model_name,
            cache_dir=self.model_cache_dir,
            local_files_only=True,
        )

        # Dictionary to hold per-repository ChromaDB clients and collections
        self.repo_clients = {}
        self.repo_collections = {}

    def _get_repo_db_path(self, repo_name: str) -> str:
        """Get the database path for a specific repository.

        Args:
            repo_name: Repository name (e.g., "nautobot/nautobot")

        Returns:
            Path to the repository's database directory
        """
        # Sanitize repo name for use as directory name
        safe_repo_name = repo_name.replace("/", "_").replace(":", "_")
        return str(Path(self.db_base_path) / safe_repo_name)

    def _get_repo_client(self, repo_name: str):
        """Get or create a ChromaDB client for a repository.

        Args:
            repo_name: Repository name

        Returns:
            ChromaDB PersistentClient for the repository
        """
        if repo_name not in self.repo_clients:
            db_path = self._get_repo_db_path(repo_name)
            self.repo_clients[repo_name] = PersistentClient(
                path=db_path, settings=Settings(anonymized_telemetry=False)
            )
        return self.repo_clients[repo_name]

    def _get_repo_collection(self, repo_name: str):
        """Get or create a ChromaDB collection for a repository.

        Args:
            repo_name: Repository name

        Returns:
            ChromaDB collection for the repository
        """
        if repo_name not in self.repo_collections:
            client = self._get_repo_client(repo_name)
            collection_name = "kb_collection"
            self.repo_collections[repo_name] = client.get_or_create_collection(
                name=collection_name, embedding_function=self.embedding_function
            )
        return self.repo_collections[repo_name]

    def _clean_metadata(self, metadata: dict) -> dict:
        """Clean metadata to ensure compatibility with ChromaDB."""
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

    def _should_process_file(self, file_path: Path, file_patterns: List[str]) -> bool:
        """Check if a file should be processed based on its extension.

        Args:
            file_path: Path to the file
            file_patterns: List of file extensions to include

        Returns:
            True if the file should be processed
        """
        return any(str(file_path).endswith(pattern) for pattern in file_patterns)

    def _index_repository_files(
        self, repo_path: Path, repo_config: RepositoryConfig
    ) -> int:
        """Index all relevant files from a repository.

        Args:
            repo_path: Path to the cloned repository
            repo_config: Repository configuration

        Returns:
            Number of files indexed
        """
        collection = self._get_repo_collection(repo_config.name)

        # Clear existing documents for this repository
        try:
            collection.delete()
            collection = self._get_repo_collection(
                repo_config.name
            )  # Recreate after delete
        except Exception as e:
            logger.warning(
                f"Could not clear existing documents for {repo_config.name}: {e}"
            )

        documents = []
        metadatas = []
        ids = []

        # Walk through repository files
        for root, _, files in os.walk(repo_path):
            for fname in files:
                fpath = Path(root) / fname

                # Check if file should be processed
                if not self._should_process_file(fpath, repo_config.file_patterns):
                    continue

                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()

                    # Skip empty files
                    if not content.strip():
                        continue

                    # Create unique ID for this document
                    relative_path = fpath.relative_to(repo_path)
                    doc_id = f"{repo_config.name}:{relative_path}"

                    # Create metadata
                    metadata = {
                        "repo": repo_config.name,
                        "file_path": str(relative_path),
                        "file_name": fname,
                        "file_extension": fpath.suffix,
                        "repo_description": repo_config.description,
                        "repo_priority": repo_config.priority,
                    }

                    cleaned_metadata = self._clean_metadata(metadata)

                    documents.append(content)
                    metadatas.append(cleaned_metadata)
                    ids.append(doc_id)

                except Exception as e:
                    logger.warning(f"Failed to read {fpath}: {e}")

        # Add documents to collection
        if documents:
            try:
                collection.add(documents=documents, metadatas=metadatas, ids=ids)
                logger.info(f"Indexed {len(documents)} files from {repo_config.name}")
                return len(documents)
            except Exception as e:
                logger.error(
                    f"Failed to add documents to collection for {repo_config.name}: {e}"
                )
                return 0
        else:
            logger.info(f"No files to index for {repo_config.name}")
            return 0

    def update_repository(
        self, repo_config: RepositoryConfig, force: bool = False
    ) -> bool:
        """Update a single repository if needed.

        Args:
            repo_config: Repository configuration
            force: Force update even if not needed

        Returns:
            True if repository was updated
        """
        if not repo_config.enabled:
            logger.info(f"Repository {repo_config.name} is disabled, skipping")
            return False

        logger.info(f"Checking repository {repo_config.name}...")

        # Check if update is needed
        if not force and not self.git_manager.needs_update(
            repo_config.name, repo_config.branch
        ):
            logger.info(f"Repository {repo_config.name} is up to date")
            return False

        # Clone or update the repository
        repo_path, was_updated = self.git_manager.clone_or_update_repo(
            repo_config.name, repo_config.branch
        )

        if not repo_path:
            logger.error(f"Failed to clone/update {repo_config.name}")
            return False

        if was_updated or force:
            # Re-index the repository
            indexed_count = self._index_repository_files(repo_path, repo_config)
            logger.info(
                f"Successfully updated and indexed {repo_config.name} ({indexed_count} files)"
            )
            return True

        return False

    def initialize_all_repositories(self, force: bool = False) -> Dict[str, bool]:
        """Initialize or update all configured repositories.

        Args:
            force: Force update all repositories

        Returns:
            Dictionary mapping repo names to update status
        """
        logger.info("Initializing all repositories...")

        repo_configs = self.repo_config_manager.load_repositories()
        results = {}

        for repo_config in repo_configs:
            try:
                was_updated = self.update_repository(repo_config, force)
                results[repo_config.name] = was_updated
            except Exception as e:
                logger.error(f"Failed to update repository {repo_config.name}: {e}")
                results[repo_config.name] = False

        logger.info(
            f"Repository initialization complete. Updated: {sum(results.values())}/{len(results)}"
        )
        return results

    def search(
        self, query: str, n_results: int = 5, repositories: Optional[List[str]] = None
    ) -> Optional[List[dict]]:
        """Search across repositories.

        Args:
            query: Search query
            n_results: Number of results to return
            repositories: List of repository names to search (None for all)

        Returns:
            List of search results with documents and metadata
        """
        try:
            # Determine which repositories to search
            if repositories is None:
                repositories = self.repo_config_manager.get_enabled_repo_names()

            all_results = []

            for repo_name in repositories:
                try:
                    collection = self._get_repo_collection(repo_name)

                    # Search this repository
                    result = collection.query(query_texts=[query], n_results=n_results)

                    if result and result.get("documents"):
                        documents = (
                            result["documents"][0] if result["documents"] else []
                        )
                        metadatas = (
                            result["metadatas"][0] if result.get("metadatas") else []
                        )
                        distances = (
                            result["distances"][0] if result.get("distances") else []
                        )

                        # Combine results with scores
                        for i, doc in enumerate(documents):
                            metadata = metadatas[i] if i < len(metadatas) else {}
                            distance = distances[i] if i < len(distances) else 1.0

                            all_results.append(
                                {
                                    "document": doc,
                                    "metadata": metadata,
                                    "distance": distance,
                                    "repository": repo_name,
                                }
                            )

                except Exception as e:
                    logger.warning(f"Failed to search repository {repo_name}: {e}")

            # Sort by distance (lower is better) and limit results
            all_results.sort(key=lambda x: x.get("distance", 1.0))
            final_results = all_results[:n_results]

            logger.debug(
                f"Search returned {len(final_results)} results for query '{query}'"
            )
            return final_results if final_results else None

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return None

    def _truncate_content_smartly(self, content: str, max_length: int) -> str:
        """Intelligently truncate content to fit within token limits.

        Args:
            content: The content to truncate
            max_length: Maximum character length

        Returns:
            Truncated content that preserves important information
        """
        if len(content) <= max_length:
            return content

        # Try to find natural break points
        lines = content.split("\n")
        result = []
        current_length = 0

        for line in lines:
            line_length = len(line) + 1  # +1 for newline
            if current_length + line_length > max_length:
                # If we have some content, break here
                if result:
                    break
                # If this is the first line and it's too long, truncate it
                remaining = max_length - current_length
                result.append(line[:remaining] + "...")
                break
            result.append(line)
            current_length += line_length

        return "\n".join(result)

    # MAIN ENTRY TO CHROMADB
    def search_optimized_for_llm(
        self,
        query: str,
        n_results: int = 5,
        repositories: Optional[List[str]] = None,
        max_content_length: int = 500,
    ) -> Optional[List[dict]]:
        """Search across repositories with LLM-optimized response format using intelligent processing.

        Args:
            query: Search query
            n_results: Number of results to return
            repositories: List of repository names to search (None for all)
            max_content_length: Maximum characters to include from each document

        Returns:
            List of LLM-optimized search results with intelligently processed content
        """
        raw_results = self.search(query, n_results, repositories)
        if not raw_results:
            return None

        optimized_results = []
        for result in raw_results:
            doc = result.get("document", "")
            metadata = result.get("metadata", {})

            # Use intelligent hybrid content processor
            processed = self.content_processor.intelligent_content_processing(
                document=doc,
                query=query,
                target_length=max_content_length,
                metadata=metadata,
            )

            # Calculate relevance score (1 - normalized distance)
            relevance = max(0.0, 1.0 - result.get("distance", 1.0))

            # Create minimal, LLM-focused result
            optimized_result = {
                "content": processed["content"],
                "relevance_score": round(relevance, 3),
                "source": {
                    "repo": metadata.get("repo", result.get("repository", "unknown")),
                    "file": metadata.get("file_path", "unknown"),
                    "type": metadata.get("file_extension", "").lstrip("."),
                },
                "processing": {
                    "method": processed["processing_method"],
                    "compression_ratio": processed["compressed_ratio"],
                    "original_size": processed["original_length"],
                },
            }

            # Only include if content is meaningful
            if len(processed["content"].strip()) > 20:
                optimized_results.append(optimized_result)

        return optimized_results if optimized_results else None

    def search_repository(
        self, query: str, repository: str, n_results: int = 5
    ) -> Optional[List[dict]]:
        """Search within a specific repository.

        Args:
            query: Search query
            repository: Repository name to search
            n_results: Number of results to return

        Returns:
            List of search results
        """
        return self.search(query, n_results, [repository])

    def search_for_llm(
        self, query: str, n_results: int = 5, repositories: Optional[List[str]] = None
    ) -> Optional[List[dict]]:
        """Search optimized for LLM consumption with minimal metadata and chunked content.

        Args:
            query: Search query
            n_results: Number of results to return
            repositories: List of repository names to search (None for all)

        Returns:
            List of optimized search results for LLM consumption
        """
        raw_results = self.search(query, n_results, repositories)
        if not raw_results:
            return None

        optimized_results = []
        for result in raw_results:
            # Calculate relevance score (1 - normalized distance)
            relevance = max(0.0, 1.0 - result.get("distance", 1.0))

            # Extract essential metadata only
            metadata = result.get("metadata", {})
            file_path = metadata.get("file_path", "unknown")

            # Chunk document content if too long (optimize for token usage)
            document = result.get("document", "")
            if len(document) > 2000:  # Configurable chunk size
                # Take first 1000 chars and last 500 chars with separator
                document = (
                    document[:1000] + "\n...[content truncated]...\n" + document[-500:]
                )

            optimized_result = {
                "content": document,
                "source": f"{result.get('repository', 'unknown')}/{file_path}",
                "relevance": round(relevance, 3),
                "type": metadata.get("file_extension", "unknown").lstrip("."),
            }

            optimized_results.append(optimized_result)

        return optimized_results

    def get_repository_stats(self) -> Dict[str, Dict]:
        """Get statistics for all repositories.

        Returns:
            Dictionary mapping repo names to their statistics
        """
        stats = {}
        repo_names = self.repo_config_manager.get_enabled_repo_names()

        for repo_name in repo_names:
            try:
                collection = self._get_repo_collection(repo_name)
                count = collection.count()
                stats[repo_name] = {"document_count": count, "enabled": True}
            except Exception as e:
                logger.warning(f"Failed to get stats for {repo_name}: {e}")
                stats[repo_name] = {
                    "document_count": 0,
                    "enabled": False,
                    "error": str(e),
                }

        return stats
