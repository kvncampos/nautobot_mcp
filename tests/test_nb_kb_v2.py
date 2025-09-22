"""
Test module for EnhancedNautobotKnowledge class.

This module tests the functionality of the enhanced knowledge base
with different query scenarios and responses.
"""

import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add the parent directory to Python path to enable imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from helpers.nb_kb_v2 import EnhancedNautobotKnowledge
from utils.repo_config import RepositoryConfig


class TestEnhancedNautobotKnowledge:
    """Test suite for EnhancedNautobotKnowledge class."""

    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for testing."""
        config_dir = tempfile.mkdtemp()
        cache_dir = tempfile.mkdtemp()

        yield config_dir, cache_dir

        # Cleanup
        shutil.rmtree(config_dir, ignore_errors=True)
        shutil.rmtree(cache_dir, ignore_errors=True)

    @pytest.fixture
    def mock_kb(self, temp_dirs):
        """Create a mocked EnhancedNautobotKnowledge instance."""
        config_dir, cache_dir = temp_dirs

        with (
            patch("helpers.nb_kb_v2.RepositoryConfigManager"),
            patch("helpers.nb_kb_v2.GitRepoManager"),
            patch("helpers.nb_kb_v2.HybridContentProcessor"),
            patch("helpers.nb_kb_v2.SentenceTransformerEmbeddingFunction"),
        ):
            kb = EnhancedNautobotKnowledge(config_dir=config_dir, cache_dir=cache_dir)
            return kb

    @pytest.fixture
    def sample_repo_config(self):
        """Create a sample repository configuration."""
        return RepositoryConfig(
            name="test/repo",
            description="Test repository",
            priority=1,
            enabled=True,
            branch="main",
            file_patterns=[".py", ".md", ".txt"],
        )

    def test_initialization(self, temp_dirs):
        """Test proper initialization of EnhancedNautobotKnowledge."""
        config_dir, cache_dir = temp_dirs

        with (
            patch("helpers.nb_kb_v2.RepositoryConfigManager"),
            patch("helpers.nb_kb_v2.GitRepoManager"),
            patch("helpers.nb_kb_v2.HybridContentProcessor"),
            patch("helpers.nb_kb_v2.SentenceTransformerEmbeddingFunction"),
        ):
            kb = EnhancedNautobotKnowledge(config_dir=config_dir, cache_dir=cache_dir)

            assert kb.db_base_path is not None
            assert kb.model_name is not None
            assert kb.repo_clients == {}
            assert kb.repo_collections == {}

    def test_get_repo_db_path(self, mock_kb):
        """Test repository database path generation."""
        repo_name = "nautobot/nautobot"
        path = mock_kb._get_repo_db_path(repo_name)

        assert "nautobot_nautobot" in path
        assert "/" not in Path(path).name  # Should be sanitized

    def test_should_process_file(self, mock_kb):
        """Test file filtering logic."""
        file_patterns = [".py", ".md", ".txt"]

        # Test matching files
        assert mock_kb._should_process_file(Path("test.py"), file_patterns)
        assert mock_kb._should_process_file(Path("README.md"), file_patterns)
        assert mock_kb._should_process_file(Path("notes.txt"), file_patterns)

        # Test non-matching files
        assert not mock_kb._should_process_file(Path("image.jpg"), file_patterns)
        assert not mock_kb._should_process_file(Path("binary.exe"), file_patterns)

    def test_clean_metadata(self, mock_kb):
        """Test metadata cleaning for ChromaDB compatibility."""
        dirty_metadata = {
            "string_field": "test",
            "int_field": 123,
            "float_field": 45.6,
            "bool_field": True,
            "list_field": ["a", "b", "c"],
            "dict_list": [{"key": "value"}],
            "invalid_object": {"nested": {"dict": "value"}},
        }

        cleaned = mock_kb._clean_metadata(dirty_metadata)

        assert cleaned["string_field"] == "test"
        assert cleaned["int_field"] == 123
        assert cleaned["float_field"] == 45.6
        assert cleaned["bool_field"] is True
        assert cleaned["list_field"] == "a, b, c"
        assert "dict_list" in cleaned
        # Complex objects should be filtered out or converted

    @patch("helpers.nb_kb_v2.os.walk")
    @patch("builtins.open")
    def test_index_repository_files(
        self, mock_open, mock_walk, mock_kb, sample_repo_config
    ):
        """Test repository file indexing."""
        # Mock file system walk
        mock_walk.return_value = [("/repo", [], ["test.py", "README.md", "image.jpg"])]

        # Mock file content
        mock_open.return_value.__enter__.return_value.read.return_value = "test content"

        # Mock collection
        mock_collection = MagicMock()
        mock_kb._get_repo_collection = MagicMock(return_value=mock_collection)

        # Test indexing
        mock_kb._index_repository_files(Path("/repo"), sample_repo_config)

        # Should process 2 files (.py and .md), skip .jpg
        assert mock_collection.add.called
        call_args = mock_collection.add.call_args
        assert len(call_args[1]["documents"]) == 2  # Only .py and .md files

    def test_search_single_repository(self, mock_kb):
        """Test searching within a single repository."""
        # Mock collection with search results
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["Sample document content"]],
            "metadatas": [[{"repo": "test/repo", "file_path": "test.py"}]],
            "distances": [[0.1]],
        }

        mock_kb._get_repo_collection = MagicMock(return_value=mock_collection)
        mock_kb.repo_config_manager.get_enabled_repo_names = MagicMock(
            return_value=["test/repo"]
        )

        results = mock_kb.search("test query", n_results=5, repositories=["test/repo"])

        assert results is not None
        assert len(results) == 1
        assert results[0]["document"] == "Sample document content"
        assert results[0]["repository"] == "test/repo"
        assert results[0]["distance"] == 0.1

    def test_search_multiple_repositories(self, mock_kb):
        """Test searching across multiple repositories."""
        # Mock collections for different repos
        mock_collection1 = MagicMock()
        mock_collection1.query.return_value = {
            "documents": [["Content from repo1"]],
            "metadatas": [[{"repo": "test/repo1", "file_path": "test1.py"}]],
            "distances": [[0.2]],
        }

        mock_collection2 = MagicMock()
        mock_collection2.query.return_value = {
            "documents": [["Content from repo2"]],
            "metadatas": [[{"repo": "test/repo2", "file_path": "test2.py"}]],
            "distances": [[0.1]],
        }

        def mock_get_collection(repo_name):
            if repo_name == "test/repo1":
                return mock_collection1
            elif repo_name == "test/repo2":
                return mock_collection2

        mock_kb._get_repo_collection = mock_get_collection
        mock_kb.repo_config_manager.get_enabled_repo_names = MagicMock(
            return_value=["test/repo1", "test/repo2"]
        )

        results = mock_kb.search("test query", n_results=5)

        assert results is not None
        assert len(results) == 2
        # Results should be sorted by distance (lower first)
        assert results[0]["distance"] == 0.1  # repo2 result (better match)
        assert results[1]["distance"] == 0.2  # repo1 result

    def test_search_optimized_for_llm(self, mock_kb):
        """Test LLM-optimized search functionality."""
        # Mock the base search method
        mock_search_results = [
            {
                "document": "This is a long document that contains important information about Nautobot development.",
                "metadata": {
                    "repo": "test/repo",
                    "file_path": "test.py",
                    "file_extension": ".py",
                },
                "distance": 0.15,
                "repository": "test/repo",
            }
        ]

        mock_kb.search = MagicMock(return_value=mock_search_results)

        # Mock the content processor
        mock_processed = {
            "content": "Processed content about Nautobot",
            "processing_method": "keyword_extraction",
            "compressed_ratio": 0.7,
            "original_length": 89,
        }
        mock_kb.content_processor.intelligent_content_processing = MagicMock(
            return_value=mock_processed
        )

        results = mock_kb.search_optimized_for_llm("nautobot development", n_results=3)

        assert results is not None
        assert len(results) == 1

        result = results[0]
        assert "content" in result
        assert "relevance_score" in result
        assert "source" in result
        assert "processing" in result

        assert result["content"] == "Processed content about Nautobot"
        assert result["relevance_score"] == 0.85  # 1 - 0.15
        assert result["source"]["repo"] == "test/repo"
        assert result["processing"]["method"] == "keyword_extraction"

    def test_search_for_llm_content_truncation(self, mock_kb):
        """Test content truncation in LLM search."""
        long_content = "A" * 3000  # Very long content

        mock_search_results = [
            {
                "document": long_content,
                "metadata": {
                    "repo": "test/repo",
                    "file_path": "test.py",
                    "file_extension": ".py",
                },
                "distance": 0.2,
                "repository": "test/repo",
            }
        ]

        mock_kb.search = MagicMock(return_value=mock_search_results)

        results = mock_kb.search_for_llm("test query", n_results=1)

        assert results is not None
        assert len(results) == 1

        # Content should be truncated
        content = results[0]["content"]
        assert len(content) < len(long_content)
        assert "content truncated" in content

    def test_get_repository_stats(self, mock_kb):
        """Test repository statistics retrieval."""
        # Mock collections with different counts
        mock_collection1 = MagicMock()
        mock_collection1.count.return_value = 150

        mock_collection2 = MagicMock()
        mock_collection2.count.return_value = 75

        def mock_get_collection(repo_name):
            if repo_name == "repo1":
                return mock_collection1
            elif repo_name == "repo2":
                return mock_collection2

        mock_kb._get_repo_collection = mock_get_collection
        mock_kb.repo_config_manager.get_enabled_repo_names = MagicMock(
            return_value=["repo1", "repo2"]
        )

        stats = mock_kb.get_repository_stats()

        assert "repo1" in stats
        assert "repo2" in stats
        assert stats["repo1"]["document_count"] == 150
        assert stats["repo2"]["document_count"] == 75
        assert stats["repo1"]["enabled"] is True
        assert stats["repo2"]["enabled"] is True

    def test_update_repository_disabled(self, mock_kb, sample_repo_config):
        """Test that disabled repositories are skipped during update."""
        sample_repo_config.enabled = False

        result = mock_kb.update_repository(sample_repo_config, force=False)

        assert result is False

    def test_initialize_all_repositories(self, mock_kb):
        """Test initialization of all repositories."""
        # Mock repository configs
        repo_configs = [
            RepositoryConfig(name="repo1", enabled=True),
            RepositoryConfig(name="repo2", enabled=True),
        ]

        mock_kb.repo_config_manager.load_repositories = MagicMock(
            return_value=repo_configs
        )
        mock_kb.update_repository = MagicMock(
            side_effect=[True, False]
        )  # First succeeds, second fails

        results = mock_kb.initialize_all_repositories()

        assert "repo1" in results
        assert "repo2" in results
        assert results["repo1"] is True
        assert results["repo2"] is False

    def test_search_empty_results(self, mock_kb):
        """Test search behavior when no results are found."""
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        mock_kb._get_repo_collection = MagicMock(return_value=mock_collection)
        mock_kb.repo_config_manager.get_enabled_repo_names = MagicMock(
            return_value=["test/repo"]
        )

        results = mock_kb.search("nonexistent query")

        assert results is None

    def test_search_repository_specific(self, mock_kb):
        """Test searching within a specific repository only."""
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["Specific repo content"]],
            "metadatas": [[{"repo": "specific/repo", "file_path": "test.py"}]],
            "distances": [[0.1]],
        }

        mock_kb._get_repo_collection = MagicMock(return_value=mock_collection)

        results = mock_kb.search_repository("test query", "specific/repo", n_results=3)

        assert results is not None
        assert len(results) == 1
        assert results[0]["repository"] == "specific/repo"

    def test_error_handling_in_search(self, mock_kb):
        """Test error handling during search operations."""
        # Mock collection that raises an exception
        mock_collection = MagicMock()
        mock_collection.query.side_effect = Exception("Search failed")

        mock_kb._get_repo_collection = MagicMock(return_value=mock_collection)
        mock_kb.repo_config_manager.get_enabled_repo_names = MagicMock(
            return_value=["test/repo"]
        )

        results = mock_kb.search("test query")

        # Should return None on error, not raise exception
        assert results is None
