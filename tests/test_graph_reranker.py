"""
Test module for GraphReranker class.

This module tests the graph-based re-ranking functionality using Neo4j and Graphiti
for temporal knowledge graphs.
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, Mock, patch

import pytest

# Add the parent directory to Python path to enable imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from helpers.graph_reranker import GraphReranker


class TestGraphReranker:
    """Test suite for GraphReranker class."""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration values."""
        with patch("helpers.graph_reranker.config") as mock_config:
            mock_config.NEO4J_URI = "bolt://localhost:7687"
            mock_config.NEO4J_USER = "neo4j"
            mock_config.NEO4J_PASSWORD = "test_password"
            mock_config.NEO4J_DATABASE = "neo4j"
            mock_config.GRAPHITI_ENABLED = True
            mock_config.GRAPHITI_LLM_MODEL = "gpt-4"
            mock_config.GRAPHITI_EMBEDDING_MODEL = "text-embedding-3-small"
            yield mock_config

    @pytest.fixture
    def mock_neo4j_driver(self):
        """Mock Neo4j driver."""
        with patch("helpers.graph_reranker.GraphDatabase.driver") as mock_driver:
            mock_instance = Mock()
            mock_instance.verify_connectivity = Mock()
            mock_driver.return_value = mock_instance
            yield mock_driver

    @pytest.fixture
    def mock_graphiti(self):
        """Mock Graphiti client."""
        with patch("helpers.graph_reranker.Graphiti") as mock_graphiti_class:
            mock_instance = Mock()
            mock_graphiti_class.return_value = mock_instance
            yield mock_graphiti_class

    @pytest.fixture
    def sample_chroma_results(self) -> List[Dict[str, Any]]:
        """Sample ChromaDB results for testing."""
        return [
            {
                "document": "GET /api/dcim/devices/ - List all devices in DCIM",
                "metadata": {
                    "path": "/api/dcim/devices/",
                    "method": "GET",
                    "operation_id": "dcim_devices_list",
                    "description": "List all devices in DCIM",
                },
            },
            {
                "document": "POST /api/dcim/devices/ - Create a new device",
                "metadata": {
                    "path": "/api/dcim/devices/",
                    "method": "POST",
                    "operation_id": "dcim_devices_create",
                    "description": "Create a new device",
                },
            },
            {
                "document": "GET /api/dcim/locations/ - List all locations",
                "metadata": {
                    "path": "/api/dcim/locations/",
                    "method": "GET",
                    "operation_id": "dcim_locations_list",
                    "description": "List all locations",
                },
            },
            {
                "document": "GET /api/circuits/providers/ - List circuit providers",
                "metadata": {
                    "path": "/api/circuits/providers/",
                    "method": "GET",
                    "operation_id": "circuits_providers_list",
                    "description": "List circuit providers",
                },
            },
            {
                "document": "GET /api/dcim/racks/ - List all racks",
                "metadata": {
                    "path": "/api/dcim/racks/",
                    "method": "GET",
                    "operation_id": "dcim_racks_list",
                    "description": "List all racks",
                },
            },
        ]

    @pytest.fixture
    def reranker(self, mock_config, mock_neo4j_driver, mock_graphiti):
        """Create a GraphReranker instance with mocked dependencies."""
        reranker = GraphReranker()
        return reranker

    def test_initialization_success(self, mock_config, mock_neo4j_driver, mock_graphiti):
        """Test successful initialization of GraphReranker."""
        reranker = GraphReranker()

        assert reranker.neo4j_uri == "bolt://localhost:7687"
        assert reranker.neo4j_user == "neo4j"
        assert reranker.neo4j_password == "test_password"
        assert reranker.neo4j_database == "neo4j"
        assert reranker.enabled is True
        assert reranker.driver is not None
        assert reranker.graphiti is not None

        mock_neo4j_driver.assert_called_once()
        mock_graphiti.assert_called_once()

    def test_initialization_disabled(self, mock_config):
        """Test initialization when Graphiti is disabled."""
        mock_config.GRAPHITI_ENABLED = False

        with patch("helpers.graph_reranker.GraphDatabase.driver") as mock_driver:
            reranker = GraphReranker()

            assert reranker.enabled is False
            assert reranker.driver is None
            assert reranker.graphiti is None
            mock_driver.assert_not_called()

    def test_initialization_neo4j_failure(self, mock_config, mock_neo4j_driver):
        """Test initialization when Neo4j connection fails."""
        from neo4j.exceptions import Neo4jError

        mock_neo4j_driver.side_effect = Neo4jError("Connection failed")

        reranker = GraphReranker()

        assert reranker.enabled is False
        assert reranker.driver is None
        assert reranker.graphiti is None

    def test_initialization_graphiti_failure(
        self, mock_config, mock_neo4j_driver, mock_graphiti
    ):
        """Test initialization when Graphiti initialization fails."""
        mock_graphiti.side_effect = Exception("Graphiti initialization failed")

        reranker = GraphReranker()

        assert reranker.enabled is False
        assert reranker.driver is not None  # Neo4j connected
        assert reranker.graphiti is None

    def test_close(self, reranker):
        """Test closing Neo4j driver connection."""
        reranker.driver = Mock()
        reranker.close()

        reranker.driver.close.assert_called_once()

    def test_rerank_disabled(self, reranker, sample_chroma_results):
        """Test re-ranking when Graphiti is disabled."""
        reranker.enabled = False

        result = reranker.rerank("test query", sample_chroma_results, n_results=3)

        assert len(result) == 3
        assert result == sample_chroma_results[:3]

    def test_rerank_no_results(self, reranker):
        """Test re-ranking with no input results."""
        result = reranker.rerank("test query", [], n_results=3)

        assert result == []

    def test_rerank_graphiti_not_initialized(self, reranker, sample_chroma_results):
        """Test re-ranking when Graphiti is not initialized."""
        reranker.graphiti = None

        result = reranker.rerank("test query", sample_chroma_results, n_results=3)

        assert len(result) == 3
        assert result == sample_chroma_results[:3]

    def test_rerank_success(self, reranker, sample_chroma_results):
        """Test successful re-ranking."""
        # Mock _calculate_graph_score to return decreasing scores
        scores = [10.0, 50.0, 30.0, 20.0, 5.0]

        with patch.object(
            reranker, "_calculate_graph_score", side_effect=scores
        ) as mock_score:
            result = reranker.rerank("test query", sample_chroma_results, n_results=3)

            assert len(result) == 3
            # Should be sorted by score: indices 1, 2, 3 (scores 50, 30, 20)
            assert result[0] == sample_chroma_results[1]  # Score 50
            assert result[1] == sample_chroma_results[2]  # Score 30
            assert result[2] == sample_chroma_results[3]  # Score 20

            assert mock_score.call_count == 5

    def test_rerank_error_handling(self, reranker, sample_chroma_results):
        """Test re-ranking error handling."""
        with patch.object(
            reranker, "_calculate_graph_score", side_effect=Exception("Scoring error")
        ):
            result = reranker.rerank("test query", sample_chroma_results, n_results=3)

            # Should return original results on error
            assert len(result) == 3
            assert result == sample_chroma_results[:3]

    def test_calculate_graph_score_no_graphiti(self, reranker, sample_chroma_results):
        """Test score calculation when Graphiti is not initialized."""
        reranker.graphiti = None

        score = reranker._calculate_graph_score("test query", sample_chroma_results[0])

        assert score == 0.0

    def test_calculate_graph_score_success(self, reranker, sample_chroma_results):
        """Test successful score calculation."""
        with (
            patch.object(reranker, "_get_usage_frequency", return_value=25.0),
            patch.object(reranker, "_get_recency_score", return_value=20.0),
            patch.object(reranker, "_get_relationship_score", return_value=15.0),
            patch.object(reranker, "_get_success_rate", return_value=8.0),
        ):
            score = reranker._calculate_graph_score(
                "test query", sample_chroma_results[0]
            )

            # Total: 25 + 20 + 15 + 8 = 68
            assert score == 68.0

    def test_calculate_graph_score_capping(self, reranker, sample_chroma_results):
        """Test score calculation with capping."""
        with (
            patch.object(reranker, "_get_usage_frequency", return_value=50.0),  # Cap at 40
            patch.object(reranker, "_get_recency_score", return_value=40.0),  # Cap at 30
            patch.object(reranker, "_get_relationship_score", return_value=25.0),  # Cap at 20
            patch.object(reranker, "_get_success_rate", return_value=15.0),  # Cap at 10
        ):
            score = reranker._calculate_graph_score(
                "test query", sample_chroma_results[0]
            )

            # Total capped: 40 + 30 + 20 + 10 = 100
            assert score == 100.0

    def test_get_usage_frequency_no_driver(self, reranker):
        """Test usage frequency when driver is not available."""
        reranker.driver = None

        score = reranker._get_usage_frequency("GET /api/test/")

        assert score == 0.0

    def test_get_usage_frequency_no_record(self, reranker):
        """Test usage frequency when endpoint has no record."""
        mock_session = Mock()
        mock_result = Mock()
        mock_result.single.return_value = None
        mock_session.run.return_value = mock_result
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=False)

        reranker.driver.session.return_value = mock_session

        score = reranker._get_usage_frequency("GET /api/test/")

        assert score == 0.0

    def test_get_usage_frequency_with_count(self, reranker):
        """Test usage frequency with usage count."""
        mock_session = Mock()
        mock_result = Mock()
        mock_result.single.return_value = {"count": 10}
        mock_session.run.return_value = mock_result
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=False)

        reranker.driver.session.return_value = mock_session

        score = reranker._get_usage_frequency("GET /api/test/")

        assert score > 0.0
        assert score <= 40.0

    def test_get_recency_score_no_driver(self, reranker):
        """Test recency score when driver is not available."""
        reranker.driver = None

        score = reranker._get_recency_score("GET /api/test/")

        assert score == 0.0

    def test_get_recency_score_no_record(self, reranker):
        """Test recency score when endpoint has no record."""
        mock_session = Mock()
        mock_result = Mock()
        mock_result.single.return_value = None
        mock_session.run.return_value = mock_result
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=False)

        reranker.driver.session.return_value = mock_session

        score = reranker._get_recency_score("GET /api/test/")

        assert score == 0.0

    def test_get_recency_score_recent(self, reranker):
        """Test recency score for recently used endpoint."""
        mock_session = Mock()
        mock_result = Mock()
        # Recent timestamp (within 1 hour)
        recent_time = datetime.now().isoformat()
        mock_result.single.return_value = {"last_used": recent_time}
        mock_session.run.return_value = mock_result
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=False)

        reranker.driver.session.return_value = mock_session

        score = reranker._get_recency_score("GET /api/test/")

        assert score == 30.0  # Full score for recent usage

    def test_get_relationship_score_no_driver(self, reranker):
        """Test relationship score when driver is not available."""
        reranker.driver = None

        score = reranker._get_relationship_score("GET /api/test/", "query")

        assert score == 0.0

    def test_get_relationship_score_no_relationships(self, reranker):
        """Test relationship score with no relationships."""
        mock_session = Mock()
        mock_result = Mock()
        mock_result.single.return_value = {"related_count": 0, "total_weight": 0}
        mock_session.run.return_value = mock_result
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=False)

        reranker.driver.session.return_value = mock_session

        score = reranker._get_relationship_score("GET /api/test/", "query")

        assert score == 0.0

    def test_get_relationship_score_with_relationships(self, reranker):
        """Test relationship score with relationships."""
        mock_session = Mock()
        mock_result = Mock()
        mock_result.single.return_value = {"related_count": 5, "total_weight": 50}
        mock_session.run.return_value = mock_result
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=False)

        reranker.driver.session.return_value = mock_session

        score = reranker._get_relationship_score("GET /api/test/", "query")

        # Base: min(5*2, 10) = 10, Weight: min(50/10, 10) = 5
        assert score == 15.0

    def test_get_success_rate_no_driver(self, reranker):
        """Test success rate when driver is not available."""
        reranker.driver = None

        score = reranker._get_success_rate("GET /api/test/")

        assert score == 0.0

    def test_get_success_rate_no_record(self, reranker):
        """Test success rate with no usage record."""
        mock_session = Mock()
        mock_result = Mock()
        mock_result.single.return_value = {"success": 0, "failure": 0}
        mock_session.run.return_value = mock_result
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=False)

        reranker.driver.session.return_value = mock_session

        score = reranker._get_success_rate("GET /api/test/")

        assert score == 5.0  # Neutral score

    def test_get_success_rate_perfect(self, reranker):
        """Test success rate with perfect success."""
        mock_session = Mock()
        mock_result = Mock()
        mock_result.single.return_value = {"success": 10, "failure": 0}
        mock_session.run.return_value = mock_result
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=False)

        reranker.driver.session.return_value = mock_session

        score = reranker._get_success_rate("GET /api/test/")

        assert score == 10.0  # Perfect score

    def test_get_success_rate_partial(self, reranker):
        """Test success rate with partial success."""
        mock_session = Mock()
        mock_result = Mock()
        mock_result.single.return_value = {"success": 7, "failure": 3}
        mock_session.run.return_value = mock_result
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=False)

        reranker.driver.session.return_value = mock_session

        score = reranker._get_success_rate("GET /api/test/")

        assert score == 7.0  # 70% success rate

    @pytest.mark.asyncio
    async def test_record_usage_disabled(self, reranker, sample_chroma_results):
        """Test recording usage when disabled."""
        reranker.enabled = False

        # Should not raise exception
        await reranker.record_usage(
            sample_chroma_results[0], "test query", success=True
        )

    @pytest.mark.asyncio
    async def test_record_usage_no_graphiti(self, reranker, sample_chroma_results):
        """Test recording usage when Graphiti is not initialized."""
        reranker.graphiti = None

        # Should not raise exception
        await reranker.record_usage(
            sample_chroma_results[0], "test query", success=True
        )

    @pytest.mark.asyncio
    async def test_record_usage_success(self, reranker, sample_chroma_results):
        """Test successful usage recording."""
        with (
            patch.object(reranker, "_update_endpoint_node", new_callable=AsyncMock) as mock_update,
            patch.object(reranker, "_create_episode", new_callable=AsyncMock) as mock_episode,
            patch.object(reranker, "_update_relationships", new_callable=AsyncMock) as mock_relations,
        ):
            await reranker.record_usage(
                sample_chroma_results[0],
                "test query",
                success=True,
                context={"related_endpoints": ["GET /api/test/"]},
            )

            mock_update.assert_called_once()
            mock_episode.assert_called_once()
            mock_relations.assert_called_once()

    @pytest.mark.asyncio
    async def test_record_usage_without_context(self, reranker, sample_chroma_results):
        """Test usage recording without context."""
        with (
            patch.object(reranker, "_update_endpoint_node", new_callable=AsyncMock) as mock_update,
            patch.object(reranker, "_create_episode", new_callable=AsyncMock) as mock_episode,
            patch.object(reranker, "_update_relationships", new_callable=AsyncMock) as mock_relations,
        ):
            await reranker.record_usage(
                sample_chroma_results[0], "test query", success=True
            )

            mock_update.assert_called_once()
            mock_episode.assert_called_once()
            mock_relations.assert_not_called()

    @pytest.mark.asyncio
    async def test_record_usage_error_handling(self, reranker, sample_chroma_results):
        """Test usage recording error handling."""
        with patch.object(
            reranker,
            "_update_endpoint_node",
            side_effect=Exception("Update failed"),
            new_callable=AsyncMock,
        ):
            # Should not raise exception
            await reranker.record_usage(
                sample_chroma_results[0], "test query", success=True
            )

    @pytest.mark.asyncio
    async def test_update_endpoint_node_no_driver(self, reranker):
        """Test updating endpoint node when driver is not available."""
        reranker.driver = None

        # Should not raise exception
        await reranker._update_endpoint_node("GET /api/test/", True)

    @pytest.mark.asyncio
    async def test_update_endpoint_node_success(self, reranker):
        """Test successful endpoint node update."""
        mock_session = Mock()
        mock_session.run = Mock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=False)

        reranker.driver.session.return_value = mock_session

        await reranker._update_endpoint_node("GET /api/test/", True)

        mock_session.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_episode_no_graphiti(self, reranker):
        """Test creating episode when Graphiti is not initialized."""
        reranker.graphiti = None

        # Should not raise exception
        await reranker._create_episode("GET /api/test/", "query", True, None)

    @pytest.mark.asyncio
    async def test_create_episode_success(self, reranker):
        """Test successful episode creation."""
        reranker.graphiti.add_episode = AsyncMock()

        await reranker._create_episode("GET /api/test/", "query", True, None)

        reranker.graphiti.add_episode.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_relationships_no_driver(self, reranker):
        """Test updating relationships when driver is not available."""
        reranker.driver = None

        # Should not raise exception
        await reranker._update_relationships("GET /api/test/", ["GET /api/other/"])

    @pytest.mark.asyncio
    async def test_update_relationships_success(self, reranker):
        """Test successful relationship updates."""
        mock_session = Mock()
        mock_session.run = Mock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=False)

        reranker.driver.session.return_value = mock_session

        await reranker._update_relationships(
            "GET /api/test/", ["GET /api/other1/", "GET /api/other2/"]
        )

        assert mock_session.run.call_count == 2

    def test_health_check_disabled(self, mock_config):
        """Test health check when Graphiti is disabled."""
        mock_config.GRAPHITI_ENABLED = False

        with patch("helpers.graph_reranker.GraphDatabase.driver"):
            reranker = GraphReranker()
            health = reranker.health_check()

            assert health["enabled"] is False
            assert health["neo4j_connected"] is False
            assert health["graphiti_initialized"] is False

    def test_health_check_success(self, reranker):
        """Test successful health check."""
        reranker.driver.verify_connectivity = Mock()

        health = reranker.health_check()

        assert health["enabled"] is True
        assert health["neo4j_connected"] is True
        assert health["graphiti_initialized"] is True
        assert len(health["errors"]) == 0

    def test_health_check_neo4j_failure(self, reranker):
        """Test health check with Neo4j connection failure."""
        reranker.driver.verify_connectivity.side_effect = Exception("Connection failed")

        health = reranker.health_check()

        assert health["enabled"] is True
        assert health["neo4j_connected"] is False
        assert len(health["errors"]) > 0

    def test_health_check_graphiti_not_initialized(self, reranker):
        """Test health check with Graphiti not initialized."""
        reranker.driver.verify_connectivity = Mock()
        reranker.graphiti = None

        health = reranker.health_check()

        assert health["enabled"] is True
        assert health["neo4j_connected"] is True
        assert health["graphiti_initialized"] is False
        assert any("Graphiti" in error for error in health["errors"])

    @pytest.mark.parametrize(
        "n_results,expected_count",
        [
            (1, 1),
            (3, 3),
            (5, 5),
            (10, 5),  # More requested than available
        ],
    )
    def test_rerank_different_n_results(
        self, reranker, sample_chroma_results, n_results, expected_count
    ):
        """Test re-ranking with different n_results values."""
        with patch.object(reranker, "_calculate_graph_score", return_value=10.0):
            result = reranker.rerank(
                "test query", sample_chroma_results, n_results=n_results
            )

            assert len(result) == min(expected_count, len(sample_chroma_results))
