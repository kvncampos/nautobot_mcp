"""
Test module for EndpointSearcherChroma class.

This module tests the endpoint searching functionality using ChromaDB
for semantic search over OpenAPI/Swagger schemas.
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import requests

# Add the parent directory to Python path to enable imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from helpers.endpoint_searcher_chroma import EndpointSearcherChroma


class TestEndpointSearcherChroma:
    """Test suite for EndpointSearcherChroma class."""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database path for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def mock_config(self):
        """Mock configuration values."""
        with patch("helpers.endpoint_searcher_chroma.config") as mock_config:
            mock_config.GLOBAL_TOOL_PROMPT = "http://test.com/api/swagger.json"
            mock_config.NAUTOBOT_TOKEN = "test_token_123"
            mock_config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
            yield mock_config

    @pytest.fixture
    def mock_chroma_db_path(self, temp_db_path):
        """Mock the get_chroma_db_path function."""
        with patch("helpers.endpoint_searcher_chroma.get_chroma_db_path") as mock_path:
            mock_path.return_value = temp_db_path
            yield mock_path

    @pytest.fixture
    def sample_openapi_schema(self):
        """Sample OpenAPI schema for testing."""
        return {
            "paths": {
                "/api/dcim/devices/": {
                    "get": {
                        "operationId": "dcim_devices_list",
                        "description": "List all devices in DCIM",
                        "parameters": [
                            {
                                "name": "limit",
                                "type": "integer",
                                "description": "Number of results to return per page",
                            }
                        ],
                    },
                    "post": {
                        "operationId": "dcim_devices_create",
                        "description": "Create a new device",
                        "parameters": [],
                    },
                },
                "/api/dcim/locations/": {
                    "get": {
                        "operationId": "dcim_locations_list",
                        "description": "List all locations",
                        "parameters": [
                            {
                                "name": "name",
                                "type": "string",
                                "description": "Filter by location name",
                            }
                        ],
                    }
                },
                "/api/circuits/providers/": {
                    "get": {
                        "operationId": "circuits_providers_list",
                        "description": "List circuit providers",
                        "parameters": [],
                    }
                },
            }
        }

    @pytest.fixture
    def mock_searcher(self, mock_config, mock_chroma_db_path):
        """Create a mocked EndpointSearcherChroma instance."""
        with (
            patch(
                "helpers.endpoint_searcher_chroma.PersistentClient"
            ) as mock_client_class,
            patch(
                "helpers.endpoint_searcher_chroma.SentenceTransformerEmbeddingFunction"
            ) as mock_embedding_class,
        ):
            # Setup mock client and collection
            mock_client = Mock()
            mock_collection = Mock()
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_client_class.return_value = mock_client

            # Setup mock embedding function
            mock_embedding = Mock()
            mock_embedding_class.return_value = mock_embedding

            searcher = EndpointSearcherChroma()
            searcher.collection = mock_collection
            return searcher

    def test_initialization(self, mock_config, mock_chroma_db_path):
        """Test EndpointSearcherChroma initialization."""
        with (
            patch(
                "helpers.endpoint_searcher_chroma.PersistentClient"
            ) as mock_client_class,
            patch(
                "helpers.endpoint_searcher_chroma.SentenceTransformerEmbeddingFunction"
            ) as mock_embedding_class,
        ):
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            mock_embedding = Mock()
            mock_embedding_class.return_value = mock_embedding

            searcher = EndpointSearcherChroma()

            # Verify configuration
            assert searcher.base_url == "http://test.com/api/swagger.json"
            assert searcher.token == "test_token_123"
            assert searcher.model_name == "all-MiniLM-L6-v2"

            # Verify ChromaDB client setup
            mock_client_class.assert_called_once()
            mock_client.get_or_create_collection.assert_called_once()

            # Verify embedding function setup
            mock_embedding_class.assert_called_once_with(
                model_name="all-MiniLM-L6-v2",
                cache_dir=searcher.model_cache_dir,
                local_files_only=True,
            )

    def test_clean_metadata_with_simple_types(self, mock_searcher):
        """Test _clean_metadata with simple data types."""
        metadata = {
            "string_field": "test_value",
            "int_field": 42,
            "float_field": 3.14,
            "bool_field": True,
        }

        result = mock_searcher._clean_metadata(metadata)

        assert result == metadata

    def test_clean_metadata_with_string_list(self, mock_searcher):
        """Test _clean_metadata with list of strings."""
        metadata = {
            "tags": ["device", "network", "hardware"],
            "categories": ["dcim", "ipam"],
        }

        result = mock_searcher._clean_metadata(metadata)

        expected = {"tags": "device, network, hardware", "categories": "dcim, ipam"}
        assert result == expected

    def test_clean_metadata_with_mixed_list(self, mock_searcher):
        """Test _clean_metadata with list of mixed simple types."""
        metadata = {
            "mixed_list": ["string", 42, 3.14, True],
        }

        result = mock_searcher._clean_metadata(metadata)

        expected = {
            "mixed_list": "string, 42, 3.14, True",
        }
        assert result == expected

    def test_clean_metadata_with_dict_list(self, mock_searcher):
        """Test _clean_metadata with list of dictionaries."""
        metadata = {
            "parameters": [
                {"name": "limit", "type": "integer"},
                {"name": "offset", "type": "integer"},
            ]
        }

        result = mock_searcher._clean_metadata(metadata)

        expected_json = json.dumps(
            [
                {"name": "limit", "type": "integer"},
                {"name": "offset", "type": "integer"},
            ]
        )
        assert result["parameters"] == expected_json

    def test_clean_metadata_filters_unsupported_types(self, mock_searcher):
        """Test _clean_metadata filters out unsupported types."""
        metadata = {
            "valid_string": "test",
            "valid_int": 42,
            "invalid_dict": {"nested": "value"},
            "invalid_set": {1, 2, 3},
            "invalid_none": None,
        }

        result = mock_searcher._clean_metadata(metadata)

        expected = {
            "valid_string": "test",
            "valid_int": 42,
        }
        assert result == expected

    @patch("helpers.endpoint_searcher_chroma.config.SSL_VERIFY", True)
    @patch("helpers.endpoint_searcher_chroma.requests.get")
    def test_initialize_collection_success(
        self, mock_get, mock_searcher, sample_openapi_schema
    ):
        """Test successful initialization of collection with OpenAPI schema."""
        # Setup mock response
        mock_response = Mock()
        mock_response.json.return_value = sample_openapi_schema
        mock_get.return_value = mock_response

        mock_searcher.initialize_collection()

        # Verify API call
        mock_get.assert_called_once_with(
            "http://test.com/api/swagger.json",
            headers={"Authorization": "Token test_token_123"},
            verify=True,
        )
        mock_response.raise_for_status.assert_called_once()

        # Verify collection.add was called
        mock_searcher.collection.add.assert_called_once()
        call_args = mock_searcher.collection.add.call_args

        # Check that correct number of documents were added
        documents = call_args.kwargs["documents"]
        metadatas = call_args.kwargs["metadatas"]
        ids = call_args.kwargs["ids"]

        assert len(documents) == 4  # 2 + 1 + 1 endpoints from sample schema
        assert len(metadatas) == 4
        assert len(ids) == 4

        # Check specific document content
        assert "GET /api/dcim/devices/ - List all devices in DCIM" in documents
        assert "POST /api/dcim/devices/ - Create a new device" in documents

        # Check metadata structure
        device_get_meta = next(
            m
            for m in metadatas
            if m["path"] == "/api/dcim/devices/" and m["method"] == "GET"
        )
        assert device_get_meta["operation_id"] == "dcim_devices_list"
        assert device_get_meta["description"] == "List all devices in DCIM"

    @patch("helpers.endpoint_searcher_chroma.config.SSL_VERIFY", False)
    @patch("helpers.endpoint_searcher_chroma.requests.get")
    def test_initialize_collection_no_token(
        self, mock_get, mock_searcher, sample_openapi_schema
    ):
        """Test initialization with no authentication token."""
        mock_searcher.token = ""
        mock_response = Mock()
        mock_response.json.return_value = sample_openapi_schema
        mock_get.return_value = mock_response

        mock_searcher.initialize_collection()

        # Verify API call without Authorization header
        mock_get.assert_called_once_with(
            "http://test.com/api/swagger.json", headers={}, verify=False
        )

    @patch("helpers.endpoint_searcher_chroma.requests.get")
    def test_initialize_collection_request_failure(self, mock_get, mock_searcher):
        """Test initialization with request failure."""
        mock_get.side_effect = requests.RequestException("Connection failed")

        # Should not raise exception, but log error
        mock_searcher.initialize_collection()

        mock_searcher.collection.add.assert_not_called()

    @patch("helpers.endpoint_searcher_chroma.requests.get")
    def test_initialize_collection_http_error(self, mock_get, mock_searcher):
        """Test initialization with HTTP error response."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
        mock_get.return_value = mock_response

        # Should not raise exception, but log error
        mock_searcher.initialize_collection()

        mock_searcher.collection.add.assert_not_called()

    @patch("helpers.endpoint_searcher_chroma.requests.get")
    def test_initialize_collection_invalid_json(self, mock_get, mock_searcher):
        """Test initialization with invalid JSON response."""
        mock_response = Mock()
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_get.return_value = mock_response

        # Should not raise exception, but log error
        mock_searcher.initialize_collection()

        mock_searcher.collection.add.assert_not_called()

    @patch("helpers.endpoint_searcher_chroma.requests.get")
    def test_initialize_collection_empty_paths(self, mock_get, mock_searcher):
        """Test initialization with empty paths in schema."""
        empty_schema = {"paths": {}}
        mock_response = Mock()
        mock_response.json.return_value = empty_schema
        mock_get.return_value = mock_response

        mock_searcher.initialize_collection()

        # Should still call add but with empty lists
        mock_searcher.collection.add.assert_called_once_with(
            documents=[], metadatas=[], ids=[]
        )

    def test_search_success(self, mock_searcher):
        """Test successful search operation."""
        # Setup mock query result
        mock_result = {
            "documents": [
                [
                    "GET /api/dcim/devices/ - List devices",
                    "POST /api/dcim/devices/ - Create device",
                ]
            ],
            "metadatas": [
                [
                    {
                        "path": "/api/dcim/devices/",
                        "method": "GET",
                        "operation_id": "dcim_devices_list",
                    },
                    {
                        "path": "/api/dcim/devices/",
                        "method": "POST",
                        "operation_id": "dcim_devices_create",
                    },
                ]
            ],
        }
        mock_searcher.collection.query.return_value = mock_result

        result = mock_searcher.search("find devices", n_results=2)

        # Verify query was called
        mock_searcher.collection.query.assert_called_once_with(
            query_texts=["find devices"], n_results=2
        )

        # Verify result structure
        assert result is not None
        assert len(result) == 2
        assert result[0]["document"] == "GET /api/dcim/devices/ - List devices"
        assert result[0]["metadata"]["method"] == "GET"
        assert result[1]["document"] == "POST /api/dcim/devices/ - Create device"
        assert result[1]["metadata"]["method"] == "POST"

    def test_search_no_results(self, mock_searcher):
        """Test search with no results."""
        mock_searcher.collection.query.return_value = {
            "documents": [[]],
            "metadatas": [[]],
        }

        result = mock_searcher.search("nonexistent endpoint")

        # The implementation returns an empty list when docs_list is empty
        assert result == []

    def test_search_empty_documents(self, mock_searcher):
        """Test search with empty documents list."""
        mock_searcher.collection.query.return_value = {"documents": [], "metadatas": []}

        result = mock_searcher.search("test query")

        assert result is None

    def test_search_malformed_result(self, mock_searcher):
        """Test search with malformed result structure."""
        mock_searcher.collection.query.return_value = {
            "documents": "not a list",
            "metadatas": "not a list",
        }

        result = mock_searcher.search("test query")

        assert result is None

    def test_search_missing_metadatas(self, mock_searcher):
        """Test search with missing metadatas."""
        mock_result = {
            "documents": [["GET /api/dcim/devices/ - List devices"]],
            "metadatas": None,
        }
        mock_searcher.collection.query.return_value = mock_result

        result = mock_searcher.search("find devices")

        # When metadatas is None, metas_list becomes empty, so zip() creates empty result
        assert result == []

    def test_search_empty_metadatas_list(self, mock_searcher):
        """Test search with empty metadatas list."""
        mock_result = {
            "documents": [["GET /api/dcim/devices/ - List devices"]],
            "metadatas": [[]],
        }
        mock_searcher.collection.query.return_value = mock_result

        result = mock_searcher.search("find devices")

        # When metadatas is empty list, zip() creates empty result
        assert result == []

    def test_search_with_partial_metadatas(self, mock_searcher):
        """Test search where documents and metadatas have different lengths."""
        mock_result = {
            "documents": [["Doc1", "Doc2", "Doc3"]],
            "metadatas": [
                [{"meta": "1"}, {"meta": "2"}]
            ],  # Only 2 metadatas for 3 docs
        }
        mock_searcher.collection.query.return_value = mock_result

        result = mock_searcher.search("find devices")

        # zip() stops at shortest iterable, so only 2 results
        assert result is not None
        assert len(result) == 2
        assert result[0]["document"] == "Doc1"
        assert result[0]["metadata"] == {"meta": "1"}
        assert result[1]["document"] == "Doc2"
        assert result[1]["metadata"] == {"meta": "2"}

    def test_search_collection_exception(self, mock_searcher):
        """Test search with collection query exception."""
        mock_searcher.collection.query.side_effect = Exception("ChromaDB error")

        result = mock_searcher.search("test query")

        assert result is None

    def test_search_default_n_results(self, mock_searcher):
        """Test search with default n_results parameter."""
        mock_searcher.collection.query.return_value = {
            "documents": [[]],
            "metadatas": [[]],
        }

        mock_searcher.search("test query")

        mock_searcher.collection.query.assert_called_once_with(
            query_texts=["test query"], n_results=5
        )

    def test_search_custom_n_results(self, mock_searcher):
        """Test search with custom n_results parameter."""
        mock_searcher.collection.query.return_value = {
            "documents": [[]],
            "metadatas": [[]],
        }

        mock_searcher.search("test query", n_results=10)

        mock_searcher.collection.query.assert_called_once_with(
            query_texts=["test query"], n_results=10
        )

    def test_metadata_cleaning_in_initialization(
        self, mock_searcher, sample_openapi_schema
    ):
        """Test that metadata cleaning is applied during initialization."""
        with patch("helpers.endpoint_searcher_chroma.requests.get") as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = sample_openapi_schema
            mock_get.return_value = mock_response

            mock_searcher.initialize_collection()

            call_args = mock_searcher.collection.add.call_args
            metadatas = call_args.kwargs["metadatas"]

            # Find metadata with parameters (should be serialized as JSON string)
            device_meta = next(
                m
                for m in metadatas
                if m["path"] == "/api/dcim/devices/" and m["method"] == "GET"
            )

            # Parameters should be serialized as JSON string
            assert isinstance(device_meta["parameters"], str)
            parsed_params = json.loads(device_meta["parameters"])
            assert len(parsed_params) == 1
            assert parsed_params[0]["name"] == "limit"

    def test_path_construction(self, mock_config, temp_db_path):
        """Test that database path is constructed correctly."""
        mock_config.EMBEDDING_MODEL = "test-model"

        with patch(
            "helpers.endpoint_searcher_chroma.get_chroma_db_path",
            return_value=temp_db_path,
        ):
            with (
                patch(
                    "helpers.endpoint_searcher_chroma.PersistentClient"
                ) as mock_client_class,
                patch(
                    "helpers.endpoint_searcher_chroma.SentenceTransformerEmbeddingFunction"
                ),
            ):
                searcher = EndpointSearcherChroma()

                expected_db_path = str(Path(temp_db_path) / "nautobot_api" / "db")
                assert searcher.db_path == expected_db_path

                # Verify client was initialized with correct path
                mock_client_class.assert_called_once()
                call_args = mock_client_class.call_args
                assert call_args.kwargs["path"] == expected_db_path

    def test_model_cache_dir_setup(self, mock_searcher):
        """Test that model cache directory is set up correctly."""
        # The actual path will be different due to how the module is loaded in tests
        # Just verify it contains the expected suffix
        assert mock_searcher.model_cache_dir.endswith("backend/models")

    @pytest.mark.parametrize(
        "query,expected_n_results",
        [
            ("devices", 5),
            ("locations", 3),
            ("circuits", 10),
            ("", 1),
        ],
    )
    def test_search_parameterized(self, mock_searcher, query, expected_n_results):
        """Parameterized test for search with different queries and result counts."""
        mock_searcher.collection.query.return_value = {
            "documents": [[]],
            "metadatas": [[]],
        }

        mock_searcher.search(query, n_results=expected_n_results)

        mock_searcher.collection.query.assert_called_once_with(
            query_texts=[query], n_results=expected_n_results
        )

    def test_integration_initialization_and_search(
        self, mock_config, mock_chroma_db_path, sample_openapi_schema
    ):
        """Integration test covering initialization and search workflow."""
        with (
            patch(
                "helpers.endpoint_searcher_chroma.PersistentClient"
            ) as mock_client_class,
            patch(
                "helpers.endpoint_searcher_chroma.SentenceTransformerEmbeddingFunction"
            ),
            patch("helpers.endpoint_searcher_chroma.requests.get") as mock_get,
        ):
            # Setup mocks
            mock_client = Mock()
            mock_collection = Mock()
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_client_class.return_value = mock_client

            mock_response = Mock()
            mock_response.json.return_value = sample_openapi_schema
            mock_get.return_value = mock_response

            # Create searcher and initialize
            searcher = EndpointSearcherChroma()
            searcher.initialize_collection()

            # Verify initialization
            mock_collection.add.assert_called_once()

            # Setup search mock
            mock_collection.query.return_value = {
                "documents": [["GET /api/dcim/devices/ - List devices"]],
                "metadatas": [[{"path": "/api/dcim/devices/", "method": "GET"}]],
            }

            # Perform search
            result = searcher.search("devices")

            # Verify search results
            assert result is not None
            assert len(result) == 1
            assert "devices" in result[0]["document"]
