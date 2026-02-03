"""Tests for the unified main.py entrypoint.

This module tests the unified entrypoint functionality including:
- Command-line argument parsing
- Environment variable handling
- Transport mode selection
- Component initialization
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def mock_env_vars():
    """Fixture to set up test environment variables."""
    env_vars = {
        "NAUTOBOT_ENV": "local",
        "LOG_LEVEL": "INFO",
        "API_PREFIX": "nautobot_openapi",
        "SERVER_NAME": "test_server",
        "SERVER_VERSION": "0.1.0",
        "POSTHOG_API_KEY": "disable",
        "SSL_VERIFY": "false",
        "MCP_TRANSPORT": "stdio",
        "MCP_PORT": "8000",
    }
    with patch.dict(os.environ, env_vars, clear=False):
        yield env_vars


@pytest.mark.unit
def test_parse_arguments_default():
    """Test argument parsing with defaults."""
    from main import parse_arguments

    with patch("sys.argv", ["main.py"]):
        args = parse_arguments()
        assert args.mode == "stdio"
        assert args.port == 8000


@pytest.mark.unit
def test_parse_arguments_stdio():
    """Test argument parsing for stdio mode."""
    from main import parse_arguments

    with patch("sys.argv", ["main.py", "--mode", "stdio"]):
        args = parse_arguments()
        assert args.mode == "stdio"


@pytest.mark.unit
def test_parse_arguments_http():
    """Test argument parsing for HTTP mode."""
    from main import parse_arguments

    with patch("sys.argv", ["main.py", "--mode", "http", "--port", "9000"]):
        args = parse_arguments()
        assert args.mode == "http"
        assert args.port == 9000


@pytest.mark.unit
def test_parse_arguments_env_vars(mock_env_vars):
    """Test argument parsing with environment variables."""
    from main import parse_arguments

    mock_env_vars["MCP_TRANSPORT"] = "http"
    mock_env_vars["MCP_PORT"] = "9000"

    with patch.dict(os.environ, mock_env_vars):
        with patch("sys.argv", ["main.py"]):
            args = parse_arguments()
            assert args.mode == "http"
            assert args.port == 9000


@pytest.mark.unit
def test_parse_arguments_cli_overrides_env(mock_env_vars):
    """Test that CLI arguments override environment variables."""
    from main import parse_arguments

    mock_env_vars["MCP_TRANSPORT"] = "http"
    mock_env_vars["MCP_PORT"] = "9000"

    with patch.dict(os.environ, mock_env_vars):
        with patch("sys.argv", ["main.py", "--mode", "stdio", "--port", "7000"]):
            args = parse_arguments()
            assert args.mode == "stdio"
            assert args.port == 7000


@pytest.mark.asyncio
@pytest.mark.unit
async def test_initialize_components(mock_env_vars):
    """Test component initialization."""
    from main import initialize_components

    with patch.dict(os.environ, mock_env_vars):
        # Mock the component classes
        with (
            patch("main.EndpointSearcherChroma") as mock_searcher_class,
            patch("main.EnhancedNautobotKnowledge") as mock_kb_class,
        ):
            # Create mock instances
            mock_searcher = MagicMock()
            mock_searcher.initialize_collection = MagicMock()
            mock_searcher_class.return_value = mock_searcher

            mock_kb = MagicMock()
            mock_kb.initialize_all_repositories = MagicMock()
            mock_kb_class.return_value = mock_kb

            # Run initialization
            searcher, kb = await initialize_components()

            # Verify components were created
            assert searcher is not None
            assert kb is not None
            mock_searcher.initialize_collection.assert_called_once()
            mock_kb.initialize_all_repositories.assert_called_once()


@pytest.mark.unit
def test_imports():
    """Test that main.py can be imported without errors."""
    import main

    assert hasattr(main, "main")
    assert hasattr(main, "parse_arguments")
    assert hasattr(main, "initialize_components")
    assert hasattr(main, "run_stdio_mode")
    assert hasattr(main, "run_http_mode")


@pytest.mark.unit
def test_mode_choices():
    """Test that only valid modes are accepted."""
    from main import parse_arguments

    # Valid modes
    for mode in ["stdio", "http"]:
        with patch("sys.argv", ["main.py", "--mode", mode]):
            args = parse_arguments()
            assert args.mode == mode

    # Invalid mode should raise error
    with patch("sys.argv", ["main.py", "--mode", "invalid"]):
        with pytest.raises(SystemExit):
            parse_arguments()


@pytest.mark.unit
def test_port_validation():
    """Test port number validation."""
    from main import parse_arguments

    # Valid port
    with patch("sys.argv", ["main.py", "--port", "8080"]):
        args = parse_arguments()
        assert args.port == 8080

    # Invalid port (non-numeric) should raise error
    with patch("sys.argv", ["main.py", "--port", "invalid"]):
        with pytest.raises(SystemExit):
            parse_arguments()


@pytest.mark.unit
def test_backward_compatibility_server_py():
    """Test that server.py still works (backward compatibility)."""
    # Just verify it can be imported
    import server

    assert hasattr(server, "main")


@pytest.mark.unit
def test_backward_compatibility_server_http_py():
    """Test that server_http.py still works (backward compatibility)."""
    # Just verify it can be imported
    import server_http

    assert hasattr(server_http, "main")
    assert hasattr(server_http, "mcp_app")


@pytest.mark.unit
def test_help_message():
    """Test that help message is displayed correctly."""
    from main import parse_arguments

    with patch("sys.argv", ["main.py", "--help"]):
        with pytest.raises(SystemExit) as exc_info:
            parse_arguments()
        # argparse exits with 0 for help
        assert exc_info.value.code == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
