# HTTP Streaming version using FastMCP for testing
# This is an alternative to server.py that uses HTTP transport instead of stdio

import asyncio
import logging
import os
from typing import Optional

import urllib3
from fastmcp import FastMCP

from helpers.endpoint_searcher_chroma import EndpointSearcherChroma
from helpers.nb_kb_v2 import EnhancedNautobotKnowledge
from helpers.tool_handlers import (
    handle_add_repo,
    handle_api_request_schema,
    handle_dynamic_api_request,
    handle_init_repos,
    handle_kb_semantic_search,
    handle_list_repos,
    handle_refresh_endpoint_index,
    handle_remove_repo,
    handle_repo_status,
    handle_update_repos,
)
from utils.config import config

# Disable SSL warnings if SSL verification is disabled
if not config.SSL_VERIFY:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL.upper()))
logger = logging.getLogger("nautobot_mcp_http")

# Set external service configurations
os.environ["POSTHOG_API_KEY"] = config.POSTHOG_API_KEY

# Initialize components
endpoint_searcher = EndpointSearcherChroma()
nautobot_kb = EnhancedNautobotKnowledge()

# Create FastMCP app
mcp_app = FastMCP(config.SERVER_NAME)


# Initialize at startup
async def startup():
    logger.info("HTTP MCP Server (ChromaDB Edition) starting")

    # Refresh endpoint index at startup
    logger.info("Refreshing endpoint index at startup...")
    endpoint_searcher.initialize_collection()
    logger.info("Endpoint index refreshed.")

    # Refresh Nautobot KB index at startup
    logger.info("Refreshing Nautobot KB index at startup...")
    nautobot_kb.initialize_all_repositories()
    logger.info("Nautobot KB index refreshed.")


# API Tools
@mcp_app.tool()
async def mcp_nautobot_openapi_api_request_schema(
    query: str, n_results: int = 5
) -> str:
    """Search for Nautobot API endpoints using natural language. Returns endpoint paths, methods, parameters, and response formats. Use this before making API requests to find the correct endpoint."""
    return await handle_api_request_schema(query, n_results, endpoint_searcher)


@mcp_app.tool()
async def mcp_nautobot_dynamic_api_request(
    method: str, path: str, params: Optional[dict] = None, body: Optional[dict] = None
) -> str:
    """Execute HTTP requests to Nautobot API for CRUD operations. Use GET to retrieve data, POST to create, PUT/PATCH to update, DELETE to remove. Query the API schema tool first to find correct endpoints and parameters. GET/DELETE use 'params', POST/PUT/PATCH use 'body'."""
    return await handle_dynamic_api_request(method, path, params, body)


@mcp_app.tool()
async def mcp_refresh_endpoint_index() -> str:
    """Manually refresh the OpenAPI endpoint index from the latest Nautobot schema."""
    return await handle_refresh_endpoint_index(endpoint_searcher)


# Knowledge Base Tools
@mcp_app.tool()
async def mcp_nautobot_kb_semantic_search(query: str, n_results: int = 5) -> str:
    """Search Nautobot GitHub repositories for code examples, best practices, and documentation. Use for: Jobs/Apps/Plugin examples, implementation patterns, API usage, feature guidance. Returns code snippets and docs with source attribution. Results truncated to 300 chars."""
    return await handle_kb_semantic_search(query, n_results, nautobot_kb)


# Repository Management Tools
@mcp_app.tool()
async def mcp_nautobot_kb_list_repos() -> str:
    """List all repositories configured in the Nautobot knowledge base with their metadata."""
    return await handle_list_repos()


@mcp_app.tool()
async def mcp_nautobot_kb_add_repo(repo: str, description: Optional[str] = None) -> str:
    """Add a new GitHub repository to the Nautobot knowledge base for indexing and search."""
    return await handle_add_repo(repo, description)


@mcp_app.tool()
async def mcp_nautobot_kb_remove_repo(repo: str) -> str:
    """Remove a repository from the Nautobot knowledge base configuration."""
    return await handle_remove_repo(repo)


@mcp_app.tool()
async def mcp_nautobot_kb_update_repos(
    repo: Optional[str] = None, force: bool = False
) -> str:
    """Update repository indexes in the knowledge base. Specify a repo to update one, or omit to update all. Use force=true to reindex even if unchanged."""
    return await handle_update_repos(nautobot_kb, repo, force)


@mcp_app.tool()
async def mcp_nautobot_kb_init_repos(force: bool = False) -> str:
    """Initialize all repositories in the knowledge base. Use force=true to reindex all repos."""
    return await handle_init_repos(nautobot_kb, force)


@mcp_app.tool()
async def mcp_nautobot_kb_repo_status() -> str:
    """Show repository status including document counts, indexing status, and configuration."""
    return await handle_repo_status(nautobot_kb)


async def main():
    # Use run_async() in async contexts
    await mcp_app.run_async(transport="streamable-http", port=8000)


if __name__ == "__main__":
    asyncio.run(main())
