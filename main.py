"""Unified entrypoint for Nautobot MCP Server.

This module provides a unified entrypoint that supports both stdio and HTTP transport modes.
The mode can be selected via command-line arguments or environment variables.

Usage:
    # stdio mode (default)
    python main.py --mode stdio

    # HTTP mode
    python main.py --mode http --port 8000

    # Using environment variables
    MCP_TRANSPORT=http MCP_PORT=8000 python main.py
"""

import argparse
import asyncio
import logging
import os
import sys
from typing import Optional

import urllib3

from helpers.endpoint_searcher_chroma import EndpointSearcherChroma
from helpers.nb_kb_v2 import EnhancedNautobotKnowledge
from utils.config import config

# Disable SSL warnings if SSL verification is disabled
if not config.SSL_VERIFY:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL.upper()))
logger = logging.getLogger("nautobot_mcp")

# Set external service configurations
os.environ["POSTHOG_API_KEY"] = config.POSTHOG_API_KEY


async def initialize_components() -> tuple[
    EndpointSearcherChroma, EnhancedNautobotKnowledge
]:
    """Initialize shared components used by both transport modes.

    Returns:
        tuple: (endpoint_searcher, nautobot_kb) instances
    """
    logger.info("Initializing Nautobot MCP Server components...")

    endpoint_searcher = EndpointSearcherChroma()
    nautobot_kb = EnhancedNautobotKnowledge()

    # Refresh endpoint index at startup
    logger.info("Refreshing endpoint index at startup...")
    endpoint_searcher.initialize_collection()
    logger.info("Endpoint index refreshed.")

    # Refresh Nautobot KB index at startup
    logger.info("Refreshing Nautobot KB index at startup...")
    nautobot_kb.initialize_all_repositories()
    logger.info("Nautobot KB index refreshed.")

    return endpoint_searcher, nautobot_kb


async def run_stdio_mode():
    """Run the MCP server in stdio transport mode."""
    import json
    from typing import Any, Dict

    import mcp.server.stdio
    import mcp.types as types
    from mcp.server import NotificationOptions, Server
    from mcp.server.models import InitializationOptions

    from helpers.tool_definitions import (
        get_add_repo_description,
        get_add_repo_input_schema,
        get_api_request_schema_description,
        get_api_request_schema_input_schema,
        get_dynamic_api_request_description,
        get_dynamic_api_request_input_schema,
        get_init_repos_description,
        get_init_repos_input_schema,
        get_kb_semantic_search_description,
        get_kb_semantic_search_input_schema,
        get_list_repos_description,
        get_list_repos_input_schema,
        get_refresh_endpoint_index_description,
        get_refresh_endpoint_index_input_schema,
        get_remove_repo_description,
        get_remove_repo_input_schema,
        get_repo_status_description,
        get_repo_status_input_schema,
        get_update_repos_description,
        get_update_repos_input_schema,
    )
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

    logger.info("Starting MCP Server in stdio mode...")

    # Initialize shared components
    endpoint_searcher, nautobot_kb = await initialize_components()

    server = Server(config.API_PREFIX)

    global_tool_prompt = config.GLOBAL_TOOL_PROMPT
    if global_tool_prompt and not global_tool_prompt.endswith(" "):
        global_tool_prompt += " "

    @server.list_tools()
    async def handle_list_tools() -> list[types.Tool]:
        return [
            # ---------------------------------------
            # ---------- NB API TOOLS ---------------
            # ---------------------------------------
            types.Tool(
                name=f"{config.API_PREFIX}_api_request_schema",
                description=get_api_request_schema_description(global_tool_prompt),
                inputSchema=get_api_request_schema_input_schema(),
            ),
            types.Tool(
                name="nautobot_dynamic_api_request",
                description=get_dynamic_api_request_description(),
                inputSchema=get_dynamic_api_request_input_schema(),
            ),
            types.Tool(
                name="refresh_endpoint_index",
                description=get_refresh_endpoint_index_description(),
                inputSchema=get_refresh_endpoint_index_input_schema(),
            ),
            # -----------------------------------
            # ---------- KB TOOLS ---------------
            # -----------------------------------
            types.Tool(
                name="nautobot_kb_semantic_search",
                description=get_kb_semantic_search_description(),
                inputSchema=get_kb_semantic_search_input_schema(),
            ),
            # -----------------------------------
            # ------- REPO MANAGEMENT TOOLS ----
            # -----------------------------------
            types.Tool(
                name="nautobot_kb_list_repos",
                description=get_list_repos_description(),
                inputSchema=get_list_repos_input_schema(),
            ),
            types.Tool(
                name="nautobot_kb_add_repo",
                description=get_add_repo_description(),
                inputSchema=get_add_repo_input_schema(),
            ),
            types.Tool(
                name="nautobot_kb_remove_repo",
                description=get_remove_repo_description(),
                inputSchema=get_remove_repo_input_schema(),
            ),
            types.Tool(
                name="nautobot_kb_update_repos",
                description=get_update_repos_description(),
                inputSchema=get_update_repos_input_schema(),
            ),
            types.Tool(
                name="nautobot_kb_init_repos",
                description=get_init_repos_description(),
                inputSchema=get_init_repos_input_schema(),
            ),
            types.Tool(
                name="nautobot_kb_repo_status",
                description=get_repo_status_description(),
                inputSchema=get_repo_status_input_schema(),
            ),
        ]

    @server.call_tool()
    async def handle_invoke_tool(
        name: str, inputs: Dict[str, Any]
    ) -> list[types.TextContent]:
        try:
            # --------------------------------------------
            # ---------- Call NB API TOOLS ---------------
            # --------------------------------------------
            if name == f"{config.API_PREFIX}_api_request_schema":
                query = inputs["query"]
                n_results = inputs.get("n_results", config.DEFAULT_SEARCH_RESULTS)
                response_text = await handle_api_request_schema(
                    query, n_results, endpoint_searcher
                )
                return [types.TextContent(type="text", text=response_text)]

            elif name == "refresh_endpoint_index":
                response_text = await handle_refresh_endpoint_index(endpoint_searcher)
                return [types.TextContent(type="text", text=response_text)]

            elif name == "nautobot_dynamic_api_request":
                method = inputs["method"]
                path = inputs["path"]
                params = inputs.get("params")
                body = inputs.get("body")
                response_text = await handle_dynamic_api_request(
                    method, path, params, body
                )
                return [types.TextContent(type="text", text=response_text)]

            # ----------------------------------------
            # ---------- CALL KB TOOLS ---------------
            # ----------------------------------------
            elif name == "nautobot_kb_semantic_search":
                query = inputs["query"]
                n_results = inputs.get("n_results", 5)
                response_text = await handle_kb_semantic_search(
                    query, n_results, nautobot_kb
                )
                return [types.TextContent(type="text", text=response_text)]

            # ----------------------------------------
            # -------- REPO MANAGEMENT TOOLS --------
            # ----------------------------------------
            elif name == "nautobot_kb_list_repos":
                response_text = await handle_list_repos()
                return [types.TextContent(type="text", text=response_text)]

            elif name == "nautobot_kb_add_repo":
                repo = inputs["repo"]
                description = inputs.get("description")
                response_text = await handle_add_repo(repo, description)
                return [types.TextContent(type="text", text=response_text)]

            elif name == "nautobot_kb_remove_repo":
                repo = inputs["repo"]
                response_text = await handle_remove_repo(repo)
                return [types.TextContent(type="text", text=response_text)]

            elif name == "nautobot_kb_update_repos":
                repo = inputs.get("repo")
                force = inputs.get("force", False)
                response_text = await handle_update_repos(nautobot_kb, repo, force)
                return [types.TextContent(type="text", text=response_text)]

            elif name == "nautobot_kb_init_repos":
                force = inputs.get("force", False)
                response_text = await handle_init_repos(nautobot_kb, force)
                return [types.TextContent(type="text", text=response_text)]

            elif name == "nautobot_kb_repo_status":
                response_text = await handle_repo_status(nautobot_kb)
                return [types.TextContent(type="text", text=response_text)]

            else:
                raise ValueError(f"Unknown tool: {name}")

        except Exception as e:
            logger.exception(f"Error invoking tool {name}")
            error_text = json.dumps({"error": str(e)}, indent=2)
            return [types.TextContent(type="text", text=error_text)]

    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        logger.info("Server running with stdio transport")
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name=config.SERVER_NAME,
                server_version=config.SERVER_VERSION,
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


async def run_http_mode(port: int = 8000):
    """Run the MCP server in HTTP transport mode using FastMCP.

    Args:
        port: Port number to listen on (default: 8000)
    """

    from fastmcp import FastMCP

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

    logger.info(f"Starting MCP Server in HTTP mode on port {port}...")

    # Initialize shared components
    endpoint_searcher, nautobot_kb = await initialize_components()

    # Create FastMCP app
    mcp_app = FastMCP(config.SERVER_NAME)

    # API Tools
    @mcp_app.tool()
    async def mcp_nautobot_openapi_api_request_schema(
        query: str, n_results: int = 5
    ) -> str:
        """Search for Nautobot API endpoints using natural language. Returns endpoint paths, methods, parameters, and response formats. Use this before making API requests to find the correct endpoint."""
        return await handle_api_request_schema(query, n_results, endpoint_searcher)

    @mcp_app.tool()
    async def mcp_nautobot_dynamic_api_request(
        method: str,
        path: str,
        params: Optional[dict] = None,
        body: Optional[dict] = None,
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
    async def mcp_nautobot_kb_add_repo(
        repo: str, description: Optional[str] = None
    ) -> str:
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

    # Run the HTTP server
    await mcp_app.run_async(transport="streamable-http", port=port)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Nautobot MCP Server - Unified entrypoint for stdio and HTTP modes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run in stdio mode (default)
  python main.py
  python main.py --mode stdio

  # Run in HTTP mode on default port (8000)
  python main.py --mode http

  # Run in HTTP mode on custom port
  python main.py --mode http --port 9000

  # Using environment variables
  MCP_TRANSPORT=http MCP_PORT=8000 python main.py
        """,
    )

    parser.add_argument(
        "--mode",
        "-m",
        choices=["stdio", "http"],
        default=os.getenv("MCP_TRANSPORT", "stdio"),
        help="Transport mode: stdio (default) or http",
    )

    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=int(os.getenv("MCP_PORT", "8000")),
        help="Port number for HTTP mode (default: 8000)",
    )

    return parser.parse_args()


async def main():
    """Main entrypoint that dispatches to the appropriate transport mode."""
    args = parse_arguments()

    logger.info(f"Nautobot MCP Server v{config.SERVER_VERSION}")
    logger.info(f"Transport mode: {args.mode}")

    try:
        if args.mode == "stdio":
            await run_stdio_mode()
        elif args.mode == "http":
            await run_http_mode(port=args.port)
        else:
            logger.error(f"Invalid mode: {args.mode}")
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Server shutting down...")
    except Exception:
        logger.exception("Fatal error in main entrypoint")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
