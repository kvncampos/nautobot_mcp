# TODO Update the Tool _api_request_schema to actually return the complete endpoints as tools. Probably could use the FastMCP that has this as a built in feature. At the moment it does not return a tool per endpoint. (This is due to limiting the options in VSCode Copilot Chat)

import asyncio
import json
import logging
import os
from typing import Any, Dict

import mcp.server.stdio
import mcp.types as types
import requests
import urllib3
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions

from helpers.endpoint_searcher_chroma import EndpointSearcherChroma
from helpers.nb_kb_v2 import EnhancedNautobotKnowledge
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
from utils.config import config

# Disable SSL warnings if SSL verification is disabled
if not config.SSL_VERIFY:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL.upper()))
logger = logging.getLogger("nautobot_mcp")

# Set external service configurations
os.environ["POSTHOG_API_KEY"] = config.POSTHOG_API_KEY


async def main():
    logger.info("Any OpenAPI Server (ChromaDB Edition) starting")

    server = Server(config.API_PREFIX)
    endpoint_searcher = EndpointSearcherChroma()
    nautobot_kb = EnhancedNautobotKnowledge()
    # Refresh endpoint index at startup
    logger.info("Refreshing endpoint index at startup...")
    endpoint_searcher.initialize_collection()
    logger.info("Endpoint index refreshed.")
    # Refresh Nautobot KB index at startup
    logger.info("Refreshing Nautobot KB index at startup...")
    # Optionally set repo_list here, e.g. nautobot_kb.repo_list = ["nautobot/nautobot"]
    nautobot_kb.initialize_all_repositories()
    logger.info("Nautobot KB index refreshed.")

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


if __name__ == "__main__":
    asyncio.run(main())
