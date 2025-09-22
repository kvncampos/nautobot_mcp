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
from utils.config import config
from utils.repo_config import RepositoryConfig, RepositoryConfigManager

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
                description=(
                    f"{global_tool_prompt}Get Nautobot API endpoint schemas that match your intent. "
                    "Returns endpoint details including path, method, parameters, and response formats."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "Describe what you want to do with the API "
                                "(e.g., 'Get location count', 'Delete a device', etc.)"
                            ),
                        },
                        "n_results": {
                            "type": "integer",
                            "description": "Number of results to return (default: 5)",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            ),
            types.Tool(
                name="nautobot_dynamic_api_request",
                description=(
                    "Execute direct HTTP requests to the Nautobot REST API for CRUD operations on network infrastructure data. "
                    "Use this tool to: 1) Retrieve data (GET) - devices, locations, interfaces, IP addresses, etc., "
                    "2) Create new objects (POST) - add devices, create circuits, define custom fields, "
                    "3) Update existing objects (PUT/PATCH) - modify device properties, update interface configurations, "
                    "4) Delete objects (DELETE) - remove outdated devices, clean up unused data. "
                    "Always use the API schema tool first to discover correct endpoints and required parameters. "
                    "Supports filtering, pagination, and bulk operations through query parameters."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "method": {
                            "type": "string",
                            "enum": ["GET", "POST", "PUT", "PATCH", "DELETE"],
                            "description": "HTTP method: GET (retrieve), POST (create), PUT (full update), PATCH (partial update), DELETE (remove).",
                        },
                        "path": {
                            "type": "string",
                            "description": "Nautobot API endpoint path starting with '/' (e.g., '/dcim/devices/', '/ipam/ip-addresses/', '/circuits/circuits/'). Use the API schema tool to discover valid paths.",
                        },
                        "params": {
                            "type": "object",
                            "description": "Query parameters for filtering, pagination, or field selection. Examples: {'name__icontains': 'switch'}, {'limit': 100}, {'offset': 50}. Used with GET/DELETE requests.",
                        },
                        "body": {
                            "type": "object",
                            "description": "JSON payload for creating or updating objects. Include required fields and any optional data. Use with POST/PUT/PATCH requests. Structure varies by endpoint - check API schema first.",
                        },
                    },
                    "required": ["method", "path"],
                },
            ),
            types.Tool(
                name="refresh_endpoint_index",
                description="Manually refresh the OpenAPI endpoint index from the latest schema.",
                inputSchema={"type": "object", "properties": {}, "required": []},
            ),
            # -----------------------------------
            # ---------- KB TOOLS ---------------
            # -----------------------------------
            types.Tool(
                name="nautobot_kb_semantic_search",
                description=(
                    "Semantic search over indexed Nautobot ecosystem GitHub repositories for code examples, "
                    "best practices, and implementation patterns. Use this tool when you need: "
                    "1) Nautobot development best practices and patterns, "
                    "2) Code examples for Jobs, Apps, Plugins, or API usage, "
                    "3) Implementation guidance for Nautobot features, "
                    "4) Reference documentation from official Nautobot repositories. "
                    "Query with specific technical terms (e.g. 'Nautobot Job example', 'custom field implementation', "
                    "'API serializer patterns', 'plugin development'). Searches official Nautobot repos, apps, and plugins. "
                    "Returns code snippets, documentation, and examples with source attribution."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Detailed search query describing what Nautobot development knowledge you need. Be specific about the feature, component, or pattern you're looking for (e.g., 'Job implementation with custom fields', 'GraphQL query examples', 'plugin model relationships').",
                        },
                        "n_results": {
                            "type": "integer",
                            "description": "Number of results to return (default: 5)",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            ),
            # -----------------------------------
            # ------- REPO MANAGEMENT TOOLS ----
            # -----------------------------------
            types.Tool(
                name="nautobot_kb_list_repos",
                description="List configured repositories in the nautobot knowledge base.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "repo_type": {
                            "type": "string",
                            "enum": ["official", "user", "all"],
                            "default": "all",
                            "description": "Type of repositories to list",
                        },
                    },
                    "required": [],
                },
            ),
            types.Tool(
                name="nautobot_kb_add_repo",
                description="Add a repository to the users nautobot knowledge base configuration.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "repo": {
                            "type": "string",
                            "description": "Repository in format 'owner/name'",
                        },
                        "description": {
                            "type": "string",
                            "description": "Repository description",
                        },
                        "category": {
                            "type": "string",
                            "description": "Repository category",
                        },
                    },
                    "required": ["repo"],
                },
            ),
            types.Tool(
                name="nautobot_kb_remove_repo",
                description="Remove a repository from the users nautobot knowledge base configuration.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "repo": {
                            "type": "string",
                            "description": "Repository in format 'owner/name'",
                        },
                    },
                    "required": ["repo"],
                },
            ),
            types.Tool(
                name="nautobot_kb_update_repos",
                description="Update nautobot knowledge base repository indexes. Can update a specific repository or all repositories.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "repo": {
                            "type": "string",
                            "description": "Specific repository to update (owner/name). If not provided, all repos will be updated.",
                        },
                        "force": {
                            "type": "boolean",
                            "default": False,
                            "description": "Force update even if no changes",
                        },
                    },
                    "required": [],
                },
            ),
            types.Tool(
                name="nautobot_kb_init_repos",
                description="Initialize all repositories in the users nautobot knowledge base.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "force": {
                            "type": "boolean",
                            "default": False,
                            "description": "Force initialization of all repos",
                        },
                    },
                    "required": [],
                },
            ),
            types.Tool(
                name="nautobot_kb_repo_status",
                description="Show nautobot knowledge base repository status including document counts and indexing status.",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
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
                logger.info(
                    f"Searching endpoint index for query: '{query}' with n_results={n_results}"
                )
                results = endpoint_searcher.search(query, n_results=n_results)
                response_text = json.dumps(
                    {
                        "api_base_url": endpoint_searcher.base_url,
                        "matching_endpoints": results,
                    },
                    indent=2,
                )
                return [types.TextContent(type="text", text=response_text)]
            elif name == "refresh_endpoint_index":
                logger.info("Manual endpoint index refresh triggered.")
                endpoint_searcher.initialize_collection()
                return [
                    types.TextContent(
                        type="text", text="Endpoint index refreshed successfully."
                    )
                ]

            elif name == "nautobot_dynamic_api_request":
                method = inputs["method"].upper()
                path = inputs["path"]
                params = inputs.get("params", {})
                body = inputs.get("body", {})

                headers = config.get_headers()

                # For GET requests, always include depth=2 to get related object details
                if method == "GET":
                    params = params.copy()  # Don't modify the original params dict
                    params.setdefault(
                        "depth", 2
                    )  # Set depth=2 if not already specified

                full_url = config.get_full_url(path)
                logger.info(f"[nautobot_dynamic_api_request] {method} {full_url}")

                response = None
                if method == "GET":
                    response = requests.get(
                        full_url,
                        headers=headers,
                        params=params,
                        timeout=config.API_TIMEOUT,
                        verify=config.SSL_VERIFY,
                    )
                elif method == "POST":
                    response = requests.post(
                        full_url,
                        headers=headers,
                        json=body,
                        timeout=config.API_TIMEOUT,
                        verify=config.SSL_VERIFY,
                    )
                elif method == "PUT":
                    response = requests.put(
                        full_url,
                        headers=headers,
                        json=body,
                        timeout=config.API_TIMEOUT,
                        verify=config.SSL_VERIFY,
                    )
                elif method == "PATCH":
                    response = requests.patch(
                        full_url,
                        headers=headers,
                        json=body,
                        timeout=config.API_TIMEOUT,
                        verify=config.SSL_VERIFY,
                    )
                elif method == "DELETE":
                    response = requests.delete(
                        full_url,
                        headers=headers,
                        params=params,
                        timeout=config.API_TIMEOUT,
                        verify=config.SSL_VERIFY,
                    )
                else:
                    raise ValueError(f"Unsupported method: {method}")

                response.raise_for_status()
                try:
                    data = response.json()
                except Exception:
                    data = {"response_text": response.text}

                return [types.TextContent(type="text", text=json.dumps(data, indent=2))]

            # ----------------------------------------
            # ---------- CALL KB TOOLS ---------------
            # ----------------------------------------
            elif name == "nautobot_kb_semantic_search":
                query = inputs["query"]
                n_results = inputs.get("n_results", 5)

                logger.info(
                    f"LLM-optimized semantic search for query: '{query}' with n_results={n_results}"
                )

                # Use the optimized search method directly
                optimized_results = nautobot_kb.search_optimized_for_llm(
                    query=query,
                    n_results=n_results,
                    max_content_length=300,  # Optimal length for LLM consumption
                )

                if optimized_results:
                    response_data = {
                        "results": optimized_results,
                        "processing_info": {
                            "total_results": len(optimized_results),
                            "llm_optimized": True,
                            "query": query,
                        },
                    }
                    response_text = json.dumps(response_data, indent=2)
                else:
                    response_text = json.dumps([])

                return [types.TextContent(type="text", text=response_text)]

            # ----------------------------------------
            # -------- REPO MANAGEMENT TOOLS --------
            # ----------------------------------------
            elif name == "nautobot_kb_list_repos":
                repo_type = inputs.get("repo_type", "all")
                config_manager = RepositoryConfigManager()
                all_repos = config_manager.load_repositories()

                if not all_repos:
                    response = {"message": "No repositories configured."}
                else:
                    repos_data = []
                    for repo in all_repos:
                        repos_data.append(
                            {
                                "name": repo.name,
                                "description": repo.description,
                                "priority": repo.priority,
                                "enabled": repo.enabled,
                                "branch": repo.branch,
                                "file_patterns": repo.file_patterns,
                            }
                        )

                    response = {
                        "repositories": repos_data,
                        "total_count": len(repos_data),
                        "filter_type": repo_type,
                    }

                response_text = json.dumps(response, indent=2)
                return [types.TextContent(type="text", text=response_text)]

            elif name == "nautobot_kb_add_repo":
                repo = inputs["repo"]
                description = inputs.get("description")
                # category parameter is accepted but not currently used in RepositoryConfig

                config_manager = RepositoryConfigManager()

                try:
                    repo_config = RepositoryConfig(
                        name=repo,
                        description=description or f"User-added repository: {repo}",
                        priority=5,
                        enabled=True,
                        branch="main",
                        file_patterns=[".py", ".md", ".txt", ".rst", ".json"],
                    )

                    success = config_manager.add_user_repository(repo_config)
                    if success:
                        response = {
                            "status": "success",
                            "message": f"Added repository: {repo}",
                        }
                    else:
                        response = {
                            "status": "error",
                            "message": f"Failed to add repository {repo} (may already exist)",
                        }
                except Exception as e:
                    response = {
                        "status": "error",
                        "message": f"Failed to add repository {repo}: {str(e)}",
                    }

                response_text = json.dumps(response, indent=2)
                return [types.TextContent(type="text", text=response_text)]

            elif name == "nautobot_kb_remove_repo":
                repo = inputs["repo"]
                config_manager = RepositoryConfigManager()

                try:
                    success = config_manager.remove_user_repository(repo)
                    if success:
                        response = {
                            "status": "success",
                            "message": f"Removed repository: {repo}",
                        }
                    else:
                        response = {
                            "status": "error",
                            "message": f"Failed to remove repository {repo} (may not exist)",
                        }
                except Exception as e:
                    response = {
                        "status": "error",
                        "message": f"Failed to remove repository {repo}: {str(e)}",
                    }

                response_text = json.dumps(response, indent=2)
                return [types.TextContent(type="text", text=response_text)]

            elif name == "nautobot_kb_update_repos":
                repo = inputs.get("repo")
                force = inputs.get("force", False)
                config_manager = RepositoryConfigManager()

                try:
                    if repo:
                        repo_config = config_manager.get_repo_config(repo)
                        if not repo_config:
                            response = {
                                "status": "error",
                                "message": f"Repository {repo} not found in configuration",
                            }
                        else:
                            success = nautobot_kb.update_repository(repo_config, force)
                            if success:
                                response = {
                                    "status": "success",
                                    "message": f"Updated {repo}",
                                }
                            else:
                                response = {
                                    "status": "info",
                                    "message": f"{repo} was already up to date or failed to update",
                                }
                    else:
                        results = nautobot_kb.initialize_all_repositories(force)
                        updated_count = sum(results.values())
                        response = {
                            "status": "success",
                            "message": f"Updated {updated_count}/{len(results)} repositories",
                            "results": {
                                repo_name: "updated" if was_updated else "skipped"
                                for repo_name, was_updated in results.items()
                            },
                        }
                except Exception as e:
                    response = {
                        "status": "error",
                        "message": f"Failed to update repositories: {str(e)}",
                    }

                response_text = json.dumps(response, indent=2)
                return [types.TextContent(type="text", text=response_text)]

            elif name == "nautobot_kb_init_repos":
                force = inputs.get("force", False)

                try:
                    results = nautobot_kb.initialize_all_repositories(force)
                    updated_count = sum(results.values())

                    response = {
                        "status": "success",
                        "message": f"Successfully initialized {updated_count}/{len(results)} repositories",
                        "results": {
                            repo_name: "updated"
                            if was_updated
                            else "skipped (up to date)"
                            for repo_name, was_updated in results.items()
                        },
                        "force_enabled": force,
                    }
                except Exception as e:
                    response = {
                        "status": "error",
                        "message": f"Failed to initialize repositories: {str(e)}",
                    }

                response_text = json.dumps(response, indent=2)
                return [types.TextContent(type="text", text=response_text)]

            elif name == "nautobot_kb_repo_status":
                config_manager = RepositoryConfigManager()
                all_repos = config_manager.load_repositories()

                try:
                    stats = nautobot_kb.get_repository_stats()

                    repos_status = []
                    for repo in all_repos:
                        repo_stats = stats.get(repo.name, {})
                        doc_count = repo_stats.get("document_count", 0)
                        is_enabled = repo_stats.get("enabled", False)
                        status = "indexed" if doc_count > 0 else "not_indexed"

                        repo_status = {
                            "name": repo.name,
                            "document_count": doc_count,
                            "status": status,
                            "enabled": is_enabled,
                            "priority": repo.priority,
                            "branch": repo.branch,
                        }

                        if "error" in repo_stats:
                            repo_status["error"] = repo_stats["error"]

                        repos_status.append(repo_status)

                    response = {
                        "repositories": repos_status,
                        "total_repositories": len(repos_status),
                        "indexed_repositories": sum(
                            1 for r in repos_status if r["status"] == "indexed"
                        ),
                        "total_documents": sum(
                            r["document_count"] for r in repos_status
                        ),
                    }
                except Exception as e:
                    response = {
                        "status": "error",
                        "message": f"Failed to get repository status: {str(e)}",
                    }

                response_text = json.dumps(response, indent=2)
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
