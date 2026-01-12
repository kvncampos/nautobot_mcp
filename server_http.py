# HTTP Streaming version using FastMCP for testing
# This is an alternative to server.py that uses HTTP transport instead of stdio

import asyncio
import json
import logging
import os
from typing import Optional

import requests
import urllib3
from fastmcp import FastMCP
from starlette.responses import JSONResponse

from helpers.endpoint_searcher_chroma import EndpointSearcherChroma
from helpers.nb_kb_v2 import EnhancedNautobotKnowledge
from utils.config import config
from utils.repo_config import RepositoryConfig, RepositoryConfigManager

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
    """Get Nautobot API endpoint schemas that match your intent. Returns endpoint details including path, method, parameters, and response formats.

    Agent usage:
    1) Call this tool first with the task stated plainly (e.g., "list devices by name").
    2) Take the `path` and method from the best match and pass them directly to `mcp_nautobot_dynamic_api_request`.
    3) Favor list endpoints with filters instead of guessing detail URLs; this avoids 404s when slugs/IDs are unknown.
    """
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
    return response_text


@mcp_app.tool()
async def mcp_nautobot_dynamic_api_request(
    method: str, path: str, params: Optional[dict] = None, body: Optional[dict] = None
) -> str:
    """Execute direct HTTP requests to the Nautobot REST API for CRUD operations on network infrastructure data.

    ⚠️ CRITICAL WORKFLOW - YOU MUST FOLLOW THIS ORDER:
    1. ALWAYS call 'mcp_nautobot_openapi_api_request_schema' FIRST with your goal (e.g., "list devices by name")
    2. Review the returned 'path' field from the schema response
    3. THEN call this tool with that exact 'path' value

    DO NOT GUESS ENDPOINT PATHS. If you call this tool without first calling the schema discovery tool, you will likely get 404 errors.

    Agent usage and retry guidance:
    - Prefer list endpoints with filters (e.g., params {"name": "device_1"}) instead of guessing detail URLs; this reduces 404s when the slug/ID is unknown.
    - If a request returns 404, re-run the schema tool and retry using the list+filter approach or the exact `url`/`path` provided by the schema response.
    - Always include the leading `/api/...` path and let this tool handle the host, auth headers, SSL, and timeouts.
    """
    method = method.upper()
    params = params or {}
    body = body or {}

    # Auto-fix: Ensure path starts with /api/ (common LLM mistake)
    if not path.startswith("/api/"):
        original_path = path
        path = f"/api/{path.lstrip('/')}"
        logger.warning(
            f"[PATH FIX] Auto-prepended /api/ to path: '{original_path}' -> '{path}'"
        )

    headers = config.get_headers()

    # For GET requests, always include depth=2 to get related object details
    if method == "GET":
        params = params.copy()
        params.setdefault("depth", 2)

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

    # Handle 404 gracefully - guide the agent to use discovery tool
    if response.status_code == 404:
        return json.dumps(
            {
                "error": "NOT_FOUND",
                "status_code": 404,
                "attempted_path": path,
                "guidance": (
                    "The path you requested does not exist. "
                    "You likely guessed the URL. "
                    "STOP and call `mcp_nautobot_openapi_api_request_schema` first with your goal "
                    "(e.g., 'list devices by name') to discover the correct endpoint. "
                    "Then use the returned `path` with list filters like {'name': 'device_1'} "
                    "instead of detail URLs."
                ),
            },
            indent=2,
        )

    # Handle other client errors with helpful context
    if response.status_code >= 400:
        try:
            error_body = response.json()
        except Exception:
            error_body = response.text
        return json.dumps(
            {
                "error": f"HTTP_{response.status_code}",
                "status_code": response.status_code,
                "attempted_path": path,
                "response": error_body,
            },
            indent=2,
        )

    try:
        data = response.json()
    except Exception:
        data = {"response_text": response.text}

    return json.dumps(data, indent=2)


@mcp_app.tool()
async def mcp_nautobot_refresh_endpoints_index() -> str:
    """Manually refresh the OpenAPI endpoint index from the latest schema."""
    logger.info("Manual endpoint index refresh triggered.")
    endpoint_searcher.initialize_collection()
    return "Endpoint index refreshed successfully."


# Knowledge Base Tools
@mcp_app.tool()
async def mcp_nautobot_kb_semantic_search(query: str, n_results: int = 5) -> str:
    """Semantic search over indexed Nautobot ecosystem GitHub repositories for code examples, best practices, and implementation patterns. Use this tool when you need: 1) Nautobot development best practices and patterns, 2) Code examples for Jobs, Apps, Plugins, or API usage, 3) Implementation guidance for Nautobot features, 4) Reference documentation from official Nautobot repositories. Query with specific technical terms (e.g. 'Nautobot Job example', 'custom field implementation', 'API serializer patterns', 'plugin development'). Searches official Nautobot repos, apps, and plugins. Returns code snippets, documentation, and examples with source attribution."""
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

    return response_text


# Repository Management Tools
@mcp_app.tool()
async def mcp_nautobot_kb_list_repos(repo_type: str = "all") -> str:
    """List configured repositories in the nautobot knowledge base."""
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

    return json.dumps(response, indent=2)


@mcp_app.tool()
async def mcp_nautobot_kb_add_repo(
    repo: str, description: Optional[str] = None, category: Optional[str] = None
) -> str:
    """Add a repository to the users nautobot knowledge base configuration."""
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

    return json.dumps(response, indent=2)


@mcp_app.tool()
async def mcp_nautobot_kb_remove_repo(repo: str) -> str:
    """Remove a repository from the users nautobot knowledge base configuration."""
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

    return json.dumps(response, indent=2)


@mcp_app.tool()
async def mcp_nautobot_kb_update_repos(
    repo: Optional[str] = None, force: bool = False
) -> str:
    """Update nautobot knowledge base repository indexes. Can update a specific repository or all repositories."""
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

    return json.dumps(response, indent=2)


@mcp_app.tool()
async def mcp_nautobot_kb_init_repos(force: bool = False) -> str:
    """Initialize all repositories in the users nautobot knowledge base."""
    try:
        results = nautobot_kb.initialize_all_repositories(force)
        updated_count = sum(results.values())

        response = {
            "status": "success",
            "message": f"Successfully initialized {updated_count}/{len(results)} repositories",
            "results": {
                repo_name: "updated" if was_updated else "skipped (up to date)"
                for repo_name, was_updated in results.items()
            },
            "force_enabled": force,
        }
    except Exception as e:
        response = {
            "status": "error",
            "message": f"Failed to initialize repositories: {str(e)}",
        }

    return json.dumps(response, indent=2)


@mcp_app.tool()
async def mcp_nautobot_kb_repo_status() -> str:
    """Show nautobot knowledge base repository status including document counts and indexing status."""
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
            "total_documents": sum(r["document_count"] for r in repos_status),
        }
    except Exception as e:
        response = {
            "status": "error",
            "message": f"Failed to get repository status: {str(e)}",
        }

    return json.dumps(response, indent=2)


# Health Check Endpoints
@mcp_app.custom_route("/health", methods=["GET"])
async def health_check(request):
    """HTTP health check endpoint that returns server status."""
    return JSONResponse(
        {
            "status": "healthy",
            "server": config.SERVER_NAME,
            "message": "Nautobot MCP Server is running",
        }
    )


@mcp_app.tool()
async def mcp_health_check() -> str:
    """MCP tool for health check that returns server status and uptime information."""
    return json.dumps(
        {
            "status": "healthy",
            "server": config.SERVER_NAME,
            "message": "Nautobot MCP Server is running",
        },
        indent=2,
    )


async def main():
    # Use run_async() with explicit path without trailing slash
    # This prevents FastAPI from issuing 307 redirects
    await mcp_app.run_async(
        transport="streamable-http",
        port=8000,
        path="/mcp",  # Explicit path without trailing slash - prevents redirect
    )


if __name__ == "__main__":
    asyncio.run(main())
