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
    """Get Nautobot API endpoint schemas that match your intent. Returns endpoint details including path, method, parameters, and response formats."""
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
    """Execute direct HTTP requests to the Nautobot REST API for CRUD operations on network infrastructure data. Use this tool to: 1) Retrieve data (GET) - devices, locations, interfaces, IP addresses, etc., 2) Create new objects (POST) - add devices, create circuits, define custom fields, 3) Update existing objects (PUT/PATCH) - modify device properties, update interface configurations, 4) Delete objects (DELETE) - remove outdated devices, clean up unused data. Always use the API schema tool first to discover correct endpoints and required parameters. Supports filtering, pagination, and bulk operations through query parameters."""
    method = method.upper()
    params = params or {}
    body = body or {}

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

    response.raise_for_status()
    try:
        data = response.json()
    except Exception:
        data = {"response_text": response.text}

    return json.dumps(data, indent=2)


@mcp_app.tool()
async def mcp_refresh_endpoint_index() -> str:
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


async def main():
    # Use run_async() in async contexts
    await mcp_app.run_async(transport="streamable-http", port=8000)


if __name__ == "__main__":
    asyncio.run(main())
