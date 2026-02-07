"""
Common tool handlers for both server.py and server_http.py.
This module contains the implementation logic for all MCP tools to avoid code duplication.
"""

import json
import logging
from typing import Any, Dict, Optional

import requests

from helpers.endpoint_searcher_chroma import EndpointSearcherChroma
from helpers.graph_reranker import GraphReranker
from helpers.nb_kb_v2 import EnhancedNautobotKnowledge
from utils.config import config
from utils.repo_config import RepositoryConfig, RepositoryConfigManager

logger = logging.getLogger("nautobot_mcp")


async def handle_api_request_schema(
    query: str,
    n_results: int,
    endpoint_searcher: EndpointSearcherChroma,
    graph_reranker: Optional[GraphReranker] = None,
) -> str:
    """
    Handle API schema search requests with optional graph-based re-ranking.

    Args:
        query: Natural language query describing the desired API operation
        n_results: Number of results to return
        endpoint_searcher: Initialized EndpointSearcherChroma instance
        graph_reranker: Optional GraphReranker instance for intelligent re-ranking

    Returns:
        JSON string containing matching endpoints and base URL
    """
    logger.info(
        f"Searching endpoint index for query: '{query}' with n_results={n_results}"
    )

    # Get initial results from ChromaDB (fetch more for re-ranking)
    initial_results = n_results * 2 if graph_reranker and graph_reranker.enabled else n_results
    results = endpoint_searcher.search(query, n_results=initial_results)

    # Apply graph-based re-ranking if available
    if graph_reranker and graph_reranker.enabled and results:
        logger.debug("Applying graph-based re-ranking")
        results = graph_reranker.rerank(query, results, n_results=n_results)

    response_data = {
        "api_base_url": endpoint_searcher.base_url,
        "matching_endpoints": results,
        "reranked": graph_reranker and graph_reranker.enabled if results else False,
    }
    return json.dumps(response_data, indent=2)


async def handle_dynamic_api_request(
    method: str,
    path: str,
    params: Optional[Dict[str, Any]] = None,
    body: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Handle dynamic API requests to Nautobot.

    Args:
        method: HTTP method (GET, POST, PUT, PATCH, DELETE)
        path: API endpoint path
        params: Query parameters for GET/DELETE requests
        body: JSON body for POST/PUT/PATCH requests

    Returns:
        JSON string containing the API response

    Raises:
        ValueError: If unsupported HTTP method is provided
        requests.HTTPError: If API request fails
    """
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


async def handle_refresh_endpoint_index(
    endpoint_searcher: EndpointSearcherChroma,
) -> str:
    """
    Handle endpoint index refresh requests.

    Args:
        endpoint_searcher: Initialized EndpointSearcherChroma instance

    Returns:
        Success message string
    """
    logger.info("Manual endpoint index refresh triggered.")
    endpoint_searcher.initialize_collection()
    return "Endpoint index refreshed successfully."


async def handle_kb_semantic_search(
    query: str, n_results: int, nautobot_kb: EnhancedNautobotKnowledge
) -> str:
    """
    Handle semantic search in the Nautobot knowledge base.

    Args:
        query: Search query for Nautobot documentation and code
        n_results: Number of results to return
        nautobot_kb: Initialized EnhancedNautobotKnowledge instance

    Returns:
        JSON string containing search results with LLM optimization
    """
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


async def handle_list_repos() -> str:
    """
    Handle repository listing requests.

    Returns:
        JSON string containing list of configured repositories
    """
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
        }

    return json.dumps(response, indent=2)


async def handle_add_repo(repo: str, description: Optional[str] = None) -> str:
    """
    Handle repository addition requests.

    Args:
        repo: Repository in format 'owner/name'
        description: Optional repository description

    Returns:
        JSON string containing success/error status
    """
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


async def handle_remove_repo(repo: str) -> str:
    """
    Handle repository removal requests.

    Args:
        repo: Repository in format 'owner/name'

    Returns:
        JSON string containing success/error status
    """
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


async def handle_update_repos(
    nautobot_kb: EnhancedNautobotKnowledge,
    repo: Optional[str] = None,
    force: bool = False,
) -> str:
    """
    Handle repository update requests.

    Args:
        nautobot_kb: Initialized EnhancedNautobotKnowledge instance
        repo: Optional specific repository to update (owner/name)
        force: Force update even if no changes detected

    Returns:
        JSON string containing update results
    """
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


async def handle_init_repos(
    nautobot_kb: EnhancedNautobotKnowledge, force: bool = False
) -> str:
    """
    Handle repository initialization requests.

    Args:
        nautobot_kb: Initialized EnhancedNautobotKnowledge instance
        force: Force initialization of all repositories

    Returns:
        JSON string containing initialization results
    """
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


async def handle_repo_status(nautobot_kb: EnhancedNautobotKnowledge) -> str:
    """
    Handle repository status requests.

    Args:
        nautobot_kb: Initialized EnhancedNautobotKnowledge instance

    Returns:
        JSON string containing repository status information
    """
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
