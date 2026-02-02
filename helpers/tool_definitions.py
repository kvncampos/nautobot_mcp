"""
Common tool definitions for both server.py and server_http.py.
This module contains improved tool descriptions and schemas following LLM best practices.
"""

from typing import Any, Dict

# Tool descriptions optimized for LLM ingestion
# Best practices:
# - Keep descriptions concise (under 300 characters for optimal LLM comprehension)
# - Use clear, actionable language
# - Include specific use cases
# - Avoid redundancy


def get_api_request_schema_description(global_tool_prompt: str = "") -> str:
    """
    Get description for API request schema tool.

    Args:
        global_tool_prompt: Optional global prompt to prepend

    Returns:
        Tool description string
    """
    base_description = (
        "Search for Nautobot API endpoints using natural language. "
        "Returns endpoint paths, methods, parameters, and response formats. "
        "Use this before making API requests to find the correct endpoint."
    )
    if global_tool_prompt and not global_tool_prompt.endswith(" "):
        global_tool_prompt += " "
    return f"{global_tool_prompt}{base_description}"


def get_dynamic_api_request_description() -> str:
    """Get description for dynamic API request tool."""
    return (
        "Execute HTTP requests to Nautobot API for CRUD operations. "
        "Use GET to retrieve data, POST to create, PUT/PATCH to update, DELETE to remove. "
        "Query the API schema tool first to find correct endpoints and parameters. "
        "GET/DELETE use 'params', POST/PUT/PATCH use 'body'."
    )


def get_refresh_endpoint_index_description() -> str:
    """Get description for refresh endpoint index tool."""
    return (
        "Manually refresh the OpenAPI endpoint index from the latest Nautobot schema."
    )


def get_kb_semantic_search_description() -> str:
    """Get description for knowledge base semantic search tool."""
    return (
        "Search Nautobot GitHub repositories for code examples, best practices, and documentation. "
        "Use for: Jobs/Apps/Plugin examples, implementation patterns, API usage, feature guidance. "
        "Returns code snippets and docs with source attribution. Results truncated to 300 chars."
    )


def get_list_repos_description() -> str:
    """Get description for list repositories tool."""
    return "List all repositories configured in the Nautobot knowledge base with their metadata."


def get_add_repo_description() -> str:
    """Get description for add repository tool."""
    return "Add a new GitHub repository to the Nautobot knowledge base for indexing and search."


def get_remove_repo_description() -> str:
    """Get description for remove repository tool."""
    return "Remove a repository from the Nautobot knowledge base configuration."


def get_update_repos_description() -> str:
    """Get description for update repositories tool."""
    return (
        "Update repository indexes in the knowledge base. "
        "Specify a repo to update one, or omit to update all. "
        "Use force=true to reindex even if unchanged."
    )


def get_init_repos_description() -> str:
    """Get description for initialize repositories tool."""
    return "Initialize all repositories in the knowledge base. Use force=true to reindex all repos."


def get_repo_status_description() -> str:
    """Get description for repository status tool."""
    return "Show repository status including document counts, indexing status, and configuration."


# Tool input schemas following JSON Schema specification
def get_api_request_schema_input_schema() -> Dict[str, Any]:
    """Get input schema for API request schema tool."""
    return {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Natural language query describing the API operation you want to perform "
                    "(e.g., 'get all devices', 'create a location', 'update interface status')"
                ),
            },
            "n_results": {
                "type": "integer",
                "description": "Number of matching endpoints to return (default: 5)",
                "default": 5,
                "minimum": 1,
                "maximum": 20,
            },
        },
        "required": ["query"],
    }


def get_dynamic_api_request_input_schema() -> Dict[str, Any]:
    """Get input schema for dynamic API request tool."""
    return {
        "type": "object",
        "properties": {
            "method": {
                "type": "string",
                "enum": ["GET", "POST", "PUT", "PATCH", "DELETE"],
                "description": (
                    "HTTP method: GET (retrieve data), POST (create new), "
                    "PUT (full update), PATCH (partial update), DELETE (remove)"
                ),
            },
            "path": {
                "type": "string",
                "description": (
                    "API endpoint path starting with '/' "
                    "(e.g., '/dcim/devices/', '/ipam/ip-addresses/'). "
                    "Use the API schema tool to discover valid paths."
                ),
            },
            "params": {
                "type": "object",
                "description": (
                    "Query parameters for GET/DELETE requests. "
                    "Examples: {'name': 'switch1'}, {'limit': 100}, {'offset': 50}"
                ),
            },
            "body": {
                "type": "object",
                "description": (
                    "JSON body for POST/PUT/PATCH requests. "
                    "Include required fields from the API schema. "
                    "Example: {'name': 'new-device', 'device_type': 1, 'site': 2}"
                ),
            },
        },
        "required": ["method", "path"],
    }


def get_refresh_endpoint_index_input_schema() -> Dict[str, Any]:
    """Get input schema for refresh endpoint index tool."""
    return {"type": "object", "properties": {}, "required": []}


def get_kb_semantic_search_input_schema() -> Dict[str, Any]:
    """Get input schema for knowledge base semantic search tool."""
    return {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Search query for Nautobot documentation and code. "
                    "Be specific about features or patterns you need "
                    "(e.g., 'Job with custom fields', 'GraphQL query examples', 'plugin models')"
                ),
            },
            "n_results": {
                "type": "integer",
                "description": "Number of results to return (default: 5)",
                "default": 5,
                "minimum": 1,
                "maximum": 20,
            },
        },
        "required": ["query"],
    }


def get_list_repos_input_schema() -> Dict[str, Any]:
    """Get input schema for list repositories tool."""
    return {
        "type": "object",
        "properties": {},
        "required": [],
    }


def get_add_repo_input_schema() -> Dict[str, Any]:
    """Get input schema for add repository tool."""
    return {
        "type": "object",
        "properties": {
            "repo": {
                "type": "string",
                "description": "GitHub repository in format 'owner/name' (e.g., 'nautobot/nautobot')",
            },
            "description": {
                "type": "string",
                "description": "Optional description of the repository",
            },
        },
        "required": ["repo"],
    }


def get_remove_repo_input_schema() -> Dict[str, Any]:
    """Get input schema for remove repository tool."""
    return {
        "type": "object",
        "properties": {
            "repo": {
                "type": "string",
                "description": "Repository to remove in format 'owner/name' (e.g., 'nautobot/nautobot')",
            },
        },
        "required": ["repo"],
    }


def get_update_repos_input_schema() -> Dict[str, Any]:
    """Get input schema for update repositories tool."""
    return {
        "type": "object",
        "properties": {
            "repo": {
                "type": "string",
                "description": (
                    "Specific repository to update in format 'owner/name'. "
                    "If omitted, all repositories will be updated."
                ),
            },
            "force": {
                "type": "boolean",
                "default": False,
                "description": "Force update even if repository appears unchanged",
            },
        },
        "required": [],
    }


def get_init_repos_input_schema() -> Dict[str, Any]:
    """Get input schema for initialize repositories tool."""
    return {
        "type": "object",
        "properties": {
            "force": {
                "type": "boolean",
                "default": False,
                "description": "Force initialization and reindexing of all repositories",
            },
        },
        "required": [],
    }


def get_repo_status_input_schema() -> Dict[str, Any]:
    """Get input schema for repository status tool."""
    return {
        "type": "object",
        "properties": {},
        "required": [],
    }
