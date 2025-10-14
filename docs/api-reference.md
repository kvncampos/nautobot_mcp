# API Reference

This page provides API documentation for the Nautobot MCP Server's key modules.

## Module Overview

The Nautobot MCP Server is organized into several key modules:

### Core Modules

#### `helpers/endpoint_searcher_chroma.py`

**EndpointSearcherChroma**: Manages OpenAPI schema indexing and semantic endpoint discovery.

Key Methods:

- `refresh_index()` - Fetch and index OpenAPI schema from Nautobot
- `search(query, n_results)` - Search for endpoints using semantic similarity
- `get_all_endpoints()` - Retrieve all indexed endpoints

#### `helpers/nb_kb_v2.py`

**EnhancedNautobotKnowledge**: Enhanced knowledge base for indexing and searching documentation.

Key Methods:

- `search(query, n_results)` - Basic semantic search
- `search_optimized_for_llm(query, n_results, max_content_length)` - Optimized search with content extraction
- `initialize_all_repositories(force)` - Initialize/update all configured repositories
- `get_repository_stats()` - Get statistics about indexed repositories

#### `helpers/content_processor.py`

**ContentProcessor**: Processes and optimizes document content for LLM consumption.

Key Methods:

- `extract_query_relevant_content(document, query, max_length)` - Extract query-relevant sections
- `extract_key_information(document, max_length)` - Extract key information from documents
- `truncate_content(content, max_length)` - Intelligently truncate content

### Utility Modules

#### `utils/config.py`

**Config**: Central configuration management from environment variables.

Key Properties:

- `NAUTOBOT_URL` - Active Nautobot instance URL
- `NAUTOBOT_TOKEN` - API authentication token
- `EMBEDDING_MODEL` - Sentence transformer model name
- `LOG_LEVEL` - Logging verbosity

#### `utils/embedding.py`

**SentenceTransformerEmbeddingFunction**: Wrapper for sentence-transformers models.

Key Methods:

- `__call__(texts)` - Generate embeddings for text list

#### `utils/repo_config.py`

**RepositoryConfigManager**: Manages repository configuration.

Key Methods:

- `load_repositories()` - Load all configured repositories
- `add_user_repository(repo_config)` - Add a user repository
- `remove_user_repository(repo_name)` - Remove a user repository

## Type Definitions

### Repository Configuration

```python
{
    "name": str,              # "owner/repo"
    "description": str,       # Human-readable description
    "priority": int,          # Search priority (lower = higher)
    "enabled": bool,          # Whether to index
    "branch": str,            # Branch to clone
    "file_patterns": List[str] # File extensions to index
}
```

### Search Result

```python
{
    "document": str,          # Document content
    "metadata": {
        "repository": str,    # Repository name
        "file_path": str,     # File path in repo
        "file_type": str,     # File extension
        "chunk_index": int    # Chunk number (if chunked)
    },
    "similarity_score": float # Similarity score (0-1)
}
```

### Endpoint Result

```python
{
    "method": str,           # HTTP method (GET, POST, etc.)
    "path": str,             # API endpoint path
    "description": str,      # Endpoint description
    "operationId": str,      # OpenAPI operation ID
    "similarity_score": float # Similarity score (0-1)
}
```

## MCP Tool Interface

All MCP tools follow this interface:

```python
async def handle_invoke_tool(
    name: str,              # Tool name
    inputs: Dict[str, Any]  # Tool parameters
) -> List[types.TextContent]:
    # Tool implementation
    result = perform_operation(inputs)
    return [types.TextContent(
        type="text",
        text=json.dumps(result, indent=2)
    )]
```

## Error Handling

All modules use consistent error responses:

```python
{
    "status": "error",
    "error": "Error description",
    "details": "Additional context"
}
```

## Logging

Modules use Python's standard logging:

```python
import logging
logger = logging.getLogger("nautobot_mcp")
logger.setLevel(logging.INFO)  # Configurable via LOG_LEVEL
```

---

## Code Examples

### Using Endpoint Searcher

```python
from helpers.endpoint_searcher_chroma import EndpointSearcherChroma

# Initialize
searcher = EndpointSearcherChroma()

# Refresh index
searcher.refresh_index()

# Search
results = searcher.search("device management", n_results=5)
for result in results:
    print(f"{result['method']} {result['path']}")
```

### Using Knowledge Base

```python
from helpers.nb_kb_v2 import EnhancedNautobotKnowledge

# Initialize
kb = EnhancedNautobotKnowledge()

# Search
results = kb.search("custom fields", n_results=3)
for result in results:
    print(f"Repository: {result['metadata']['repository']}")
    print(f"File: {result['metadata']['file_path']}")
    print(f"Content: {result['document'][:200]}...")
```

### Configuration Access

```python
from utils.config import config

print(f"Nautobot URL: {config.NAUTOBOT_URL}")
print(f"Environment: {config.NAUTOBOT_ENV}")
print(f"Model: {config.EMBEDDING_MODEL}")
```

---

For more information, see:

- [Tools Reference](tools.md) - MCP tool documentation
- [Architecture](architecture.md) - System design
- [Development Guide](development.md) - Contributing code
