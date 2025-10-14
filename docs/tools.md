# MCP Tools Reference

The Nautobot MCP Server provides a comprehensive set of tools for interacting with Nautobot APIs and searching documentation. This page documents all available tools, their parameters, and usage examples.

## Tool Categories

- [API Discovery & Execution](#api-tools)
- [Knowledge Base Search](#knowledge-base-tools)
- [Repository Management](#repository-management-tools)
- [System Tools](#system-tools)

---

## API Tools

### `nautobot_openapi_api_request_schema`

Search for relevant Nautobot API endpoints using natural language queries.

**Purpose:** Discover API endpoints when you don't know the exact path or want to find similar endpoints.

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | string | Yes | Natural language description of desired endpoint |
| `n_results` | integer | No | Number of results to return (default: 5) |

#### Example Usage

**Query:**
```
Find endpoints for managing devices
```

**Response:**
```json
{
  "results": [
    {
      "method": "GET",
      "path": "/api/dcim/devices/",
      "description": "List all devices",
      "similarity_score": 0.89
    },
    {
      "method": "POST",
      "path": "/api/dcim/devices/",
      "description": "Create a new device",
      "similarity_score": 0.87
    },
    {
      "method": "GET",
      "path": "/api/dcim/devices/{id}/",
      "description": "Retrieve a specific device",
      "similarity_score": 0.85
    }
  ]
}
```

#### Use Cases

- 🔍 **API Discovery:** "What endpoints exist for IP address management?"
- 📝 **Learning:** "How do I create a new location?"
- 🔄 **Migration:** "Find all endpoints related to interfaces"

---

### `nautobot_dynamic_api_request`

Execute HTTP requests against any Nautobot API endpoint.

**Purpose:** Perform CRUD operations on Nautobot resources with full control over request parameters.

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `method` | string | Yes | HTTP method: GET, POST, PUT, PATCH, DELETE |
| `path` | string | Yes | API endpoint path (e.g., `/api/dcim/devices/`) |
| `params` | object | No | URL query parameters |
| `data` | object | No | Request body (for POST, PUT, PATCH) |

#### Example Usage

**Example 1: GET Request**

```json
{
  "method": "GET",
  "path": "/api/dcim/devices/",
  "params": {
    "status": "active",
    "limit": 10
  }
}
```

**Example 2: POST Request**

```json
{
  "method": "POST",
  "path": "/api/dcim/devices/",
  "data": {
    "name": "switch-01",
    "device_type": "cisco-catalyst-3850",
    "site": "main-dc",
    "status": "active"
  }
}
```

**Example 3: PATCH Request**

```json
{
  "method": "PATCH",
  "path": "/api/dcim/devices/12345/",
  "data": {
    "status": "offline"
  }
}
```

#### Response Format

```json
{
  "status": 200,
  "data": {
    "id": "12345",
    "name": "switch-01",
    "status": "active"
  },
  "count": 1
}
```

#### Use Cases

- 📊 **Data Retrieval:** "Get all devices in Main DC"
- ✏️ **Create Resources:** "Create a new IP address"
- 🔄 **Updates:** "Change device status to maintenance"
- 🗑️ **Deletion:** "Remove outdated location"

---

### `refresh_endpoint_index`

Manually refresh the OpenAPI endpoint index from Nautobot.

**Purpose:** Update the endpoint search index after Nautobot updates or plugin installations.

#### Parameters

None

#### Example Usage

**Command:**
```
Refresh the endpoint index
```

**Response:**
```json
{
  "status": "success",
  "message": "Endpoint index refreshed",
  "endpoints_indexed": 250,
  "time_taken": "2.3s"
}
```

#### When to Use

- ⬆️ After Nautobot upgrades
- 🔌 After installing new plugins
- 🔄 Periodically for consistency

---

## Knowledge Base Tools

### `nautobot_kb_semantic_search`

Search through indexed Nautobot documentation and code using semantic similarity.

**Purpose:** Find relevant documentation, code examples, and best practices from indexed repositories.

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | string | Yes | Search query (natural language) |
| `n_results` | integer | No | Number of results to return (default: 5) |
| `optimize_for_llm` | boolean | No | Optimize content for LLM consumption (default: false) |

#### Example Usage

**Query:**
```json
{
  "query": "How do I create custom fields in Nautobot?",
  "n_results": 3,
  "optimize_for_llm": true
}
```

**Response:**
```json
{
  "results": [
    {
      "repository": "nautobot/nautobot",
      "file_path": "docs/user-guides/custom-fields.md",
      "content": "Custom fields allow you to add additional attributes...",
      "similarity_score": 0.92,
      "metadata": {
        "file_type": "markdown",
        "last_updated": "2024-10-10"
      }
    },
    {
      "repository": "nautobot/nautobot",
      "file_path": "examples/custom_fields_example.py",
      "content": "from nautobot.extras.models import CustomField...",
      "similarity_score": 0.88,
      "metadata": {
        "file_type": "python",
        "last_updated": "2024-09-15"
      }
    }
  ],
  "total_searched": 5000,
  "query_time": "0.15s"
}
```

#### Search Strategies

**Basic Search:**
```
"device configuration"
```

**Specific Topics:**
```
"GraphQL API authentication"
```

**Code Examples:**
```
"pynautobot example for creating devices"
```

**Best Practices:**
```
"best practices for organizing locations"
```

#### Use Cases

- 📚 **Documentation:** "How does the plugin system work?"
- 💻 **Code Examples:** "Show me PyNautobot examples"
- 🎓 **Learning:** "What are device roles?"
- 🔍 **Troubleshooting:** "How to debug Nautobot jobs"

---

## Repository Management Tools

### `nautobot_kb_list_repos`

List all repositories configured in the knowledge base.

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `repo_type` | string | No | Filter by type: "official", "user", "all" (default: "all") |

#### Example Usage

**Query:**
```json
{
  "repo_type": "all"
}
```

**Response:**
```json
{
  "repositories": [
    {
      "name": "nautobot/nautobot",
      "description": "Core Nautobot application",
      "type": "official",
      "enabled": true,
      "documents": 450,
      "last_updated": "2024-10-14"
    },
    {
      "name": "nautobot/nautobot-plugin-golden-config",
      "description": "Golden Config Plugin",
      "type": "official",
      "enabled": true,
      "documents": 85,
      "last_updated": "2024-10-12"
    }
  ],
  "total": 2
}
```

---

### `nautobot_kb_add_repo`

Add a new repository to the knowledge base.

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `repo` | string | Yes | Repository name (owner/repo) |
| `description` | string | No | Repository description |
| `branch` | string | No | Branch to index (default: "main") |
| `enabled` | boolean | No | Enable indexing (default: true) |

#### Example Usage

**Add Repository:**
```json
{
  "repo": "myorg/nautobot-custom-plugin",
  "description": "Custom internal plugin",
  "branch": "main",
  "enabled": true
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Repository added successfully",
  "repo": "myorg/nautobot-custom-plugin",
  "indexing_started": true
}
```

---

### `nautobot_kb_remove_repo`

Remove a repository from the knowledge base.

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `repo` | string | Yes | Repository name to remove |

#### Example Usage

**Remove Repository:**
```json
{
  "repo": "myorg/old-plugin"
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Repository removed",
  "documents_deleted": 45
}
```

---

### `nautobot_kb_update_repos`

Update (pull latest changes) for configured repositories.

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `force` | boolean | No | Force re-index all content (default: false) |

#### Example Usage

**Update All Repositories:**
```json
{
  "force": false
}
```

**Response:**
```json
{
  "status": "success",
  "updated": 5,
  "new_documents": 12,
  "updated_documents": 8,
  "time_taken": "45s"
}
```

---

### `nautobot_kb_init_repos`

Initialize/re-initialize all repositories in the knowledge base.

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `force` | boolean | No | Force re-initialization (default: false) |

#### Example Usage

```json
{
  "force": true
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Repositories initialized",
  "total_repos": 5,
  "total_documents": 1250,
  "time_taken": "3m 45s"
}
```

---

### `nautobot_kb_repo_status`

Get status information about knowledge base repositories.

#### Parameters

None

#### Example Usage

**Get Status:**
```
Check repository status
```

**Response:**
```json
{
  "repositories": [
    {
      "name": "nautobot/nautobot",
      "status": "active",
      "documents": 450,
      "last_indexed": "2024-10-14 20:30:00",
      "indexing": false
    }
  ],
  "total_documents": 1250,
  "indexing_active": false
}
```

---

## System Tools

### Error Handling

All tools return consistent error responses:

```json
{
  "status": "error",
  "error": "Error description",
  "details": "Additional error context"
}
```

### Common Errors

| Error Code | Cause | Solution |
|------------|-------|----------|
| 401 | Invalid API token | Check `NAUTOBOT_TOKEN` in `.env` |
| 403 | Insufficient permissions | Grant token appropriate permissions |
| 404 | Resource not found | Verify endpoint path or resource ID |
| 429 | Rate limit exceeded | Wait or add GitHub token |
| 500 | Server error | Check Nautobot server logs |

---

## Using Tools with Copilot

### Natural Language Queries

Copilot automatically selects appropriate tools based on your query:

**Examples:**

```
"Find device management endpoints"
→ Uses: nautobot_openapi_api_request_schema

"Get all active devices"
→ Uses: nautobot_dynamic_api_request

"How do I create a job?"
→ Uses: nautobot_kb_semantic_search

"Add nautobot-plugin-ssot to the knowledge base"
→ Uses: nautobot_kb_add_repo
```

### Explicit Tool Usage

You can request specific tools:

```
Use nautobot_kb_semantic_search to find information about GraphQL
```

### Chaining Tools

Copilot can chain multiple tools:

```
First find the endpoint for creating devices, 
then show me an example from the knowledge base,
finally create a test device named "test-switch-01"
```

This will use:
1. `nautobot_openapi_api_request_schema`
2. `nautobot_kb_semantic_search`
3. `nautobot_dynamic_api_request`

---

## Best Practices

### API Tools

✅ **Do:**

- Use specific queries for endpoint discovery
- Include necessary parameters in API requests
- Validate data before POST/PATCH operations

❌ **Don't:**

- Make destructive operations without confirmation
- Hardcode resource IDs (query first)
- Skip error handling

### Knowledge Base Tools

✅ **Do:**

- Use natural language in queries
- Enable LLM optimization for longer docs
- Search before implementing features

❌ **Don't:**

- Use overly generic queries
- Ignore source repository information
- Expect real-time Nautobot data

### Repository Management

✅ **Do:**

- Add repos relevant to your work
- Update repos periodically
- Remove unused repos to save space

❌ **Don't:**

- Add very large repos without consideration
- Force re-index frequently
- Add repos without descriptions

---

## Performance Tips

### Search Optimization

- **Specific queries** return better results than generic ones
- **Enable LLM optimization** for better summarization
- **Limit results** to what you need (default: 5)

### Caching

Tools leverage caching:

- **Endpoint index** cached until refresh
- **Embeddings** cached in ChromaDB
- **Repository content** cached locally

### Resource Usage

Monitor resource usage:

```bash
# Check backend size
du -sh backend/

# Check repository cache
du -sh ~/.cache/nautobot_mcp/
```

---

## Next Steps

- [Examples](examples.md) - Real-world usage examples
- [Architecture](architecture.md) - How tools work internally
- [Troubleshooting](troubleshooting.md) - Solve common issues
