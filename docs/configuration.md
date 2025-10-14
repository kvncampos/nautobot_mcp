# Configuration Reference

This guide provides detailed information about all configuration options available in the Nautobot MCP Server.

## Configuration File

Configuration is managed through environment variables in the `.env` file located at the project root.

### Creating Your Configuration

```bash
# Copy the example file
cp .env.example .env

# Edit with your preferred editor
nano .env  # or vim, code, etc.
```

## Core Configuration

### Nautobot Connection

#### `NAUTOBOT_TOKEN`

**Required:** Yes  
**Type:** String  
**Description:** API token for authenticating with Nautobot.

```ini
NAUTOBOT_TOKEN=your_nautobot_api_token_here
```

!!! info "Getting a Token"
    1. Log into Nautobot
    2. Navigate to Profile → API Tokens
    3. Create a new token with appropriate permissions
    4. Copy the token value

#### `NAUTOBOT_ENV`

**Required:** No  
**Type:** String  
**Default:** `local`  
**Options:** `local`, `nonprod`, `prod`  
**Description:** Select which Nautobot environment to connect to.

```ini
NAUTOBOT_ENV=local
```

#### Environment URLs

Configure URLs for each environment:

```ini
NAUTOBOT_URL_LOCAL=http://localhost:8000
NAUTOBOT_URL_NONPROD=https://nautobot-nonprod.example.com
NAUTOBOT_URL_PROD=https://nautobot.example.com
```

The active URL is determined by `NAUTOBOT_ENV`.

### GitHub Integration

#### `GITHUB_TOKEN`

**Required:** No (recommended for best experience)  
**Type:** String  
**Description:** GitHub Personal Access Token for repository access.

```ini
GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

**Required Scopes:**

- `repo` - For private repository access
- `public_repo` - For public repository access (if not using `repo`)

**Why Use a Token?**

- Access private repositories
- Avoid GitHub API rate limits (60/hour → 5000/hour)
- Clone repositories faster

!!! warning "Rate Limits"
    Without a token, you may experience:
    ```
    403 API rate limit exceeded for [ip address]
    ```

## Server Configuration

### `API_PREFIX`

**Required:** No  
**Type:** String  
**Default:** `nautobot_openapi`  
**Description:** Prefix for MCP tool names.

```ini
API_PREFIX=nautobot_openapi
```

Tools will be named: `{API_PREFIX}_dynamic_api_request`, etc.

### `SERVER_NAME`

**Required:** No  
**Type:** String  
**Default:** `nautobot_mcp`  
**Description:** Name of the MCP server.

```ini
SERVER_NAME=nautobot_mcp
```

### `SERVER_VERSION`

**Required:** No  
**Type:** String  
**Default:** `0.2.0`  
**Description:** Server version reported to MCP clients.

```ini
SERVER_VERSION=0.2.0
```

## Logging Configuration

### `LOG_LEVEL`

**Required:** No  
**Type:** String  
**Default:** `INFO`  
**Options:** `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`  
**Description:** Controls the verbosity of log output.

```ini
LOG_LEVEL=INFO
```

**Log Level Details:**

| Level | Use Case | Output Volume |
|-------|----------|---------------|
| `DEBUG` | Development, troubleshooting | Very high |
| `INFO` | Normal operation | Moderate |
| `WARNING` | Production monitoring | Low |
| `ERROR` | Critical issues only | Very low |

**Example Output:**

```bash
# INFO level
2024-10-14 20:30:01 [INFO] Server ready

# DEBUG level
2024-10-14 20:30:01 [DEBUG] Loading model: all-MiniLM-L6-v2
2024-10-14 20:30:01 [DEBUG] Initializing ChromaDB at: /path/to/backend
2024-10-14 20:30:02 [DEBUG] Fetching schema from: http://localhost:8000/api/docs
```

## Embedding Configuration

### `EMBEDDING_MODEL`

**Required:** No  
**Type:** String  
**Default:** `all-MiniLM-L6-v2`  
**Description:** Sentence transformer model for generating embeddings.

```ini
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

**Available Models:**

| Model | Size | Speed | Quality | Dimensions |
|-------|------|-------|---------|------------|
| `all-MiniLM-L6-v2` | ~90MB | Fast | Good | 384 |
| `all-mpnet-base-v2` | ~420MB | Moderate | Better | 768 |
| `multi-qa-mpnet-base-dot-v1` | ~420MB | Moderate | Best | 768 |

!!! tip "Model Selection"
    - Use `all-MiniLM-L6-v2` for best performance
    - Use `all-mpnet-base-v2` for better search quality
    - First run downloads the model (~5-10 minutes)

### `DEFAULT_SEARCH_RESULTS`

**Required:** No  
**Type:** Integer  
**Default:** `5`  
**Range:** 1-100  
**Description:** Default number of results to return from searches.

```ini
DEFAULT_SEARCH_RESULTS=5
```

## Network Configuration

### `API_TIMEOUT`

**Required:** No  
**Type:** Integer  
**Default:** `10`  
**Unit:** Seconds  
**Description:** Timeout for API requests to Nautobot.

```ini
API_TIMEOUT=10
```

Increase for slow networks or large responses:

```ini
API_TIMEOUT=30  # For slow connections
```

### `SSL_VERIFY`

**Required:** No  
**Type:** Boolean  
**Default:** `True`  
**Options:** `True`, `False`  
**Description:** Enable/disable SSL certificate verification.

```ini
SSL_VERIFY=True
```

!!! danger "Production Warning"
    Only set to `False` in development environments with self-signed certificates.
    
    ```ini
    # Development only!
    SSL_VERIFY=False
    ```

## Analytics Configuration

### `POSTHOG_API_KEY`

**Required:** No  
**Type:** String  
**Default:** `disable`  
**Description:** PostHog analytics API key for usage tracking.

```ini
POSTHOG_API_KEY=disable
```

To enable analytics:

```ini
POSTHOG_API_KEY=phc_your_key_here
```

!!! info "Privacy"
    When disabled, no telemetry data is collected.

## Repository Configuration

Repository configuration is managed through JSON files in the `config/` directory.

### Official Repositories

**File:** `config/repositories.json`  
**Purpose:** Pre-configured Nautobot-related repositories  
**Managed:** By project maintainers

Example:

```json
{
  "repositories": [
    {
      "name": "nautobot/nautobot",
      "description": "Core Nautobot application",
      "priority": 1,
      "enabled": true,
      "branch": "develop",
      "file_patterns": [".py", ".md", ".rst", ".txt", ".yaml", ".yml"]
    },
    {
      "name": "nautobot/nautobot-plugin-golden-config",
      "description": "Golden Config Plugin",
      "priority": 2,
      "enabled": true,
      "branch": "main",
      "file_patterns": [".py", ".md"]
    }
  ]
}
```

### User Repositories

**File:** `config/user_repositories.json`  
**Purpose:** User-added custom repositories  
**Managed:** Via MCP tools or manual editing

Example:

```json
{
  "repositories": [
    {
      "name": "myorg/custom-nautobot-plugin",
      "description": "Internal Nautobot plugin",
      "priority": 3,
      "enabled": true,
      "branch": "main",
      "file_patterns": [".py", ".md"]
    }
  ]
}
```

### Repository Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | String | Yes | GitHub repository (owner/repo) |
| `description` | String | Yes | Human-readable description |
| `priority` | Integer | No | Search priority (lower = higher priority) |
| `enabled` | Boolean | No | Enable/disable indexing |
| `branch` | String | No | Branch to clone (default: main) |
| `file_patterns` | Array | No | File extensions to index |

## Environment-Specific Configuration

### Development Environment

```ini
# Development .env
NAUTOBOT_ENV=local
NAUTOBOT_URL_LOCAL=http://localhost:8000
NAUTOBOT_TOKEN=dev_token_here
GITHUB_TOKEN=your_github_token

LOG_LEVEL=DEBUG
SSL_VERIFY=False
API_TIMEOUT=30

EMBEDDING_MODEL=all-MiniLM-L6-v2
DEFAULT_SEARCH_RESULTS=10
```

### Production Environment

```ini
# Production .env
NAUTOBOT_ENV=prod
NAUTOBOT_URL_PROD=https://nautobot.example.com
NAUTOBOT_TOKEN=prod_token_here
GITHUB_TOKEN=your_github_token

LOG_LEVEL=WARNING
SSL_VERIFY=True
API_TIMEOUT=10

EMBEDDING_MODEL=all-MiniLM-L6-v2
DEFAULT_SEARCH_RESULTS=5
POSTHOG_API_KEY=phc_production_key
```

## Configuration Validation

### Check Current Configuration

Run the server with debug logging to see loaded configuration:

```bash
LOG_LEVEL=DEBUG python server.py
```

Output includes:

```
[DEBUG] Configuration loaded:
[DEBUG]   NAUTOBOT_ENV: local
[DEBUG]   NAUTOBOT_URL: http://localhost:8000
[DEBUG]   LOG_LEVEL: DEBUG
[DEBUG]   EMBEDDING_MODEL: all-MiniLM-L6-v2
```

### Test Configuration

Test API connectivity:

```python
from utils.config import config
import requests

# Test Nautobot connection
response = requests.get(
    f"{config.NAUTOBOT_URL}/api/",
    headers={"Authorization": f"Token {config.NAUTOBOT_TOKEN}"},
    verify=config.SSL_VERIFY
)
print(f"Status: {response.status_code}")
```

## Troubleshooting Configuration

### Common Issues

#### Invalid Token

**Symptom:**
```
401 Unauthorized
```

**Solution:**
1. Verify token in Nautobot UI
2. Check token hasn't expired
3. Ensure token has correct permissions

#### Wrong Environment URL

**Symptom:**
```
Connection refused
```

**Solution:**
1. Verify `NAUTOBOT_ENV` matches intended environment
2. Check URL for that environment
3. Test URL accessibility: `curl -I <url>`

#### GitHub Rate Limit

**Symptom:**
```
403 API rate limit exceeded
```

**Solution:**
Add GitHub token to `.env`:
```ini
GITHUB_TOKEN=ghp_xxxxxxxxxxxx
```

#### SSL Certificate Errors

**Symptom:**
```
SSL: CERTIFICATE_VERIFY_FAILED
```

**Solution:**
For development:
```ini
SSL_VERIFY=False
```

For production, install proper certificates.

### Configuration Precedence

Configuration is loaded in this order:

1. Environment variables (highest priority)
2. `.env` file
3. Default values (lowest priority)

Example:

```bash
# Override .env for one run
NAUTOBOT_ENV=prod python server.py
```

## Best Practices

### Security

1. **Never commit `.env`** - Already in `.gitignore`
2. **Use separate tokens** per environment
3. **Rotate tokens regularly** - Especially production
4. **Limit token permissions** - Read-only when possible

### Performance

1. **Use `all-MiniLM-L6-v2`** for faster embeddings
2. **Adjust `DEFAULT_SEARCH_RESULTS`** based on needs
3. **Set appropriate `API_TIMEOUT`** for network
4. **Enable only needed repositories**

### Maintenance

1. **Document custom settings** in comments
2. **Keep `.env.example` updated**
3. **Use environment-specific configs**
4. **Monitor log output** for issues

## Example Configurations

### Minimal Configuration

Bare minimum to run:

```ini
NAUTOBOT_TOKEN=your_token
NAUTOBOT_URL_LOCAL=http://localhost:8000
```

### Recommended Configuration

Balanced for most users:

```ini
# Core
NAUTOBOT_TOKEN=your_token
NAUTOBOT_ENV=local
NAUTOBOT_URL_LOCAL=http://localhost:8000

# GitHub (recommended)
GITHUB_TOKEN=your_github_token

# Logging
LOG_LEVEL=INFO

# Performance
EMBEDDING_MODEL=all-MiniLM-L6-v2
DEFAULT_SEARCH_RESULTS=5
API_TIMEOUT=10
```

### Advanced Configuration

For power users:

```ini
# Core
NAUTOBOT_TOKEN=your_token
NAUTOBOT_ENV=nonprod
NAUTOBOT_URL_LOCAL=http://localhost:8000
NAUTOBOT_URL_NONPROD=https://nautobot-nonprod.example.com
NAUTOBOT_URL_PROD=https://nautobot.example.com

# GitHub
GITHUB_TOKEN=your_github_token

# Server
API_PREFIX=nautobot
SERVER_NAME=nautobot_mcp_prod
SERVER_VERSION=0.2.0

# Logging
LOG_LEVEL=DEBUG

# Embedding
EMBEDDING_MODEL=all-mpnet-base-v2
DEFAULT_SEARCH_RESULTS=10

# Network
API_TIMEOUT=30
SSL_VERIFY=True

# Analytics
POSTHOG_API_KEY=phc_your_key
```

## Next Steps

- [Quick Start Guide](quickstart.md) - Get started
- [Architecture](architecture.md) - Understand the design
- [Troubleshooting](troubleshooting.md) - Solve issues
