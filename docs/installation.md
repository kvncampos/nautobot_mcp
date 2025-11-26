# Installation Guide

This guide will walk you through installing the Nautobot MCP Server on your system.

## Prerequisites

Before installing, ensure you have:

- **Python 3.11** (< 3.12) - Required for compatibility with all dependencies
- **Git** - For cloning the repository and managing knowledge base repositories
- **Access to a Nautobot instance** - Either local or remote
- **GitHub Personal Access Token** (optional but recommended) - For accessing private repositories

### System Requirements

- **OS**: Linux, macOS, or Windows (WSL2 recommended for Windows)
- **RAM**: Minimum 4GB (8GB+ recommended for better performance)
- **Disk Space**: 5GB+ for models and cached repositories
- **Network**: Internet access for downloading dependencies and cloning repositories

## Installation Methods

### Method 1: Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver.

#### Step 1: Install uv

=== "macOS/Linux"
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

=== "Windows"
    ```powershell
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

#### Step 2: Clone the Repository

```bash
git clone https://github.com/kvncampos/nautobot_mcp.git
cd nautobot_mcp
```

#### Step 3: Install Dependencies

```bash
uv sync
```

This command will:

- Create a virtual environment
- Install all dependencies from `pyproject.toml`
- Set up the project for development

### Method 2: Using pip

If you prefer traditional pip installation:

#### Step 1: Clone the Repository

```bash
git clone https://github.com/kvncampos/nautobot_mcp.git
cd nautobot_mcp
```

#### Step 2: Create Virtual Environment

```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### Step 3: Install Dependencies

```bash
pip install -e .
```

For development with additional tools:

```bash
pip install -e ".[dev]"
```

## Configuration

### Step 1: Create Environment File

Copy the example environment file:

```bash
cp .env.example .env
```

### Step 2: Configure Environment Variables

Edit `.env` with your settings:

```ini
# Nautobot Configuration
NAUTOBOT_TOKEN=your_nautobot_api_token_here
NAUTOBOT_ENV=local  # Options: local, nonprod, prod

# Environment-specific URLs
NAUTOBOT_URL_LOCAL=http://localhost:8000
NAUTOBOT_URL_NONPROD=https://nautobot-nonprod.example.com
NAUTOBOT_URL_PROD=https://nautobot.example.com

# GitHub Configuration (optional but recommended)
GITHUB_TOKEN=your_github_personal_access_token

# Server Configuration
API_PREFIX=nautobot_openapi
SERVER_NAME=nautobot_mcp
SERVER_VERSION=0.2.0
LOG_LEVEL=INFO

# Database Configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2
DEFAULT_SEARCH_RESULTS=5

# Advanced Configuration
API_TIMEOUT=10
SSL_VERIFY=True
POSTHOG_API_KEY=disable
```

### Step 3: Obtain Required Tokens

#### Nautobot API Token

1. Log in to your Nautobot instance
2. Navigate to your profile → API Tokens
3. Click "Add Token"
4. Set appropriate permissions (read access at minimum)
5. Copy the generated token to `NAUTOBOT_TOKEN` in `.env`

#### GitHub Personal Access Token (Optional)

For accessing private repositories or to avoid rate limits:

1. Go to GitHub Settings → Developer settings → Personal access tokens
2. Click "Generate new token (classic)"
3. Select scopes:
    - `repo` - For private repositories
    - `public_repo` - For public repositories only
4. Copy the token to `GITHUB_TOKEN` in `.env`

!!! warning "Token Security"
    Never commit your `.env` file to version control. The `.gitignore` file is configured to exclude it by default.

## Verify Installation

### Test Basic Functionality

Run the server:

```bash
python server.py
```

You should see output similar to:

```
2024-10-14 20:30:01 [INFO] nautobot_mcp: Initializing Nautobot MCP Server...
2024-10-14 20:30:02 [INFO] nautobot_mcp: ChromaDB collections initialized
2024-10-14 20:30:03 [INFO] nautobot_mcp: API endpoint index refreshed
2024-10-14 20:30:04 [INFO] nautobot_mcp: Server ready to accept requests
```

### Run Tests

Verify the installation by running tests:

```bash
# Using uv
uv run python -m pytest tests/

# Using pip
pytest tests/
```

### Check Embedding Model

The first run will download the sentence-transformers model (~90MB):

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

## Post-Installation Steps

### Initialize Knowledge Base

On first run, the server will automatically:

1. Create ChromaDB collections
2. Fetch and index the OpenAPI schema from your Nautobot instance
3. Clone configured repositories
4. Index documentation and code files

This process may take 5-10 minutes depending on:

- Number of configured repositories
- Size of repositories
- Network speed

### Configure VS Code Integration

See the [Quick Start Guide](quickstart.md#vs-code-integration) for VS Code Copilot integration.

## Troubleshooting Installation

### Python Version Issues

Ensure you're using Python 3.11:

```bash
python --version  # Should show Python 3.11.x
```

If you have multiple Python versions:

```bash
python3.11 -m venv venv
source venv/bin/activate
```

### SSL Certificate Errors

If you encounter SSL errors with self-signed certificates:

```ini
# In .env
SSL_VERIFY=False
```

!!! warning "Production Warning"
    Only disable SSL verification in development environments.

### ChromaDB Permission Issues

If you see permission errors:

```bash
chmod -R 755 backend/
```

### GitHub Rate Limits

Without a GitHub token, you may hit rate limits:

```
403 Client Error: rate limit exceeded
```

Solution: Add a GitHub token to your `.env` file.

### Port Conflicts

If port 8000 is in use, modify your Nautobot URL or check for other services:

```bash
# Check what's using port 8000
lsof -i :8000  # macOS/Linux
netstat -ano | findstr :8000  # Windows
```

## Updating

To update to the latest version:

```bash
cd nautobot_mcp
git pull origin main
uv sync  # or: pip install -e .
```

## Uninstalling

To completely remove the installation:

```bash
# Remove virtual environment
rm -rf venv/

# Remove cached data
rm -rf backend/

# Remove cloned repositories
rm -rf config/user_repositories.json

# Remove the project directory
cd ..
rm -rf nautobot_mcp/
```

## Next Steps

- [Quick Start Guide](quickstart.md) - Get started with your first query
- [Configuration Guide](configuration.md) - Advanced configuration options
- [Architecture Overview](architecture.md) - Understand how it works
