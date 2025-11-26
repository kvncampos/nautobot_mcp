# Development Guide

Guide for developers who want to contribute to or extend the Nautobot MCP Server.

## Development Setup

### Prerequisites

- Python 3.11
- Git
- uv (recommended) or pip

### Clone and Install

```bash
# Clone repository
git clone https://github.com/kvncampos/nautobot_mcp.git
cd nautobot_mcp

# Install with dev dependencies
uv sync --group dev

# Or with pip
pip install -e ".[dev]"
```

### Install Pre-commit Hooks

```bash
pre-commit install
```

## Project Structure

```
nautobot_mcp/
├── server.py                 # Main MCP server
├── helpers/                  # Core modules
│   ├── nb_kb_v2.py          # Knowledge base
│   ├── endpoint_searcher_chroma.py
│   └── content_processor.py
├── utils/                    # Utilities
│   ├── config.py
│   ├── embedding.py
│   └── git_manager.py
├── tests/                    # Test suite
├── examples/                 # Usage examples
└── docs/                     # Documentation
```

## Running Tests

```bash
# Run all tests
pytest

# Run specific categories
pytest -m "unit"
pytest -m "integration"
pytest -m "offline"

# Run with coverage
pytest --cov=helpers --cov=utils

# Verbose output
pytest -v -s
```

## Code Quality

### Formatting

```bash
# Format all code
ruff format .

# Check formatting
ruff format --check .
```

### Linting

```bash
# Lint code
ruff check .

# Fix auto-fixable issues
ruff check --fix .
```

### Type Checking

The project uses type hints. Consider adding mypy:

```bash
pip install mypy
mypy helpers/ utils/
```

## Adding New Features

### Adding a New MCP Tool

1. **Define the tool schema:**

```python
# In server.py
tool_schema = {
    "name": "my_new_tool",
    "description": "What the tool does",
    "inputSchema": {
        "type": "object",
        "properties": {
            "param1": {
                "type": "string",
                "description": "Parameter description"
            }
        },
        "required": ["param1"]
    }
}
```

2. **Implement the handler:**

```python
@server.call_tool()
async def handle_invoke_tool(name: str, inputs: Dict[str, Any]):
    if name == "my_new_tool":
        param1 = inputs["param1"]
        # Implementation
        result = do_something(param1)
        return [types.TextContent(type="text", text=json.dumps(result))]
```

3. **Add tests:**

```python
# In tests/test_my_tool.py
def test_my_new_tool():
    # Test implementation
    pass
```

4. **Update documentation:**

Add to `docs/tools.md`.

### Adding a New Helper Module

1. Create file in `helpers/`
2. Implement functionality
3. Add tests in `tests/`
4. Update documentation

## Testing Guidelines

### Unit Tests

Test individual functions in isolation:

```python
def test_config_loading():
    config = Config()
    assert config.SERVER_NAME == "nautobot_mcp"
```

### Integration Tests

Test component interactions:

```python
@pytest.mark.integration
def test_endpoint_search():
    searcher = EndpointSearcherChroma()
    results = searcher.search("device")
    assert len(results) > 0
```

### Offline Tests

Tests that don't require network:

```python
@pytest.mark.offline
def test_embedding_function():
    # Test with cached data
    pass
```

## Documentation

### Building Docs

```bash
# Install docs dependencies
uv sync --group docs

# Build documentation
mkdocs build

# Serve locally
mkdocs serve
```

View at http://localhost:8000

### Writing Documentation

- Use clear, concise language
- Include code examples
- Add screenshots where helpful
- Cross-reference related pages

## Contributing

See [Contributing Guide](contributing.md) for:

- Code style guidelines
- Pull request process
- Issue reporting
- Feature requests

## Debugging

### Debug Mode

```bash
LOG_LEVEL=DEBUG python server.py
```

### VS Code Debugging

Create `.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: MCP Server",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/server.py",
      "console": "integratedTerminal",
      "env": {
        "LOG_LEVEL": "DEBUG"
      }
    }
  ]
}
```

## Release Process

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create git tag
4. Push to GitHub
5. GitHub Actions builds and publishes

---

## Resources

- [MCP Specification](https://spec.modelcontextprotocol.io/)
- [ChromaDB Docs](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)
