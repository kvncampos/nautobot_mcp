# Contributing to Nautobot MCP Server

Thank you for considering contributing to the Nautobot MCP Server! This document provides guidelines for contributing.

## Ways to Contribute

- 🐛 Report bugs
- 💡 Suggest features
- 📖 Improve documentation
- 🔧 Submit code changes
- 🧪 Add tests
- 💬 Help others in discussions

## Getting Started

1. Fork the repository
2. Clone your fork
3. Create a feature branch
4. Make your changes
5. Submit a pull request

## Development Setup

See the [Development Guide](development.md) for detailed setup instructions.

## Code Guidelines

### Style

- Follow PEP 8
- Use type hints
- Write docstrings for public functions
- Keep functions focused and small

### Example

```python
def search_endpoints(query: str, n_results: int = 5) -> List[Dict[str, Any]]:
    """Search for API endpoints using semantic similarity.
    
    Args:
        query: Natural language search query
        n_results: Number of results to return
        
    Returns:
        List of endpoint dictionaries with metadata
    """
    # Implementation
    pass
```

## Testing

- Write tests for new features
- Ensure all tests pass
- Maintain or improve coverage

```bash
# Run tests
pytest

# With coverage
pytest --cov=helpers --cov=utils
```

## Documentation

- Update docs for new features
- Add examples
- Keep README current

## Pull Request Process

1. **Create an Issue** - Discuss major changes first
2. **Branch Naming** - Use descriptive names: `feature/add-x`, `fix/issue-123`
3. **Commit Messages** - Clear, descriptive messages
4. **Tests** - Add tests for new code
5. **Documentation** - Update relevant docs
6. **Review** - Address feedback

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update

## Testing
How the changes were tested

## Checklist
- [ ] Tests pass
- [ ] Documentation updated
- [ ] Code follows style guidelines
```

## Reporting Issues

### Bug Reports

Include:

- Description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Environment (OS, Python version)
- Logs (with DEBUG level)

### Feature Requests

Include:

- Use case description
- Proposed solution
- Alternative approaches
- Additional context

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Assume good intentions

## Questions?

- Open a [Discussion](https://github.com/kvncampos/nautobot_mcp/discussions)
- Email: kvncampos@duck.com

Thank you for contributing! 🎉
