# Knowledge Base Management

Guide to managing the Nautobot MCP Server knowledge base.

## Overview

The knowledge base indexes documentation and code from GitHub repositories, enabling semantic search over Nautobot-related content.

## Repository Configuration

### Official Repositories

Managed in `config/repositories.json`:

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
    }
  ]
}
```

### User Repositories

Add custom repositories via MCP tools or `config/user_repositories.json`.

## Managing Repositories

### List Repositories

```
List all repositories in the knowledge base
```

### Add Repository

```
Add repository "myorg/nautobot-plugin" to the knowledge base
```

### Remove Repository

```
Remove repository "myorg/old-plugin" from the knowledge base
```

### Update Repositories

```
Update all repositories in the knowledge base
```

## Search Strategies

### Basic Search

```
Search for "custom fields"
```

Returns relevant documentation and code.

### Optimized Search

For better LLM consumption:

```
Search for "custom fields" with LLM optimization
```

## Indexed Content

- **Python Code** (`.py`)
- **Markdown Docs** (`.md`)
- **reStructuredText** (`.rst`)
- **Text Files** (`.txt`)
- **Configuration** (`.yaml`, `.yml`, `.json`)

## Best Practices

1. **Add Relevant Repos** - Only add repositories related to your work
2. **Update Regularly** - Keep content fresh with periodic updates
3. **Monitor Size** - Large repos take time and space
4. **Use Descriptions** - Help identify repositories later

## Troubleshooting

See the [Troubleshooting Guide](troubleshooting.md#knowledge-base-issues) for common issues.
