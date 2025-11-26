# Quick Start Guide

Get up and running with the Nautobot MCP Server in minutes. This guide assumes you've completed the [installation](installation.md).

## Starting the Server

### Method 1: Direct Execution

Simply run the server script:

```bash
cd nautobot_mcp
python server.py
```

The server will:

1. ✅ Initialize ChromaDB collections
2. ✅ Fetch OpenAPI schema from Nautobot
3. ✅ Update knowledge base repositories
4. ✅ Start serving MCP requests

Expected output:

```
2024-10-14 20:30:01 [INFO] Initializing Nautobot MCP Server...
2024-10-14 20:30:02 [INFO] ChromaDB initialized at: /path/to/backend/nautobot_mcp
2024-10-14 20:30:03 [INFO] Refreshing API endpoints from OpenAPI schema...
2024-10-14 20:30:04 [INFO] Indexed 250 API endpoints
2024-10-14 20:30:05 [INFO] Updating knowledge base repositories...
2024-10-14 20:30:10 [INFO] Knowledge base ready with 5 repositories
2024-10-14 20:30:10 [INFO] Server ready on stdio
```

### Method 2: With uv

Using `uv` to manage the environment:

```bash
uv run python server.py
```

## VS Code Integration

Integrate the Nautobot MCP Server with GitHub Copilot in VS Code for an AI-powered Nautobot experience.

### Prerequisites

- VS Code with GitHub Copilot extension installed
- Active GitHub Copilot subscription
- Nautobot MCP Server installed locally

### Step 1: Open MCP Configuration

1. Open VS Code
2. Press `Cmd+Shift+P` (macOS) or `Ctrl+Shift+P` (Windows/Linux)
3. Type and select: **`MCP: Open User Configuration`**

This opens the MCP settings file (usually `~/.config/Code/User/globalStorage/github.copilot-chat/mcp.json` on macOS/Linux).

### Step 2: Add Server Configuration

Add the Nautobot MCP Server to your configuration:

=== "Using uv (Recommended)"
    ```json
    {
      "servers": {
        "nautobot_mcp": {
          "type": "stdio",
          "command": "uv",
          "args": [
            "run",
            "--directory",
            "/absolute/path/to/nautobot_mcp",
            "python3",
            "-m",
            "server"
          ],
          "env": {
            "PYTHONPATH": "/absolute/path/to/nautobot_mcp"
          }
        }
      },
      "inputs": []
    }
    ```

=== "Using Python Directly"
    ```json
    {
      "servers": {
        "nautobot_mcp": {
          "type": "stdio",
          "command": "/absolute/path/to/venv/bin/python",
          "args": [
            "/absolute/path/to/nautobot_mcp/server.py"
          ],
          "env": {
            "PYTHONPATH": "/absolute/path/to/nautobot_mcp"
          }
        }
      },
      "inputs": []
    }
    ```

!!! important "Use Absolute Paths"
    Replace `/absolute/path/to/nautobot_mcp` with the full path to your installation.
    
    Example: `/Users/johndoe/projects/nautobot_mcp` or `/home/johndoe/nautobot_mcp`

### Step 3: Restart VS Code

1. Close all VS Code windows
2. Reopen VS Code
3. Open the Copilot Chat panel

### Step 4: Verify Connection

In the Copilot Chat panel, you should see:

- ✅ A small indicator showing MCP tools are available
- ✅ The Nautobot MCP Server in the list of active servers

Test with a simple query:

```
@workspace What Nautobot API endpoints are available for managing devices?
```

## Your First Queries

Now that the server is running and connected to VS Code Copilot, try these examples:

### Example 1: Discover API Endpoints

Ask Copilot to find relevant endpoints:

```
Find API endpoints related to device management in Nautobot
```

Copilot will use the `nautobot_openapi_api_request_schema` tool to search for relevant endpoints.

**Expected Response:**
```
I found several endpoints for device management:

1. GET /api/dcim/devices/ - List all devices
2. POST /api/dcim/devices/ - Create a new device
3. GET /api/dcim/devices/{id}/ - Retrieve a specific device
4. PATCH /api/dcim/devices/{id}/ - Update a device
5. DELETE /api/dcim/devices/{id}/ - Delete a device
```

### Example 2: Get API Data

Request data from your Nautobot instance:

```
Get all locations from my Nautobot instance
```

Copilot will use the `nautobot_dynamic_api_request` tool to fetch the data.

**Expected Response:**
```
Here are the locations in your Nautobot instance:

1. Main DC (Status: Active)
2. Branch Office 1 (Status: Active)
3. Branch Office 2 (Status: Planned)
...
```

### Example 3: Search Knowledge Base

Ask questions about Nautobot features:

```
How do I create custom fields in Nautobot?
```

Copilot will use the `nautobot_kb_semantic_search` tool to find relevant documentation.

**Expected Response:**
```
Based on the Nautobot documentation, here's how to create custom fields:

1. Navigate to Extensibility > Custom Fields
2. Click "Add Custom Field"
3. Configure the following:
   - Content Type: Select the object type
   - Name: Field name (no spaces)
   - Label: Display name
   - Type: Choose data type (text, integer, boolean, etc.)
...
```

### Example 4: Complex Queries

Combine multiple tools for complex queries:

```
Show me how to create a new device in Nautobot, 
then actually create one with name "test-switch-01" 
in the "Main DC" location
```

Copilot will:

1. Search the knowledge base for device creation guidance
2. Find the appropriate API endpoint
3. Execute the API call with the provided parameters
4. Return the result

## Common Workflows

### Workflow 1: Exploring the API

1. **Discover endpoints:**
   ```
   What endpoints are available for IP address management?
   ```

2. **Get endpoint details:**
   ```
   Show me the parameters for creating an IP address
   ```

3. **Execute the request:**
   ```
   Create an IP address 192.168.1.10/24 with description "Server IP"
   ```

### Workflow 2: Learning Nautobot

1. **Ask conceptual questions:**
   ```
   What are device roles in Nautobot?
   ```

2. **Find code examples:**
   ```
   Show me examples of using PyNautobot to query devices
   ```

3. **Get best practices:**
   ```
   What are the best practices for organizing locations in Nautobot?
   ```

### Workflow 3: Automation Development

1. **Research capabilities:**
   ```
   How can I automate device onboarding in Nautobot?
   ```

2. **Get code templates:**
   ```
   Show me a Python script template for adding devices via API
   ```

3. **Test API calls:**
   ```
   Test the device creation endpoint with sample data
   ```

## Tips for Better Results

### Be Specific

❌ **Too vague:**
```
Tell me about devices
```

✅ **Better:**
```
Show me the API endpoint for listing devices with filters by status and location
```

### Use Context

Provide relevant details in your queries:

```
I need to create a device of type "switch" with role "access-switch" 
in the "Main DC" location. What's the API call?
```

### Chain Operations

Break complex tasks into steps:

```
1. First, find the location ID for "Main DC"
2. Then, create a device in that location
```

### Reference Documentation

Ask for specific documentation:

```
Show me the documentation for the GraphQL API in Nautobot
```

## Keyboard Shortcuts

### VS Code Copilot Chat

- `Cmd+I` / `Ctrl+I` - Open inline chat
- `Cmd+Shift+I` / `Ctrl+Shift+I` - Open chat panel
- `Cmd+Shift+P` → "Copilot: Toggle Chat" - Toggle chat visibility

## Troubleshooting

### Server Not Connected

If Copilot can't find the MCP server:

1. Check VS Code's Output panel (`View → Output`)
2. Select "GitHub Copilot Chat" from the dropdown
3. Look for connection errors

Common issues:

- **Incorrect path**: Verify the path in `mcp.json` is absolute
- **Server not running**: The server starts automatically with VS Code
- **Permission issues**: Ensure the server script is executable

### No Tools Available

If queries don't use MCP tools:

1. Verify the server is in the active servers list
2. Try mentioning the tool explicitly:
   ```
   Use the nautobot_kb_semantic_search tool to find information about...
   ```

### Slow Responses

On first use, responses may be slower due to:

- Initial model loading (~90MB download)
- Repository cloning and indexing
- Vector embedding generation

Subsequent queries will be faster.

### Authentication Errors

If you see "401 Unauthorized":

1. Verify `NAUTOBOT_TOKEN` in `.env`
2. Check token permissions in Nautobot
3. Ensure `NAUTOBOT_ENV` matches your configuration

## Next Steps

- [Configuration Guide](configuration.md) - Customize your setup
- [Tools Reference](tools.md) - Learn about available MCP tools
- [Examples](examples.md) - More detailed use cases
- [Architecture](architecture.md) - Understand how it works

## Getting Help

If you encounter issues:

- Check the [Troubleshooting Guide](troubleshooting.md)
- Review server logs for errors
- Open an issue on [GitHub](https://github.com/kvncampos/nautobot_mcp/issues)
