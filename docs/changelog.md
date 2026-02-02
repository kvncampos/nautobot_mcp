# Changelog

All notable changes to the Nautobot MCP Server will be documented in this file.

## [Unreleased]

### Added
- MKDocs documentation with GitHub Pages support
- Comprehensive user guides and examples
- Architecture documentation with diagrams
- Common tool handlers module (`helpers/tool_handlers.py`) for shared implementation logic
- Common tool definitions module (`helpers/tool_definitions.py`) for consistent tool schemas
- Comprehensive unit tests for tool definitions (`tests/test_tool_definitions.py`)

### Changed
- **[Breaking]** Improved tool descriptions following LLM ingestion best practices
  - Descriptions are now more concise (under 300 characters)
  - Added clear, actionable language for better LLM comprehension
  - Documented parameter usage patterns (GET/DELETE use `params`, POST/PUT/PATCH use `body`)
- **[Breaking]** Removed unused parameters:
  - Removed `repo_type` parameter from `nautobot_kb_list_repos` (was accepted but not used)
  - Removed `category` parameter from `nautobot_kb_add_repo` (was accepted but not used)
- Refactored both `server.py` and `server_http.py` to eliminate code duplication
  - Reduced `server.py` from 631 to 249 lines (60% reduction)
  - Reduced `server_http.py` from 405 to 151 lines (62% reduction)
  - Extracted ~400 lines of duplicate code to shared modules
- Improved maintainability by establishing single source of truth for tool implementations

### Fixed
- Inconsistent tool descriptions between stdio and HTTP implementations
- Missing parameter constraints and validations in tool schemas
- Unclear documentation about which HTTP methods use which parameters

## [0.2.0] - 2024-10-14

### Added
- Enhanced knowledge base with optimized search
- Repository management tools
- Support for multiple Nautobot environments
- ChromaDB-based vector storage

### Changed
- Improved embedding model configuration
- Better error handling and logging

## [0.1.0] - 2024-10-01

### Added
- Initial MCP server implementation
- OpenAPI endpoint discovery
- Basic knowledge base search
- VS Code Copilot integration

---

## Contributing

See the [Contributing Guide](contributing.md) for information on how to contribute to this project.
