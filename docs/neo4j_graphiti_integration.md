# Neo4j + Graphiti Integration Guide

## Overview

The Nautobot MCP server now includes **temporal knowledge graph** capabilities using Neo4j and Graphiti. This enhancement improves semantic search precision by learning from endpoint usage patterns and reducing context bloat.

## What's New

### Enhanced Search Intelligence
- **Graph-based re-ranking**: ChromaDB results are re-ranked using temporal knowledge graphs
- **Reduced results**: Returns 2-3 most relevant endpoints instead of 5
- **Learning system**: Improves over time by tracking successful endpoint usage
- **Relationship awareness**: Understands endpoint workflows and common usage patterns

### Key Features
1. **Usage Frequency Tracking** - Prioritizes frequently used endpoints
2. **Recency Scoring** - Favors recently successful endpoints
3. **Relationship Analysis** - Identifies endpoints commonly used together
4. **Success Rate Monitoring** - Learns from successful vs. failed operations

## Architecture

```
┌─────────────────┐
│  User Query     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  ChromaDB       │ ← Vector search returns 6 candidates
│  (Vector)       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  GraphReranker  │ ← Graph intelligence re-ranks
│  (Neo4j+        │
│   Graphiti)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Top 3 Results  │ ← Most relevant endpoints returned
└─────────────────┘
```

## Configuration

### Environment Variables

Add to your `.env` file:

```bash
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password_here
NEO4J_DATABASE=neo4j

# Graphiti Configuration
GRAPHITI_ENABLED=true
GRAPHITI_LLM_MODEL=gpt-4
GRAPHITI_EMBEDDING_MODEL=text-embedding-3-small

# OpenAI API key (required for Graphiti)
OPENAI_API_KEY=your_openai_api_key_here

# Search Results (now defaults to 3)
DEFAULT_SEARCH_RESULTS=3
```

### Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | _(required)_ | Neo4j password |
| `NEO4J_DATABASE` | `neo4j` | Neo4j database name |
| `GRAPHITI_ENABLED` | `true` | Enable/disable graph re-ranking |
| `GRAPHITI_LLM_MODEL` | `gpt-4` | LLM model for Graphiti |
| `GRAPHITI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `DEFAULT_SEARCH_RESULTS` | `3` | Number of results to return |

## Setup

### 1. Install Neo4j

**Docker (Recommended):**
```bash
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your_password \
  neo4j:latest
```

**Or use Neo4j AuraDB** (managed cloud service):
- Sign up at https://neo4j.com/cloud/aura/
- Create a free instance
- Use the provided connection URI

### 2. Install Dependencies

```bash
uv sync
```

This will install:
- `neo4j>=5.27.0`
- `graphiti-core>=0.3.5`

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your Neo4j credentials and OpenAI API key
```

### 4. Start the Server

```bash
# Stdio mode (for MCP clients)
uv run python server.py

# HTTP mode (for testing)
uv run python server_http.py
```

## Usage

### Automatic Operation

The graph re-ranker works automatically when enabled:

1. **User searches** for endpoints using `api_request_schema` tool
2. **ChromaDB returns** 6 candidate results (2x the requested amount)
3. **GraphReranker scores** each candidate based on:
   - Usage frequency (0-40 points)
   - Recency (0-30 points)
   - Relationships (0-20 points)
   - Success rate (0-10 points)
4. **Top 3 results** are returned to the user

### Manual Usage Recording (Optional)

You can manually record endpoint usage to improve learning:

```python
from helpers.graph_reranker import GraphReranker

reranker = GraphReranker()

# Record successful usage
await reranker.record_usage(
    endpoint=endpoint_metadata,
    query="list all devices",
    success=True,
    context={
        "related_endpoints": ["GET /api/dcim/locations/"]
    }
)
```

### Health Check

Check graph system health:

```python
health = reranker.health_check()
print(health)
# {
#     "enabled": True,
#     "neo4j_connected": True,
#     "graphiti_initialized": True,
#     "errors": []
# }
```

## Scoring Algorithm

The re-ranker uses a weighted scoring system (max 100 points):

### 1. Usage Frequency (0-40 points)
- Logarithmic scaling
- 1 use = 5 pts, 10 uses = 15 pts, 100 uses = 25 pts
- Favors proven endpoints

### 2. Recency (0-30 points)
- Time-based decay function
- <1 hour = 30 pts (full)
- 24 hours = 15 pts (half)
- >1 week = 0 pts
- Prioritizes recently successful endpoints

### 3. Relationship Strength (0-20 points)
- Counts connections to other endpoints
- Each related endpoint = 2 pts (max 10)
- Relationship weight bonus (max 10)
- Understands workflows

### 4. Success Rate (0-10 points)
- Percentage of successful uses
- 100% success = 10 pts
- 0% success = 0 pts
- Untested = 5 pts (neutral)

## Disabling Graph Re-ranking

To disable and revert to pure ChromaDB search:

```bash
# In .env
GRAPHITI_ENABLED=false
```

The system will gracefully fall back to ChromaDB-only operation.

## Troubleshooting

### Neo4j Connection Issues

**Problem:** `Failed to connect to Neo4j`

**Solutions:**
1. Verify Neo4j is running: `docker ps | grep neo4j`
2. Check connection URI in `.env`
3. Verify credentials are correct
4. Test connection: `curl http://localhost:7474`

### Graphiti Not Initialized

**Problem:** `Graphiti not initialized`

**Solutions:**
1. Check OpenAI API key is set
2. Verify Neo4j connection is working
3. Check logs for detailed error messages

### Slow Performance

**Problem:** Searches are slower than before

**Solutions:**
1. Ensure Neo4j indexes are created (automatic after first use)
2. Reduce `DEFAULT_SEARCH_RESULTS` if returning too many
3. Check Neo4j query performance: `PROFILE MATCH ...`
4. Consider disabling graph re-ranking for development

## Performance Considerations

- **First search**: Slower (graph initialization)
- **Subsequent searches**: Fast (<100ms overhead)
- **Memory**: ~50MB additional for Neo4j driver
- **Storage**: Minimal graph data (<1MB for typical usage)

## Development

### Running Tests

```bash
# All tests
uv run pytest tests/test_graph_reranker.py -v

# Specific test
uv run pytest tests/test_graph_reranker.py::TestGraphReranker::test_rerank_success -v

# With coverage
uv run pytest tests/test_graph_reranker.py --cov=helpers.graph_reranker
```

### Code Quality

```bash
# Lint
uv run ruff check helpers/graph_reranker.py

# Format
uv run ruff format helpers/graph_reranker.py

# Type check (if using mypy)
uv run mypy helpers/graph_reranker.py
```

## Architecture Decisions

### Why Neo4j + Graphiti?

1. **Temporal context**: Graphiti provides time-aware knowledge graphs
2. **Relationship modeling**: Neo4j excels at relationship queries
3. **Learning over time**: Graph structure naturally captures usage patterns
4. **Explainable**: Score components are traceable and debuggable

### Why Keep ChromaDB?

1. **Fast vector search**: Excellent for initial candidate retrieval
2. **No breaking changes**: Existing functionality preserved
3. **Graceful degradation**: Works without graph system
4. **Separation of concerns**: Vector search vs. graph intelligence

### Design Principles

1. **DRY**: No duplication between ChromaDB and graph logic
2. **KISS**: Simple scoring algorithm, clear responsibilities
3. **Fail-safe**: Graph failures don't break search
4. **Observable**: Comprehensive logging and health checks

## Future Enhancements

Potential improvements for future iterations:

1. **User-specific learning**: Personalize results per user
2. **Context propagation**: Use previous queries in same session
3. **A/B testing**: Compare graph vs. non-graph results
4. **Query expansion**: Use graph to suggest related endpoints
5. **Anomaly detection**: Flag unusual endpoint usage patterns

## Support

For issues or questions:
1. Check logs: Look for `[graph_reranker]` entries
2. Run health check: Verify Neo4j connectivity
3. Test without graph: Set `GRAPHITI_ENABLED=false`
4. Open GitHub issue with logs and configuration

## License

Same as parent project (see LICENSE file).
