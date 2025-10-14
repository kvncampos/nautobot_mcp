# Troubleshooting

Common issues and solutions when using the Nautobot MCP Server.

## Installation Issues

### Python Version Errors

**Symptom:**
```
ERROR: Python 3.12 is not supported
```

**Solution:**
Ensure you're using Python 3.11:

```bash
python --version  # Should show 3.11.x
python3.11 -m venv venv
source venv/bin/activate
```

### Dependency Installation Failures

**Symptom:**
```
ERROR: Could not find a version that satisfies the requirement...
```

**Solution:**
```bash
# Clear pip cache
pip cache purge

# Reinstall
pip install --no-cache-dir -e .
```

## Connection Issues

### Nautobot Connection Failed

**Symptom:**
```
ConnectionError: Failed to connect to Nautobot
```

**Solutions:**

1. **Check URL:**
   ```bash
   curl -I http://localhost:8000/api/
   ```

2. **Verify Token:**
   ```bash
   curl -H "Authorization: Token YOUR_TOKEN" http://localhost:8000/api/
   ```

3. **Check Environment:**
   ```ini
   # In .env
   NAUTOBOT_ENV=local
   NAUTOBOT_URL_LOCAL=http://localhost:8000
   ```

### SSL Certificate Errors

**Symptom:**
```
SSL: CERTIFICATE_VERIFY_FAILED
```

**Solution for Development:**
```ini
# In .env
SSL_VERIFY=False
```

**Solution for Production:**
Install proper certificates or configure SSL properly.

## MCP Integration Issues

### VS Code Can't Find Server

**Symptom:**
Server doesn't appear in Copilot's available tools.

**Solutions:**

1. **Check Configuration Path:**
   Ensure path in `mcp.json` is absolute:
   ```json
   "/Users/yourname/projects/nautobot_mcp"
   ```

2. **Restart VS Code:**
   Close all windows and reopen.

3. **Check Output Log:**
   `View → Output → GitHub Copilot Chat`

### Tools Not Working

**Symptom:**
Copilot doesn't use MCP tools.

**Solutions:**

1. **Verify Server Status:**
   Check VS Code Output panel for errors.

2. **Test Manually:**
   ```bash
   python server.py
   # Should show "Server ready"
   ```

3. **Be Explicit:**
   ```
   Use nautobot_kb_semantic_search to find...
   ```

## Knowledge Base Issues

### Repository Clone Fails

**Symptom:**
```
ERROR: Failed to clone repository
```

**Solutions:**

1. **Add GitHub Token:**
   ```ini
   GITHUB_TOKEN=ghp_your_token_here
   ```

2. **Check Repository Access:**
   ```bash
   git clone https://github.com/owner/repo
   ```

3. **Verify Disk Space:**
   ```bash
   df -h
   ```

### Search Returns No Results

**Symptom:**
Knowledge base searches return empty results.

**Solutions:**

1. **Initialize Repositories:**
   ```python
   from helpers.nb_kb_v2 import EnhancedNautobotKnowledge
   kb = EnhancedNautobotKnowledge()
   kb.initialize_all_repositories(force=True)
   ```

2. **Check Repository Status:**
   Use the `nautobot_kb_repo_status` tool.

3. **Verify ChromaDB:**
   ```bash
   ls -la backend/nautobot_mcp/
   ```

## Performance Issues

### Slow Startup

**Symptom:**
Server takes minutes to start.

**Causes & Solutions:**

1. **First-time Model Download:**
   - Normal on first run (~90MB download)
   - Subsequent starts will be faster

2. **Repository Indexing:**
   - Happens on first run or after updates
   - Can take 5-10 minutes for large repos

3. **Disable Auto-Update:**
   Comment out auto-update in `server.py` temporarily.

### Slow Searches

**Symptom:**
Searches take > 5 seconds.

**Solutions:**

1. **Reduce Search Results:**
   ```ini
   DEFAULT_SEARCH_RESULTS=3
   ```

2. **Use Lighter Model:**
   ```ini
   EMBEDDING_MODEL=all-MiniLM-L6-v2
   ```

3. **Check Disk I/O:**
   ```bash
   iostat -x 1
   ```

## Database Issues

### ChromaDB Permission Errors

**Symptom:**
```
PermissionError: [Errno 13] Permission denied: 'backend/...'
```

**Solution:**
```bash
chmod -R 755 backend/
```

### ChromaDB Corruption

**Symptom:**
```
Database or disk is full
```

**Solution:**
```bash
# Backup
cp -r backend/ backend_backup/

# Remove and reinitialize
rm -rf backend/nautobot_mcp/
python server.py  # Will recreate
```

## API Issues

### 401 Unauthorized

**Symptom:**
All API calls return 401.

**Solutions:**

1. **Check Token:**
   ```bash
   # Test in Nautobot UI
   # Profile → API Tokens → Verify active
   ```

2. **Verify Environment:**
   ```python
   from utils.config import config
   print(config.NAUTOBOT_TOKEN)
   ```

### 429 Rate Limit

**Symptom:**
```
429 Too Many Requests
```

**Solution:**
Wait or implement request throttling in your application.

### 500 Server Error

**Symptom:**
```
500 Internal Server Error
```

**Solution:**
Check Nautobot server logs for the actual error.

## Logging & Debugging

### Enable Debug Logging

```ini
# In .env
LOG_LEVEL=DEBUG
```

### View Logs

```bash
# Run with output
python server.py 2>&1 | tee server.log

# In VS Code
# View → Output → GitHub Copilot Chat
```

### Test Components

```python
# Test endpoint searcher
from helpers.endpoint_searcher_chroma import EndpointSearcherChroma
searcher = EndpointSearcherChroma()
results = searcher.search("device", n_results=3)
print(results)

# Test knowledge base
from helpers.nb_kb_v2 import EnhancedNautobotKnowledge
kb = EnhancedNautobotKnowledge()
results = kb.search("custom fields", n_results=3)
print(results)
```

## Getting Help

### Before Asking for Help

1. Check this troubleshooting guide
2. Review logs with `LOG_LEVEL=DEBUG`
3. Test individual components
4. Verify configuration

### Where to Get Help

- **GitHub Issues:** [Open an issue](https://github.com/kvncampos/nautobot_mcp/issues)
- **Discussions:** [GitHub Discussions](https://github.com/kvncampos/nautobot_mcp/discussions)
- **Email:** kvncampos@duck.com

### What to Include

When reporting issues:

- Error messages (full traceback)
- Configuration (sanitized `.env`)
- Steps to reproduce
- Environment (OS, Python version)
- Logs (with DEBUG level)

---

## Next Steps

- [Configuration](configuration.md) - Configuration reference
- [Examples](examples.md) - Usage examples
- [Architecture](architecture.md) - System design
