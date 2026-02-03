# Docker Quick Start Guide

This guide will help you get the Nautobot MCP server running with Docker in minutes.

## Prerequisites

- Docker installed (version 20.10 or later)
- Docker Compose installed (version 2.0 or later)
- At least 4GB of available RAM
- At least 2GB of free disk space

## Quick Start

### 1. Clone and Configure

```bash
# Clone the repository
git clone <repository-url>
cd nautobot_mcp

# Create your environment file
cp .env.example .env

# Edit .env with your credentials
nano .env  # or use your favorite editor
```

**Minimum required configuration in `.env`:**
```bash
NAUTOBOT_ENV=local  # or nonprod, prod
GITHUB_TOKEN=your_github_token_here  # for knowledge base
```

### 2. Build and Run

**Option A: stdio mode (for MCP clients)**
```bash
docker-compose up -d
```

**Option B: HTTP mode (for web integrations)**
```bash
MCP_TRANSPORT=http docker-compose up -d
```

### 3. Verify It's Running

```bash
# Check container status
docker-compose ps

# Follow logs
docker-compose logs -f

# Check health (HTTP mode only)
curl http://localhost:8000/health
```

## Common Operations

### View Logs
```bash
# Follow all logs
docker-compose logs -f

# View last 100 lines
docker-compose logs --tail=100

# View logs for last 1 hour
docker-compose logs --since=1h
```

### Restart the Server
```bash
docker-compose restart
```

### Stop the Server
```bash
docker-compose down
```

### Update After Code Changes
```bash
# Rebuild and restart
docker-compose up -d --build
```

### Reset Everything (including data)
```bash
# WARNING: This deletes all ChromaDB data
docker-compose down -v
docker-compose up -d
```

## Configuration

### Transport Modes

**stdio mode** - For MCP clients (Claude Desktop, VS Code, etc.)
```bash
# In .env
MCP_TRANSPORT=stdio

# Or via command line
docker-compose up -d
```

**HTTP mode** - For web-based integrations
```bash
# In .env
MCP_TRANSPORT=http
MCP_PORT=8000

# Or via command line
MCP_TRANSPORT=http MCP_PORT=8000 docker-compose up -d
```

### Data Persistence

ChromaDB data and model cache are stored in Docker volumes:

```bash
# List volumes
docker volume ls | grep nautobot-mcp

# Inspect a volume
docker volume inspect nautobot-mcp-chroma

# Backup a volume
docker run --rm -v nautobot-mcp-chroma:/data -v $(pwd):/backup \
  alpine tar czf /backup/chroma-backup.tar.gz -C /data .

# Restore a volume
docker run --rm -v nautobot-mcp-chroma:/data -v $(pwd):/backup \
  alpine tar xzf /backup/chroma-backup.tar.gz -C /data
```

### Resource Limits

Default limits are set in `docker-compose.yml`:
- CPU: 2 cores (limit), 0.5 cores (reservation)
- Memory: 4GB (limit), 1GB (reservation)

To change limits, edit `docker-compose.yml`:
```yaml
deploy:
  resources:
    limits:
      cpus: '4.0'
      memory: 8G
    reservations:
      cpus: '1.0'
      memory: 2G
```

## Troubleshooting

### Container won't start

```bash
# Check logs for errors
docker-compose logs

# Verify configuration
docker-compose config

# Try rebuilding
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### "Port already in use" error

```bash
# Check what's using port 8000
lsof -i :8000  # or netstat -tulpn | grep 8000

# Use a different port
MCP_PORT=9000 docker-compose up -d
```

### Out of memory errors

```bash
# Check current usage
docker stats nautobot-mcp-server

# Increase memory limit in docker-compose.yml
# Then restart:
docker-compose down
docker-compose up -d
```

### Permission errors

```bash
# Check volume permissions
docker-compose exec nautobot-mcp ls -la /app/backend/

# Recreate volumes if needed
docker-compose down -v
docker-compose up -d
```

### ChromaDB data not persisting

```bash
# Verify volumes exist
docker volume ls | grep nautobot-mcp

# Check volume mount
docker-compose exec nautobot-mcp ls -la /app/backend/chroma_db/
```

## Integration with VS Code

Add this to your VS Code MCP configuration:

```json
{
  "servers": {
    "nautobot_mcp": {
      "type": "stdio",
      "command": "docker",
      "args": [
        "run",
        "-i",
        "--rm",
        "--env-file",
        "/path/to/nautobot_mcp/.env",
        "-v",
        "nautobot-mcp-chroma:/app/backend/chroma_db",
        "-v",
        "nautobot-mcp-models:/app/backend/models",
        "nautobot-mcp:latest",
        "--mode",
        "stdio"
      ]
    }
  }
}
```

## Advanced Usage

### Custom Configuration

Mount a custom configuration directory:
```yaml
# Add to docker-compose.yml volumes:
volumes:
  - ./config:/app/config:ro
```

### Multiple Instances

Run multiple instances on different ports:
```bash
# Instance 1 (port 8000)
MCP_PORT=8000 docker-compose -p nautobot-mcp-1 up -d

# Instance 2 (port 9000)
MCP_PORT=9000 docker-compose -p nautobot-mcp-2 up -d
```

### Development Mode

For development with live code reload:
```yaml
# Add to docker-compose.yml:
volumes:
  - .:/app:ro  # Mount source code
```

Then restart on code changes:
```bash
docker-compose restart
```

## Production Deployment

### Best Practices

1. **Use specific tags, not `latest`**
   ```bash
   docker tag nautobot-mcp:latest nautobot-mcp:v1.0.0
   ```

2. **Enable SSL verification**
   ```bash
   SSL_VERIFY=True
   ```

3. **Set appropriate resource limits**
   - Monitor actual usage first
   - Set limits with headroom

4. **Use Docker secrets for tokens**
   ```yaml
   secrets:
     nautobot_token:
       external: true
   ```

5. **Implement health checks**
   - Already configured in docker-compose.yml
   - Monitor with your orchestration platform

6. **Regular backups**
   ```bash
   # Automated backup script
   #!/bin/bash
   DATE=$(date +%Y%m%d_%H%M%S)
   docker run --rm \
     -v nautobot-mcp-chroma:/data \
     -v /backups:/backup \
     alpine tar czf /backup/chroma-${DATE}.tar.gz -C /data .
   ```

## Next Steps

- Configure your Nautobot credentials in `.env`
- Add custom repositories to `config/user_repositories.json`
- Integrate with your MCP client
- Check logs to verify initialization completed
- Test the available tools

## Support

If you encounter issues:
1. Check the logs: `docker-compose logs -f`
2. Verify your `.env` configuration
3. Check the troubleshooting section above
4. Review the main README.md
5. Open an issue on GitHub
