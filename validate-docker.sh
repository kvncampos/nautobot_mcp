#!/usr/bin/env bash
# Docker Installation Validation Script for Nautobot MCP
# This script validates the Docker installation and configuration

set -e

echo "======================================"
echo "Nautobot MCP Docker Validation Script"
echo "======================================"
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print success
print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

# Function to print error
print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Function to print warning
print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Check Docker installation
echo "1. Checking Docker installation..."
if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version)
    print_success "Docker is installed: $DOCKER_VERSION"
else
    print_error "Docker is not installed"
    echo "   Please install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check Docker Compose installation
echo ""
echo "2. Checking Docker Compose installation..."
if docker compose version &> /dev/null; then
    COMPOSE_VERSION=$(docker compose version)
    print_success "Docker Compose is installed: $COMPOSE_VERSION"
else
    print_error "Docker Compose is not installed"
    echo "   Please install Docker Compose: https://docs.docker.com/compose/install/"
    exit 1
fi

# Check if Docker daemon is running
echo ""
echo "3. Checking Docker daemon..."
if docker info &> /dev/null; then
    print_success "Docker daemon is running"
else
    print_error "Docker daemon is not running"
    echo "   Please start Docker daemon"
    exit 1
fi

# Check for .env file
echo ""
echo "4. Checking for .env file..."
if [ -f .env ]; then
    print_success ".env file exists"
    
    # Check for required environment variables
    echo ""
    echo "5. Checking required environment variables..."
    
    # Safely parse required variables from .env without executing its contents
    # Only parse KEY=VALUE lines, ignore comments and empty lines
    if [ -z "${NAUTOBOT_ENV+x}" ]; then
        NAUTOBOT_ENV=$(grep -E '^[[:space:]]*NAUTOBOT_ENV=' .env 2>/dev/null | head -n1 | cut -d '=' -f2- | sed 's/^["'\'']\(.*\)["'\'']$/\1/')
    fi
    if [ -z "${GITHUB_TOKEN+x}" ]; then
        GITHUB_TOKEN=$(grep -E '^[[:space:]]*GITHUB_TOKEN=' .env 2>/dev/null | head -n1 | cut -d '=' -f2- | sed 's/^["'\'']\(.*\)["'\'']$/\1/')
    fi
    
    if [ -n "$NAUTOBOT_ENV" ]; then
        print_success "NAUTOBOT_ENV is set: $NAUTOBOT_ENV"
    else
        print_warning "NAUTOBOT_ENV is not set (defaulting to 'local')"
    fi
    
    if [ -n "$GITHUB_TOKEN" ]; then
        print_success "GITHUB_TOKEN is set"
    else
        print_warning "GITHUB_TOKEN is not set (knowledge base features may be limited)"
    fi
    
else
    print_error ".env file not found"
    echo "   Please copy .env.example to .env and configure it"
    echo "   Run: cp .env.example .env"
    exit 1
fi

# Validate docker-compose.yml
echo ""
echo "6. Validating docker-compose.yml..."
if docker compose config --quiet; then
    print_success "docker-compose.yml is valid"
else
    print_error "docker-compose.yml has errors"
    exit 1
fi

# Check available disk space
echo ""
echo "7. Checking disk space..."
AVAILABLE_SPACE=$(df -h . | awk 'NR==2 {print $4}')
print_success "Available disk space: $AVAILABLE_SPACE"
if [ "$(df . | awk 'NR==2 {print $4}')" -lt 2097152 ]; then
    print_warning "Less than 2GB of free space available"
    echo "   Recommended: At least 2GB for Docker images and data"
fi

# Check for existing volumes
echo ""
echo "8. Checking for existing volumes..."
if docker volume ls | grep -q nautobot-mcp-chroma; then
    print_success "ChromaDB volume exists"
else
    print_warning "ChromaDB volume does not exist (will be created on first run)"
fi

if docker volume ls | grep -q nautobot-mcp-models; then
    print_success "Models volume exists"
else
    print_warning "Models volume does not exist (will be created on first run)"
fi

# Check if container is already running
echo ""
echo "9. Checking for running containers..."
if docker ps | grep -q nautobot-mcp-server; then
    print_warning "Nautobot MCP container is already running"
    echo "   Run 'docker-compose ps' to see status"
    echo "   Run 'docker-compose logs -f' to view logs"
else
    print_success "No existing containers running"
fi

# Summary
echo ""
echo "======================================"
echo "Validation Summary"
echo "======================================"
echo ""
echo "Docker Installation: OK"
echo "Docker Compose: OK"
echo "Configuration: OK"
echo ""
echo "Next steps:"
echo "1. Build and start the container:"
echo "   docker-compose up -d"
echo ""
echo "2. View logs:"
echo "   docker-compose logs -f"
echo ""
echo "3. Check status:"
echo "   docker-compose ps"
echo ""
echo "For stdio mode (default):"
echo "   docker-compose up -d"
echo ""
echo "For HTTP mode:"
echo "   MCP_TRANSPORT=http docker-compose up -d"
echo ""
print_success "Validation complete!"
