#!/usr/bin/env python3
"""
Example script showing how to use the Nautobot MCP tools with SSL verification disabled.
"""

import os

# IMPORTANT: Set SSL_VERIFY=false BEFORE starting the MCP server
# You can either:
# 1. Set it as an environment variable: export SSL_VERIFY=false
# 2. Create a .env file with: SSL_VERIFY=false
# 3. Set it programmatically like below (for testing only)

os.environ["SSL_VERIFY"] = "false"

from utils.config import config


def show_configuration():
    """Display current configuration."""
    print("Current Nautobot Configuration:")
    print(f"  Environment: {config.NAUTOBOT_ENV}")
    print(f"  Base URL: {config.NAUTOBOT_BASE_URL}")
    print(f"  SSL Verify: {config.SSL_VERIFY}")
    print(f"  Has Token: {'Yes' if config.NAUTOBOT_TOKEN else 'No'}")
    print()


def example_usage():
    """Show example of how to use the configuration."""
    print("Example: Setting up environment for MCP server")
    print("1. Create a .env file in your project root with:")
    print("   SSL_VERIFY=false")
    print("   NAUTOBOT_ENV=prod  # or nonprod, or local")
    print()
    print("2. Or export environment variables:")
    print("   export SSL_VERIFY=false")
    print("   export NAUTOBOT_ENV=prod")
    print()
    print("3. Then start your MCP server")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("Nautobot MCP SSL Configuration Guide")
    print("=" * 60)
    show_configuration()
    example_usage()
    print("✓ SSL verification has been successfully disabled!")
    print("✓ You can now connect to Nautobot instances with self-signed certificates")
