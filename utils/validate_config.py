#!/usr/bin/env python3
"""
Configuration validation script for Nautobot MCP Server.

This script helps validate that your configuration is properly set up
and can connect to your Nautobot instance.
"""

import sys
from typing import List, Tuple

import requests
import urllib3

from utils.config import config

# Disable SSL warnings if SSL verification is disabled
if not config.SSL_VERIFY:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def validate_config() -> List[Tuple[str, bool, str]]:
    """Validate configuration and return results."""
    results = []

    # Test 1: Basic config loading
    try:
        results.append(
            (
                "Configuration loading",
                True,
                f"✓ Loaded with API_PREFIX: {config.API_PREFIX}",
            )
        )
    except Exception as e:
        results.append(
            ("Configuration loading", False, f"✗ Failed to load config: {e}")
        )
        return results

    # Test 2: Required URLs are accessible
    try:
        schema_response = requests.get(
            config.GLOBAL_TOOL_PROMPT,
            headers=config.get_headers(),
            timeout=5,
            verify=config.SSL_VERIFY,
        )
        if schema_response.status_code == 200:
            results.append(
                (
                    "OpenAPI Schema",
                    True,
                    f"✓ Schema accessible at {config.GLOBAL_TOOL_PROMPT}",
                )
            )
        else:
            results.append(
                (
                    "OpenAPI Schema",
                    False,
                    f"✗ Schema returned {schema_response.status_code}",
                )
            )
    except Exception as e:
        results.append(("OpenAPI Schema", False, f"✗ Cannot reach schema URL: {e}"))

    # Test 3: API endpoint connectivity
    try:
        api_response = requests.get(
            config.NAUTOBOT_BASE_URL,
            headers=config.get_headers(),
            timeout=5,
            verify=config.SSL_VERIFY,
        )
        if api_response.status_code in [
            200,
            401,
        ]:  # 401 is OK, means auth required but endpoint exists
            results.append(
                (
                    "API Endpoint",
                    True,
                    f"✓ API endpoint reachable at {config.NAUTOBOT_BASE_URL}",
                )
            )
        else:
            results.append(
                ("API Endpoint", False, f"✗ API returned {api_response.status_code}")
            )
    except Exception as e:
        results.append(("API Endpoint", False, f"✗ Cannot reach API: {e}"))

    # Test 4: Authentication (if token provided)
    if config.NAUTOBOT_TOKEN:
        try:
            auth_response = requests.get(
                f"{config.NAUTOBOT_BASE_URL}/users/users/",  # Common endpoint to test auth
                headers=config.get_headers(),
                timeout=5,
                verify=config.SSL_VERIFY,
            )
            if auth_response.status_code == 200:
                results.append(
                    ("Authentication", True, "✓ Token authentication successful")
                )
            elif auth_response.status_code == 401:
                results.append(
                    ("Authentication", False, "✗ Token authentication failed (401)")
                )
            else:
                results.append(
                    (
                        "Authentication",
                        False,
                        f"✗ Unexpected response: {auth_response.status_code}",
                    )
                )
        except Exception as e:
            results.append(("Authentication", False, f"✗ Auth test failed: {e}"))
    else:
        results.append(
            ("Authentication", False, "⚠ No NAUTOBOT_TOKEN provided (optional)")
        )

    return results


def main():
    """Run configuration validation."""
    print("🔧 Nautobot MCP Server Configuration Validator")
    print("=" * 50)

    results = validate_config()

    all_passed = True
    for test_name, passed, message in results:
        status = "PASS" if passed else "FAIL"
        print(f"{test_name:20} [{status:4}] {message}")
        if not passed and "optional" not in message.lower() and "⚠" not in message:
            all_passed = False

    print("=" * 50)
    if all_passed:
        print("🎉 All critical tests passed! Your configuration looks good.")
        print("\nTo start the MCP server:")
        print("  uv run python server.py")
    else:
        print("❌ Some tests failed. Please check your configuration.")
        print("\nNext steps:")
        print("1. Copy .env.example to .env")
        print("2. Update .env with your Nautobot instance details")
        print("3. Ensure your Nautobot instance is running and accessible")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
