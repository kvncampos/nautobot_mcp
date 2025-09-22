#!/usr/bin/env python3
"""
Simple test script to verify SSL configuration is working.
"""

import os
import sys
from pathlib import Path

import requests
import urllib3

# Set SSL_VERIFY to false for testing BEFORE importing config
os.environ["SSL_VERIFY"] = "false"

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.config import config  # noqa: E402


def test_ssl_config():
    """Test that SSL verification setting is properly loaded."""
    print(f"SSL_VERIFY setting: {config.SSL_VERIFY}")
    print(f"NAUTOBOT_BASE_URL: {config.NAUTOBOT_BASE_URL}")
    print(f"Has token: {'Yes' if config.NAUTOBOT_TOKEN else 'No'}")

    # Test API request with SSL verification disabled
    try:
        # Disable SSL warnings if SSL verification is disabled
        if not config.SSL_VERIFY:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        headers = config.get_headers()
        url = config.get_full_url("/dcim/locations/")

        print(f"\nTesting GET request to: {url}")
        print(f"SSL Verify: {config.SSL_VERIFY}")

        response = requests.get(
            url,
            headers=headers,
            params={"name": "DFW-ATO"},
            timeout=10,
            verify=config.SSL_VERIFY,
        )

        print(f"Response status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"Found {len(data.get('results', []))} locations")
            for location in data.get("results", []):
                print(f"  - {location.get('name', 'Unknown')}")
        else:
            print(f"Response: {response.text[:200]}...")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    test_ssl_config()
