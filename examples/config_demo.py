#!/usr/bin/env python3
"""
Demonstration of the new environment-based configuration system.

This script shows how changing just the NAUTOBOT_ENV environment variable
automatically switches all related configuration settings.
"""

import os

from helpers.creds_helper import NautobotEnv
from utils.config import Config


def demonstrate_config_switching():
    """Show how configuration changes based on environment."""

    print("🔧 Nautobot MCP Configuration Demonstration\n")

    # Show all available environments
    print("Available environments:")
    for env in NautobotEnv:
        print(f"  - {env.value}")
    print()

    # Demonstrate switching between environments
    environments_to_test = [NautobotEnv.LOCAL, NautobotEnv.NONPROD, NautobotEnv.PROD]

    for env in environments_to_test:
        # Temporarily set the environment
        os.environ["NAUTOBOT_ENV"] = env.value

        # Reload the config (in practice, you'd restart the application)
        # For demo purposes, we'll create a new Config instance
        from importlib import reload

        import utils.config as config_module

        reload(config_module)
        from utils.config import Config as ReloadedConfig

        print(f"📍 Environment: {env.value.upper()}")
        print(f"   Base URL: {ReloadedConfig.NAUTOBOT_BASE_URL}")
        print(f"   Token: {ReloadedConfig.NAUTOBOT_TOKEN[:20]}...")
        print(f"   Swagger URL: {ReloadedConfig.GLOBAL_TOOL_PROMPT}")
        print()


def show_current_config():
    """Show the current active configuration."""
    print("🎯 Current Active Configuration:")
    print(f"   Environment: {Config.NAUTOBOT_ENV}")
    print(f"   Base URL: {Config.NAUTOBOT_BASE_URL}")
    print(f"   Token: {Config.NAUTOBOT_TOKEN[:20]}...")
    print(f"   Swagger URL: {Config.GLOBAL_TOOL_PROMPT}")
    print(f"   GitHub Token: {'Set' if Config.GITHUB_TOKEN else 'Not set'}")
    print(f"   Log Level: {Config.LOG_LEVEL}")


if __name__ == "__main__":
    print("Current configuration:")
    show_current_config()
    print("\n" + "=" * 60 + "\n")
    demonstrate_config_switching()
    print("💡 To switch environments permanently, set NAUTOBOT_ENV in your .env file")
