"""Configuration management for Nautobot MCP Server.

This module centralizes all configuration settings and provides
a clean interface for accessing environment variables with defaults.
"""

import os

from dotenv import load_dotenv

from helpers.creds_helper import NautobotCredentialMapping, NautobotEnv

# Load environment variables from .env file if it exists
load_dotenv()


class Config:
    """Configuration class that handles all application settings."""

    # Environment Configuration - Change this one setting to switch environments
    NAUTOBOT_ENV: NautobotEnv = NautobotEnv(os.getenv("NAUTOBOT_ENV", "local"))

    # Get credentials based on environment
    _credentials = NautobotCredentialMapping.get_credentials(NAUTOBOT_ENV)

    # API Configuration - Now dynamically set based on environment
    NAUTOBOT_BASE_URL: str = _credentials["NAUTOBOT_URL"].rstrip("/") + "/api"
    NAUTOBOT_TOKEN: str = _credentials["NAUTOBOT_TOKEN"]
    GLOBAL_TOOL_PROMPT: str = (
        _credentials["NAUTOBOT_URL"].rstrip("/") + "/api/swagger.json"
    )

    # Github Configuration
    GITHUB_TOKEN: str = os.getenv("GITHUB_TOKEN", "")

    # Server Configuration
    API_PREFIX: str = os.getenv("API_PREFIX", "nautobot_openapi")
    SERVER_NAME: str = os.getenv("SERVER_NAME", "any_openapi")
    SERVER_VERSION: str = os.getenv("SERVER_VERSION", "0.2.0")

    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # ChromaDB Configuration
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    DEFAULT_SEARCH_RESULTS: int = int(os.getenv("DEFAULT_SEARCH_RESULTS", "5"))

    # Knowledge Base Mode Configuration
    KNOWLEDGE_BASE_MODE: str = os.getenv(
        "KNOWLEDGE_BASE_MODE", "offline"
    )  # "offline", "ai", or "single_model"

    # External Services
    POSTHOG_API_KEY: str = os.getenv("POSTHOG_API_KEY", "disable")

    # Request Timeouts
    API_TIMEOUT: int = int(os.getenv("API_TIMEOUT", "10"))

    # SSL Configuration
    SSL_VERIFY: bool = os.getenv("SSL_VERIFY", "true").lower() in ("true", "1", "yes")

    @classmethod
    def validate(cls) -> None:
        """Delegate validation to validate_config.py."""
        import utils.validate_config as validate_config

        validate_config.validate_config()

    @classmethod
    def get_headers(cls) -> dict:
        """Get standardized headers for Nautobot API requests."""
        headers = {"Content-Type": "application/json"}
        if cls.NAUTOBOT_TOKEN:
            headers["Authorization"] = f"Token {cls.NAUTOBOT_TOKEN}"
        return headers

    @classmethod
    def get_full_url(cls, path: str) -> str:
        """Construct full URL for API endpoints."""
        return f"{cls.NAUTOBOT_BASE_URL.rstrip('/')}{path}"


# Create a global config instance
config = Config()

# Validate configuration on import
config.validate()
