"""Repository configuration management for Nautobot MCP Knowledge Base.

This module handles loading and managing repository configurations
from JSON files, making the system extensible.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class RepositoryConfig:
    """Represents a single repository configuration."""

    def __init__(
        self,
        name: str,
        description: str = "",
        priority: int = 5,
        enabled: bool = True,
        branch: str = "main",
        file_patterns: Optional[List[str]] = None,
    ):
        self.name = name
        self.description = description
        self.priority = priority
        self.enabled = enabled
        self.branch = branch
        self.file_patterns = file_patterns or [".py", ".md", ".txt", ".rst"]

    @classmethod
    def from_dict(cls, data: Dict) -> "RepositoryConfig":
        """Create RepositoryConfig from dictionary."""
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            priority=data.get("priority", 5),
            enabled=data.get("enabled", True),
            branch=data.get("branch", "main"),
            file_patterns=data.get("file_patterns", [".py", ".md", ".txt", ".rst"]),
        )

    def to_dict(self) -> Dict:
        """Convert RepositoryConfig to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "priority": self.priority,
            "enabled": self.enabled,
            "branch": self.branch,
            "file_patterns": self.file_patterns,
        }


class RepositoryConfigManager:
    """Manages repository configurations from JSON files."""

    def __init__(self, config_dir: Optional[str] = None):
        """Initialize the configuration manager.

        Args:
            config_dir: Directory containing configuration files
        """
        if config_dir:
            self.config_dir = Path(config_dir)
        else:
            # Default to config directory in project root
            self.config_dir = Path(__file__).parent.parent / "config"

        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.repos_file = self.config_dir / "repositories.json"
        self.user_repos_file = self.config_dir / "user_repositories.json"

    def load_repositories(self) -> List[RepositoryConfig]:
        """Load all repository configurations.

        Returns:
            List of enabled repository configurations sorted by priority
        """
        repos = []

        # Load main repositories file
        if self.repos_file.exists():
            repos.extend(self._load_repos_from_file(self.repos_file))

        # Load user-specific repositories file
        if self.user_repos_file.exists():
            repos.extend(self._load_repos_from_file(self.user_repos_file))

        # Filter enabled repos and sort by priority
        enabled_repos = [repo for repo in repos if repo.enabled]
        enabled_repos.sort(key=lambda x: x.priority)

        return enabled_repos

    def _load_repos_from_file(self, file_path: Path) -> List[RepositoryConfig]:
        """Load repositories from a JSON file."""
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            repos = []
            # Handle different sections in the JSON
            for section_name in ["official_repos", "community_repos", "custom_repos"]:
                if section_name in data:
                    for repo_data in data[section_name]:
                        repos.append(RepositoryConfig.from_dict(repo_data))

            logger.info(f"Loaded {len(repos)} repositories from {file_path}")
            return repos

        except Exception as e:
            logger.error(f"Failed to load repositories from {file_path}: {e}")
            return []

    def add_user_repository(self, repo_config: RepositoryConfig) -> bool:
        """Add a repository to user configuration.

        Args:
            repo_config: Repository configuration to add

        Returns:
            True if successfully added
        """
        try:
            # Load existing user repos
            user_repos = []
            if self.user_repos_file.exists():
                with open(self.user_repos_file, "r") as f:
                    data = json.load(f)
                    user_repos = data.get("custom_repos", [])

            # Check if repo already exists
            repo_names = [repo["name"] for repo in user_repos]
            if repo_config.name in repo_names:
                logger.warning(
                    f"Repository {repo_config.name} already exists in user config"
                )
                return False

            # Add new repo
            user_repos.append(repo_config.to_dict())

            # Save updated config
            data = {"custom_repos": user_repos}
            with open(self.user_repos_file, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Added repository {repo_config.name} to user config")
            return True

        except Exception as e:
            logger.error(f"Failed to add user repository: {e}")
            return False

    def remove_user_repository(self, repo_name: str) -> bool:
        """Remove a repository from user configuration.

        Args:
            repo_name: Name of repository to remove

        Returns:
            True if successfully removed
        """
        try:
            if not self.user_repos_file.exists():
                return False

            with open(self.user_repos_file, "r") as f:
                data = json.load(f)

            user_repos = data.get("custom_repos", [])
            original_count = len(user_repos)

            # Filter out the repository
            user_repos = [repo for repo in user_repos if repo["name"] != repo_name]

            if len(user_repos) == original_count:
                logger.warning(f"Repository {repo_name} not found in user config")
                return False

            # Save updated config
            data["custom_repos"] = user_repos
            with open(self.user_repos_file, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Removed repository {repo_name} from user config")
            return True

        except Exception as e:
            logger.error(f"Failed to remove user repository: {e}")
            return False

    def create_sample_user_config(self) -> None:
        """Create a sample user repositories configuration file."""
        sample_config = {
            "custom_repos": [
                {
                    "name": "your-org/your-nautobot-plugin",
                    "description": "Your custom Nautobot plugin",
                    "priority": 3,
                    "enabled": False,
                    "branch": "main",
                    "file_patterns": [".py", ".md", ".txt", ".rst"],
                }
            ]
        }

        try:
            with open(self.user_repos_file, "w") as f:
                json.dump(sample_config, f, indent=2)
            logger.info(f"Created sample user config at {self.user_repos_file}")
        except Exception as e:
            logger.error(f"Failed to create sample user config: {e}")

    def get_enabled_repo_names(self) -> List[str]:
        """Get list of enabled repository names."""
        repos = self.load_repositories()
        return [repo.name for repo in repos]

    def get_repo_config(self, repo_name: str) -> Optional[RepositoryConfig]:
        """Get configuration for a specific repository.

        Args:
            repo_name: Name of the repository

        Returns:
            Repository configuration or None if not found
        """
        repos = self.load_repositories()
        for repo in repos:
            if repo.name == repo_name:
                return repo
        return None
