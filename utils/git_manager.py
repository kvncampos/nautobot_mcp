"""Git repository management for Nautobot MCP Knowledge Base.

This module handles cloning, updating, and tracking git repositories
for the knowledge base system.
"""

import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import requests

from utils.config import config

logger = logging.getLogger(__name__)


class GitRepoManager:
    """Manages git repositories for the knowledge base."""

    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the Git repository manager.

        Args:
            cache_dir: Directory to store cloned repositories. If None, uses temp directory.
        """
        self.cache_dir = (
            Path(cache_dir)
            if cache_dir
            else Path(tempfile.gettempdir()) / "nautobot_mcp_repos"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / "repo_metadata.json"
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """Load repository metadata from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
        return {}

    def _save_metadata(self) -> None:
        """Save repository metadata to disk."""
        try:
            with open(self.metadata_file, "w") as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")

    def get_remote_commit_hash(
        self, repo_url: str, branch: str = "main"
    ) -> Optional[str]:
        """Get the latest commit hash from remote repository.

        Args:
            repo_url: Repository URL (e.g., "nautobot/nautobot")
            branch: Branch name to check

        Returns:
            Latest commit hash or None if failed
        """
        try:
            # Convert repo_url format if needed
            if not repo_url.startswith("http"):
                api_url = f"https://api.github.com/repos/{repo_url}/commits/{branch}"
            else:
                # Extract owner/repo from URL
                parts = repo_url.replace("https://github.com/", "").replace(".git", "")
                api_url = f"https://api.github.com/repos/{parts}/commits/{branch}"

            headers = {"Accept": "application/vnd.github+json"}

            # Add GitHub token if available for authentication and higher rate limits
            if config.GITHUB_TOKEN:
                headers["Authorization"] = f"token {config.GITHUB_TOKEN}"

            response = requests.get(api_url, headers=headers, timeout=10)
            response.raise_for_status()

            data = response.json()
            return data["sha"]

        except Exception as e:
            logger.error(f"Failed to get remote commit hash for {repo_url}: {e}")
            return None

    def get_local_commit_hash(self, repo_path: Path) -> Optional[str]:
        """Get the current commit hash of a local repository.

        Args:
            repo_path: Path to the local repository

        Returns:
            Current commit hash or None if failed
        """
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except Exception as e:
            logger.error(f"Failed to get local commit hash for {repo_path}: {e}")
            return None

    def clone_or_update_repo(
        self, repo_url: str, branch: str = "main"
    ) -> Tuple[Optional[Path], bool]:
        """Clone or update a repository.

        Args:
            repo_url: Repository URL (e.g., "nautobot/nautobot")
            branch: Branch to clone/update

        Returns:
            Tuple of (repo_path, was_updated)
        """
        # Normalize repo name for directory
        repo_name = repo_url.replace("/", "_").replace(":", "_")
        repo_path = self.cache_dir / repo_name

        try:
            # Get remote commit hash
            remote_hash = self.get_remote_commit_hash(repo_url, branch)
            if not remote_hash:
                logger.error(f"Failed to get remote hash for {repo_url}")
                return None, False

            # Check if we need to update
            repo_key = f"{repo_url}:{branch}"
            local_hash = self.metadata.get(repo_key, {}).get("commit_hash")

            # Also check if we're on the correct branch
            needs_branch_switch = False
            if repo_path.exists():
                try:
                    current_branch_result = subprocess.run(
                        ["git", "branch", "--show-current"],
                        cwd=repo_path,
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    current_branch = current_branch_result.stdout.strip()
                    needs_branch_switch = current_branch != branch
                except subprocess.CalledProcessError:
                    # If we can't determine the current branch, assume we need to update
                    needs_branch_switch = True

            if (
                repo_path.exists()
                and local_hash == remote_hash
                and not needs_branch_switch
            ):
                logger.info(f"Repository {repo_url} is up to date")
                return repo_path, False

            # Clone or update repository
            if repo_path.exists():
                logger.info(f"Updating repository {repo_url}")
                self._git_pull(repo_path, branch)
            else:
                logger.info(f"Cloning repository {repo_url}")
                self._git_clone(repo_url, repo_path, branch)

            # Verify local hash matches remote
            actual_local_hash = self.get_local_commit_hash(repo_path)
            if actual_local_hash != remote_hash:
                logger.warning(
                    f"Local hash {actual_local_hash} doesn't match remote {remote_hash}"
                )

            # Update metadata
            self.metadata[repo_key] = {
                "commit_hash": remote_hash,
                "local_path": str(repo_path),
                "last_updated": self._get_current_timestamp(),
            }
            self._save_metadata()

            return repo_path, True

        except Exception as e:
            logger.error(f"Failed to clone/update {repo_url}: {e}")
            return None, False

    def _git_clone(self, repo_url: str, repo_path: Path, branch: str) -> None:
        """Clone a git repository."""
        # Convert to full GitHub URL if needed
        if not repo_url.startswith("http"):
            if config.GITHUB_TOKEN:
                # Use token for private repository access
                full_url = f"https://{config.GITHUB_TOKEN}@github.com/{repo_url}.git"
            else:
                full_url = f"https://github.com/{repo_url}.git"
        else:
            full_url = repo_url

        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                branch,
                full_url,
                str(repo_path),
            ],
            check=True,
            capture_output=True,
        )

    def _git_pull(self, repo_path: Path, branch: str) -> None:
        """Update an existing git repository."""
        try:
            # First, check what branch we're currently on
            current_branch_result = subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True,
            )
            current_branch = current_branch_result.stdout.strip()

            # Fetch the target branch
            subprocess.run(
                ["git", "fetch", "origin", branch],
                cwd=repo_path,
                check=True,
                capture_output=True,
            )

            # If we're not on the target branch, we need to switch
            if current_branch != branch:
                # Check if the local branch exists
                branch_check = subprocess.run(
                    ["git", "show-ref", "--verify", "--quiet", f"refs/heads/{branch}"],
                    cwd=repo_path,
                    capture_output=True,
                )

                if branch_check.returncode != 0:
                    # Local branch doesn't exist, check if remote tracking branch exists
                    remote_check = subprocess.run(
                        [
                            "git",
                            "show-ref",
                            "--verify",
                            "--quiet",
                            f"refs/remotes/origin/{branch}",
                        ],
                        cwd=repo_path,
                        capture_output=True,
                    )

                    if remote_check.returncode == 0:
                        # Remote tracking branch exists, create local branch from it
                        subprocess.run(
                            ["git", "checkout", "-b", branch, f"origin/{branch}"],
                            cwd=repo_path,
                            check=True,
                            capture_output=True,
                        )
                    else:
                        # Remote tracking branch doesn't exist, fetch it first
                        subprocess.run(
                            ["git", "fetch", "origin", f"{branch}:{branch}"],
                            cwd=repo_path,
                            check=True,
                            capture_output=True,
                        )
                        subprocess.run(
                            ["git", "checkout", branch],
                            cwd=repo_path,
                            check=True,
                            capture_output=True,
                        )
                else:
                    # Local branch exists, switch to it
                    subprocess.run(
                        ["git", "checkout", branch],
                        cwd=repo_path,
                        check=True,
                        capture_output=True,
                    )

            # Now reset to the latest remote commit
            subprocess.run(
                ["git", "reset", "--hard", f"origin/{branch}"],
                cwd=repo_path,
                check=True,
                capture_output=True,
            )

        except subprocess.CalledProcessError as e:
            # If we still can't update due to shallow clone issues, re-clone
            logger.warning(f"Standard git pull failed, attempting to re-clone: {e}")

            # Get the remote URL before deleting
            try:
                remote_result = subprocess.run(
                    ["git", "remote", "get-url", "origin"],
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                repo_url = remote_result.stdout.strip()

                # Convert back to short form if it's a GitHub URL
                if repo_url.startswith("https://github.com/"):
                    repo_url = repo_url.replace("https://github.com/", "").replace(
                        ".git", ""
                    )

                # Remove the existing directory and re-clone
                import shutil

                shutil.rmtree(repo_path)
                self._git_clone(repo_url, repo_path, branch)

            except Exception as clone_error:
                logger.error(f"Failed to re-clone repository: {clone_error}")
                raise

    def _get_current_timestamp(self) -> str:
        """Get current timestamp as ISO string."""
        from datetime import datetime

        return datetime.now().isoformat()

    def needs_update(self, repo_url: str, branch: str = "main") -> bool:
        """Check if a repository needs updating.

        Args:
            repo_url: Repository URL
            branch: Branch name

        Returns:
            True if update is needed
        """
        repo_key = f"{repo_url}:{branch}"
        local_hash = self.metadata.get(repo_key, {}).get("commit_hash")
        remote_hash = self.get_remote_commit_hash(repo_url, branch)

        return local_hash != remote_hash

    def cleanup_old_repos(self) -> None:
        """Clean up repositories that are no longer in use."""
        # This could be implemented to remove repos not in current config
        pass
