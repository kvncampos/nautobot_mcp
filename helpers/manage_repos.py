#!/usr/bin/env python3
"""
Repository Management Script for Nautobot MCP Knowledge Base

This script provides command-line tools for managing repositories in the knowledge base.

Usage:
    uv run manage_repos.py <command> [options]
"""

import argparse
import sys
from typing import Optional

from helpers.nb_kb_v2 import EnhancedNautobotKnowledge
from utils.repo_config import RepositoryConfig, RepositoryConfigManager


def list_repositories(config_manager: RepositoryConfigManager, repo_type: str):
    """List configured repositories."""
    all_repos = config_manager.load_repositories()

    # For now, we'll show all repos since we don't have a way to distinguish types
    if not all_repos:
        print("No repositories configured.")
        return

    print("\nConfigured Repositories:")
    print("-" * 70)
    print(f"{'Repository':<30} {'Priority':<8} {'Enabled':<8} {'Branch':<10}")
    print("-" * 70)

    for repo in all_repos:
        enabled_str = "✓" if repo.enabled else "✗"
        print(f"{repo.name:<30} {repo.priority:<8} {enabled_str:<8} {repo.branch:<10}")
        if repo.description:
            print(f"  Description: {repo.description}")
        print()


def add_repository(
    config_manager: RepositoryConfigManager,
    repo: str,
    description: Optional[str] = None,
    category: Optional[str] = None,
):
    """Add a repository to the user configuration."""
    try:
        repo_config = RepositoryConfig(
            name=repo,
            description=description or f"User-added repository: {repo}",
            priority=5,
            enabled=True,
            branch="main",
            file_patterns=[".py", ".md", ".txt", ".rst", ".json"],
        )

        success = config_manager.add_user_repository(repo_config)
        if success:
            print(f"✓ Added repository: {repo}")
        else:
            print(f"✗ Failed to add repository {repo} (may already exist)")
    except Exception as e:
        print(f"✗ Failed to add repository {repo}: {e}")


def remove_repository(config_manager: RepositoryConfigManager, repo: str):
    """Remove a repository from the user configuration."""
    try:
        success = config_manager.remove_user_repository(repo)
        if success:
            print(f"✓ Removed repository: {repo}")
        else:
            print(f"✗ Failed to remove repository {repo} (may not exist)")
    except Exception as e:
        print(f"✗ Failed to remove repository {repo}: {e}")


def update_repositories(
    kb: EnhancedNautobotKnowledge, repo: Optional[str] = None, force: bool = False
):
    """Update repository indexes."""
    config_manager = RepositoryConfigManager()

    if repo:
        print(f"Updating repository: {repo}")
        repo_config = config_manager.get_repo_config(repo)
        if not repo_config:
            print(f"✗ Repository {repo} not found in configuration")
            return

        success = kb.update_repository(repo_config, force)
        if success:
            print(f"✓ Updated {repo}")
        else:
            print(f"• {repo} was already up to date or failed to update")
    else:
        print("Updating all repositories...")
        results = kb.initialize_all_repositories(force)
        updated_count = sum(results.values())
        print(f"✓ Updated {updated_count}/{len(results)} repositories")


def initialize_repositories(kb: EnhancedNautobotKnowledge, force: bool = False):
    """Initialize all repositories."""
    print("Initializing all repositories...")
    if force:
        print("• Force initialization enabled")

    try:
        results = kb.initialize_all_repositories(force)
        updated_count = sum(results.values())
        print(f"✓ Successfully initialized {updated_count}/{len(results)} repositories")

        # Show details
        for repo_name, was_updated in results.items():
            status = "✓ Updated" if was_updated else "• Skipped (up to date)"
            print(f"  {repo_name}: {status}")

    except Exception as e:
        print(f"✗ Failed to initialize repositories: {e}")


def show_status(kb: EnhancedNautobotKnowledge):
    """Show repository status."""
    config_manager = RepositoryConfigManager()
    all_repos = config_manager.load_repositories()
    stats = kb.get_repository_stats()

    print("\nRepository Status:")
    print("-" * 80)
    print(f"{'Repository':<30} {'Documents':<12} {'Status':<15} {'Enabled':<8}")
    print("-" * 80)

    for repo in all_repos:
        repo_stats = stats.get(repo.name, {})
        doc_count = repo_stats.get("document_count", 0)
        is_enabled = repo_stats.get("enabled", False)
        status = "✓ Indexed" if doc_count > 0 else "Not indexed"
        enabled_str = "✓" if is_enabled else "✗"

        print(f"{repo.name:<30} {doc_count:<12} {status:<15} {enabled_str:<8}")

        if "error" in repo_stats:
            print(f"  Error: {repo_stats['error']}")


def main():
    parser = argparse.ArgumentParser(
        description="Manage repositories in Nautobot MCP Knowledge Base"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # List command
    list_parser = subparsers.add_parser("list", help="List configured repositories")
    list_parser.add_argument(
        "--type",
        choices=["official", "user", "all"],
        default="all",
        help="Type of repositories to list",
    )

    # Add command
    add_parser = subparsers.add_parser("add", help="Add a repository")
    add_parser.add_argument("repo", help="Repository in format 'owner/name'")
    add_parser.add_argument("--description", help="Repository description")
    add_parser.add_argument("--category", help="Repository category")

    # Remove command
    remove_parser = subparsers.add_parser("remove", help="Remove a repository")
    remove_parser.add_argument("repo", help="Repository in format 'owner/name'")

    # Update command
    update_parser = subparsers.add_parser("update", help="Update repository indexes")
    update_parser.add_argument(
        "--repo", help="Specific repository to update (owner/name)"
    )
    update_parser.add_argument(
        "--force", action="store_true", help="Force update even if no changes"
    )

    # Initialize command
    init_parser = subparsers.add_parser("init", help="Initialize all repositories")
    init_parser.add_argument(
        "--force", action="store_true", help="Force initialization of all repos"
    )

    # Status command
    subparsers.add_parser("status", help="Show repository status")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        config_manager = RepositoryConfigManager()
        kb = EnhancedNautobotKnowledge()

        if args.command == "list":
            list_repositories(config_manager, args.type)
        elif args.command == "add":
            add_repository(config_manager, args.repo, args.description, args.category)
        elif args.command == "remove":
            remove_repository(config_manager, args.repo)
        elif args.command == "update":
            update_repositories(kb, args.repo, args.force)
        elif args.command == "init":
            initialize_repositories(kb, args.force)
        elif args.command == "status":
            show_status(kb)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
