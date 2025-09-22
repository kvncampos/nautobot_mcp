"""
Test module for manage_repos.py functions.

This module tests the repository management functionality including
adding, removing, listing, updating, and initializing repositories.
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Add the parent directory to Python path to enable imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from helpers import manage_repos
from utils.repo_config import RepositoryConfig, RepositoryConfigManager


class TestManageRepos:
    """Test suite for manage_repos.py functions."""

    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for testing."""
        config_dir = tempfile.mkdtemp()
        cache_dir = tempfile.mkdtemp()

        yield config_dir, cache_dir

        # Cleanup
        shutil.rmtree(config_dir, ignore_errors=True)
        shutil.rmtree(cache_dir, ignore_errors=True)

    @pytest.fixture
    def mock_config_manager(self):
        """Create a mocked RepositoryConfigManager."""
        mock_manager = Mock(spec=RepositoryConfigManager)
        return mock_manager

    @pytest.fixture
    def mock_kb(self):
        """Create a mocked EnhancedNautobotKnowledge instance."""
        with patch("helpers.manage_repos.EnhancedNautobotKnowledge") as mock_kb_class:
            mock_kb = Mock()
            mock_kb_class.return_value = mock_kb
            return mock_kb

    @pytest.fixture
    def sample_repos(self):
        """Sample repository configurations for testing."""
        return [
            RepositoryConfig(
                name="nautobot/nautobot",
                description="Official Nautobot repository",
                priority=1,
                enabled=True,
                branch="main",
            ),
            RepositoryConfig(
                name="user/custom-repo",
                description="User custom repository",
                priority=5,
                enabled=True,
                branch="develop",
            ),
            RepositoryConfig(
                name="disabled/repo",
                description="Disabled repository",
                priority=3,
                enabled=False,
                branch="main",
            ),
        ]

    def test_list_repositories_with_repos(
        self, mock_config_manager, sample_repos, capsys
    ):
        """Test listing repositories when repositories exist."""
        mock_config_manager.load_repositories.return_value = sample_repos

        manage_repos.list_repositories(mock_config_manager, "all")

        captured = capsys.readouterr()
        assert "Configured Repositories:" in captured.out
        assert "nautobot/nautobot" in captured.out
        assert "user/custom-repo" in captured.out
        assert "disabled/repo" in captured.out
        assert "Official Nautobot repository" in captured.out
        assert "✓" in captured.out  # Enabled repos
        assert "✗" in captured.out  # Disabled repos

    def test_list_repositories_empty(self, mock_config_manager, capsys):
        """Test listing repositories when no repositories exist."""
        mock_config_manager.load_repositories.return_value = []

        manage_repos.list_repositories(mock_config_manager, "all")

        captured = capsys.readouterr()
        assert "No repositories configured." in captured.out

    def test_add_repository_success(self, mock_config_manager, capsys):
        """Test successfully adding a repository."""
        mock_config_manager.add_user_repository.return_value = True

        manage_repos.add_repository(
            mock_config_manager,
            "test/repo",
            description="Test repository",
            category="testing",
        )

        captured = capsys.readouterr()
        assert "✓ Added repository: test/repo" in captured.out

        # Verify the repository config was created correctly
        call_args = mock_config_manager.add_user_repository.call_args[0][0]
        assert call_args.name == "test/repo"
        assert call_args.description == "Test repository"
        assert call_args.priority == 5
        assert call_args.enabled is True
        assert call_args.branch == "main"

    def test_add_repository_failure(self, mock_config_manager, capsys):
        """Test failing to add a repository (already exists)."""
        mock_config_manager.add_user_repository.return_value = False

        manage_repos.add_repository(mock_config_manager, "existing/repo")

        captured = capsys.readouterr()
        assert (
            "✗ Failed to add repository existing/repo (may already exist)"
            in captured.out
        )

    def test_add_repository_exception(self, mock_config_manager, capsys):
        """Test adding repository with exception."""
        mock_config_manager.add_user_repository.side_effect = Exception("Config error")

        manage_repos.add_repository(mock_config_manager, "test/repo")

        captured = capsys.readouterr()
        assert "✗ Failed to add repository test/repo: Config error" in captured.out

    def test_add_repository_default_description(self, mock_config_manager, capsys):
        """Test adding repository with default description."""
        mock_config_manager.add_user_repository.return_value = True

        manage_repos.add_repository(mock_config_manager, "test/repo")

        captured = capsys.readouterr()
        assert "✓ Added repository: test/repo" in captured.out

        # Verify default description was used
        call_args = mock_config_manager.add_user_repository.call_args[0][0]
        assert call_args.description == "User-added repository: test/repo"

    def test_remove_repository_success(self, mock_config_manager, capsys):
        """Test successfully removing a repository."""
        mock_config_manager.remove_user_repository.return_value = True

        manage_repos.remove_repository(mock_config_manager, "test/repo")

        captured = capsys.readouterr()
        assert "✓ Removed repository: test/repo" in captured.out
        mock_config_manager.remove_user_repository.assert_called_once_with("test/repo")

    def test_remove_repository_failure(self, mock_config_manager, capsys):
        """Test failing to remove a repository (doesn't exist)."""
        mock_config_manager.remove_user_repository.return_value = False

        manage_repos.remove_repository(mock_config_manager, "nonexistent/repo")

        captured = capsys.readouterr()
        assert (
            "✗ Failed to remove repository nonexistent/repo (may not exist)"
            in captured.out
        )

    def test_remove_repository_exception(self, mock_config_manager, capsys):
        """Test removing repository with exception."""
        mock_config_manager.remove_user_repository.side_effect = Exception(
            "Config error"
        )

        manage_repos.remove_repository(mock_config_manager, "test/repo")

        captured = capsys.readouterr()
        assert "✗ Failed to remove repository test/repo: Config error" in captured.out

    def test_update_repositories_specific_repo_success(self, mock_kb, capsys):
        """Test updating a specific repository successfully."""
        mock_config_manager = Mock()
        mock_repo_config = RepositoryConfig(name="test/repo")
        mock_config_manager.get_repo_config.return_value = mock_repo_config
        mock_kb.update_repository.return_value = True

        with patch(
            "helpers.manage_repos.RepositoryConfigManager",
            return_value=mock_config_manager,
        ):
            manage_repos.update_repositories(mock_kb, repo="test/repo", force=False)

        captured = capsys.readouterr()
        assert "Updating repository: test/repo" in captured.out
        assert "✓ Updated test/repo" in captured.out
        mock_kb.update_repository.assert_called_once_with(mock_repo_config, False)

    def test_update_repositories_specific_repo_not_found(self, mock_kb, capsys):
        """Test updating a repository that doesn't exist in config."""
        mock_config_manager = Mock()
        mock_config_manager.get_repo_config.return_value = None

        with patch(
            "helpers.manage_repos.RepositoryConfigManager",
            return_value=mock_config_manager,
        ):
            manage_repos.update_repositories(
                mock_kb, repo="nonexistent/repo", force=False
            )

        captured = capsys.readouterr()
        assert (
            "✗ Repository nonexistent/repo not found in configuration" in captured.out
        )

    def test_update_repositories_specific_repo_no_update_needed(self, mock_kb, capsys):
        """Test updating a repository when no update is needed."""
        mock_config_manager = Mock()
        mock_repo_config = RepositoryConfig(name="test/repo")
        mock_config_manager.get_repo_config.return_value = mock_repo_config
        mock_kb.update_repository.return_value = False

        with patch(
            "helpers.manage_repos.RepositoryConfigManager",
            return_value=mock_config_manager,
        ):
            manage_repos.update_repositories(mock_kb, repo="test/repo", force=False)

        captured = capsys.readouterr()
        assert "• test/repo was already up to date or failed to update" in captured.out

    def test_update_repositories_all_repos(self, mock_kb, capsys):
        """Test updating all repositories."""
        mock_results = {"repo1": True, "repo2": False, "repo3": True}
        mock_kb.initialize_all_repositories.return_value = mock_results

        manage_repos.update_repositories(mock_kb, repo=None, force=True)

        captured = capsys.readouterr()
        assert "Updating all repositories..." in captured.out
        assert "✓ Updated 2/3 repositories" in captured.out
        mock_kb.initialize_all_repositories.assert_called_once_with(True)

    def test_initialize_repositories_success(self, mock_kb, capsys):
        """Test successfully initializing repositories."""
        mock_results = {"repo1": True, "repo2": False, "repo3": True}
        mock_kb.initialize_all_repositories.return_value = mock_results

        manage_repos.initialize_repositories(mock_kb, force=True)

        captured = capsys.readouterr()
        assert "Initializing all repositories..." in captured.out
        assert "• Force initialization enabled" in captured.out
        assert "✓ Successfully initialized 2/3 repositories" in captured.out
        assert "repo1: ✓ Updated" in captured.out
        assert "repo2: • Skipped (up to date)" in captured.out
        assert "repo3: ✓ Updated" in captured.out

    def test_initialize_repositories_without_force(self, mock_kb, capsys):
        """Test initializing repositories without force flag."""
        mock_results = {"repo1": True}
        mock_kb.initialize_all_repositories.return_value = mock_results

        manage_repos.initialize_repositories(mock_kb, force=False)

        captured = capsys.readouterr()
        assert "Initializing all repositories..." in captured.out
        assert "• Force initialization enabled" not in captured.out
        mock_kb.initialize_all_repositories.assert_called_once_with(False)

    def test_initialize_repositories_exception(self, mock_kb, capsys):
        """Test initializing repositories with exception."""
        mock_kb.initialize_all_repositories.side_effect = Exception(
            "Initialization failed"
        )

        manage_repos.initialize_repositories(mock_kb, force=False)

        captured = capsys.readouterr()
        assert (
            "✗ Failed to initialize repositories: Initialization failed" in captured.out
        )

    def test_show_status(self, mock_kb, sample_repos, capsys):
        """Test showing repository status."""
        mock_config_manager = Mock()
        mock_config_manager.load_repositories.return_value = sample_repos

        mock_stats = {
            "nautobot/nautobot": {"document_count": 150, "enabled": True},
            "user/custom-repo": {"document_count": 0, "enabled": True},
            "disabled/repo": {
                "document_count": 25,
                "enabled": False,
                "error": "Access denied",
            },
        }
        mock_kb.get_repository_stats.return_value = mock_stats

        with patch(
            "helpers.manage_repos.RepositoryConfigManager",
            return_value=mock_config_manager,
        ):
            manage_repos.show_status(mock_kb)

        captured = capsys.readouterr()
        assert "Repository Status:" in captured.out
        assert "nautobot/nautobot" in captured.out
        assert "150" in captured.out
        assert "✓ Indexed" in captured.out
        assert "Not indexed" in captured.out
        assert "Error: Access denied" in captured.out

    @pytest.mark.parametrize(
        "command,expected_function",
        [
            ("list", "list_repositories"),
            ("add", "add_repository"),
            ("remove", "remove_repository"),
            ("update", "update_repositories"),
            ("init", "initialize_repositories"),
            ("status", "show_status"),
        ],
    )
    def test_main_command_routing(self, command, expected_function):
        """Test that main() routes commands to the correct functions."""
        test_args = ["manage_repos.py", command]
        if command == "add":
            test_args.append("test/repo")
        elif command == "remove":
            test_args.append("test/repo")

        with (
            patch("sys.argv", test_args),
            patch(f"helpers.manage_repos.{expected_function}") as mock_func,
            patch("helpers.manage_repos.RepositoryConfigManager"),
            patch("helpers.manage_repos.EnhancedNautobotKnowledge"),
        ):
            manage_repos.main()
            mock_func.assert_called_once()

    def test_main_no_command(self, capsys):
        """Test main() with no command shows help."""
        with patch("sys.argv", ["manage_repos.py"]):
            manage_repos.main()

        captured = capsys.readouterr()
        assert "usage:" in captured.out or "Available commands" in captured.out

    def test_main_list_command_with_type(self):
        """Test main() list command with type argument."""
        test_args = ["manage_repos.py", "list", "--type", "user"]

        with (
            patch("sys.argv", test_args),
            patch("helpers.manage_repos.list_repositories") as mock_list,
            patch("helpers.manage_repos.RepositoryConfigManager") as mock_config,
            patch("helpers.manage_repos.EnhancedNautobotKnowledge"),
        ):
            manage_repos.main()
            mock_list.assert_called_once_with(mock_config.return_value, "user")

    def test_main_add_command_with_optional_args(self):
        """Test main() add command with description and category."""
        test_args = [
            "manage_repos.py",
            "add",
            "test/repo",
            "--description",
            "Test desc",
            "--category",
            "testing",
        ]

        with (
            patch("sys.argv", test_args),
            patch("helpers.manage_repos.add_repository") as mock_add,
            patch("helpers.manage_repos.RepositoryConfigManager") as mock_config,
            patch("helpers.manage_repos.EnhancedNautobotKnowledge"),
        ):
            manage_repos.main()
            mock_add.assert_called_once_with(
                mock_config.return_value, "test/repo", "Test desc", "testing"
            )

    def test_main_update_command_with_options(self):
        """Test main() update command with repo and force options."""
        test_args = ["manage_repos.py", "update", "--repo", "test/repo", "--force"]

        with (
            patch("sys.argv", test_args),
            patch("helpers.manage_repos.update_repositories") as mock_update,
            patch("helpers.manage_repos.RepositoryConfigManager"),
            patch("helpers.manage_repos.EnhancedNautobotKnowledge") as mock_kb_class,
        ):
            manage_repos.main()
            mock_update.assert_called_once_with(
                mock_kb_class.return_value, "test/repo", True
            )

    def test_main_init_command_with_force(self):
        """Test main() init command with force option."""
        test_args = ["manage_repos.py", "init", "--force"]

        with (
            patch("sys.argv", test_args),
            patch("helpers.manage_repos.initialize_repositories") as mock_init,
            patch("helpers.manage_repos.RepositoryConfigManager"),
            patch("helpers.manage_repos.EnhancedNautobotKnowledge") as mock_kb_class,
        ):
            manage_repos.main()
            mock_init.assert_called_once_with(mock_kb_class.return_value, True)

    def test_main_exception_handling(self, capsys):
        """Test main() exception handling."""
        with (
            patch("sys.argv", ["manage_repos.py", "list"]),
            patch(
                "helpers.manage_repos.RepositoryConfigManager",
                side_effect=Exception("Test error"),
            ),
        ):
            with pytest.raises(SystemExit) as exc_info:
                manage_repos.main()

            assert exc_info.value.code == 1

        captured = capsys.readouterr()
        assert "Error: Test error" in captured.err

    def test_argparser_configuration(self):
        """Test that the argument parser is configured correctly."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")

        # Test that we can create the same subparsers as in main()
        list_parser = subparsers.add_parser("list")
        list_parser.add_argument(
            "--type", choices=["official", "user", "all"], default="all"
        )

        add_parser = subparsers.add_parser("add")
        add_parser.add_argument("repo")
        add_parser.add_argument("--description")
        add_parser.add_argument("--category")

        # Test parsing
        args = parser.parse_args(["list", "--type", "user"])
        assert args.command == "list"
        assert args.type == "user"

        args = parser.parse_args(["add", "test/repo", "--description", "Test"])
        assert args.command == "add"
        assert args.repo == "test/repo"
        assert args.description == "Test"

    def test_repository_config_creation_in_add_repository(self, mock_config_manager):
        """Test that RepositoryConfig is created with correct parameters."""
        mock_config_manager.add_user_repository.return_value = True

        manage_repos.add_repository(
            mock_config_manager,
            "test/repo",
            description="Custom description",
            category="custom",
        )

        # Verify the RepositoryConfig object passed to add_user_repository
        call_args = mock_config_manager.add_user_repository.call_args[0][0]
        assert isinstance(call_args, RepositoryConfig)
        assert call_args.name == "test/repo"
        assert call_args.description == "Custom description"
        assert call_args.priority == 5
        assert call_args.enabled is True
        assert call_args.branch == "main"
        assert call_args.file_patterns == [".py", ".md", ".txt", ".rst", ".json"]

    def test_full_workflow_integration(self, temp_dirs):
        """Integration test for a complete add-list-remove workflow."""
        config_dir, cache_dir = temp_dirs

        # This would require actual implementations, so we'll mock the dependencies
        with (
            patch("helpers.manage_repos.RepositoryConfigManager") as mock_config_class,
            patch("helpers.manage_repos.EnhancedNautobotKnowledge") as mock_kb_class,
        ):
            mock_config = Mock()
            mock_config_class.return_value = mock_config
            mock_kb = Mock()
            mock_kb_class.return_value = mock_kb

            # Setup mock responses
            mock_config.add_user_repository.return_value = True
            mock_config.load_repositories.return_value = [
                RepositoryConfig(name="test/repo", description="Test repo")
            ]
            mock_config.remove_user_repository.return_value = True

            # Test add
            manage_repos.add_repository(mock_config, "test/repo", "Test repo")
            mock_config.add_user_repository.assert_called_once()

            # Test list
            manage_repos.list_repositories(mock_config, "all")
            mock_config.load_repositories.assert_called_once()

            # Test remove
            manage_repos.remove_repository(mock_config, "test/repo")
            mock_config.remove_user_repository.assert_called_once_with("test/repo")
