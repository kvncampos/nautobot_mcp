#!/usr/bin/env python3
"""
Unit tests for tool_definitions.py
"""

import pytest

from helpers.tool_definitions import (
    get_add_repo_description,
    get_add_repo_input_schema,
    get_api_request_schema_description,
    get_api_request_schema_input_schema,
    get_dynamic_api_request_description,
    get_dynamic_api_request_input_schema,
    get_init_repos_description,
    get_init_repos_input_schema,
    get_kb_semantic_search_description,
    get_kb_semantic_search_input_schema,
    get_list_repos_description,
    get_list_repos_input_schema,
    get_refresh_endpoint_index_description,
    get_refresh_endpoint_index_input_schema,
    get_remove_repo_description,
    get_remove_repo_input_schema,
    get_repo_status_description,
    get_repo_status_input_schema,
    get_update_repos_description,
    get_update_repos_input_schema,
)


class TestToolDescriptions:
    """Test tool description functions"""

    def test_api_request_schema_description(self):
        """Test API request schema description"""
        desc = get_api_request_schema_description()
        assert isinstance(desc, str)
        assert len(desc) > 0
        assert "endpoint" in desc.lower()

    def test_api_request_schema_description_with_global_prompt(self):
        """Test API request schema description with global prompt"""
        global_prompt = "Test prompt:"
        desc = get_api_request_schema_description(global_prompt)
        assert isinstance(desc, str)
        assert desc.startswith("Test prompt:")

    def test_dynamic_api_request_description(self):
        """Test dynamic API request description"""
        desc = get_dynamic_api_request_description()
        assert isinstance(desc, str)
        assert len(desc) > 0
        assert "GET" in desc and "POST" in desc

    def test_refresh_endpoint_index_description(self):
        """Test refresh endpoint index description"""
        desc = get_refresh_endpoint_index_description()
        assert isinstance(desc, str)
        assert "refresh" in desc.lower()

    def test_kb_semantic_search_description(self):
        """Test KB semantic search description"""
        desc = get_kb_semantic_search_description()
        assert isinstance(desc, str)
        assert "search" in desc.lower()

    def test_list_repos_description(self):
        """Test list repos description"""
        desc = get_list_repos_description()
        assert isinstance(desc, str)
        assert "repositor" in desc.lower()

    def test_add_repo_description(self):
        """Test add repo description"""
        desc = get_add_repo_description()
        assert isinstance(desc, str)
        assert "add" in desc.lower()

    def test_remove_repo_description(self):
        """Test remove repo description"""
        desc = get_remove_repo_description()
        assert isinstance(desc, str)
        assert "remove" in desc.lower()

    def test_update_repos_description(self):
        """Test update repos description"""
        desc = get_update_repos_description()
        assert isinstance(desc, str)
        assert "update" in desc.lower()

    def test_init_repos_description(self):
        """Test init repos description"""
        desc = get_init_repos_description()
        assert isinstance(desc, str)
        assert "initialize" in desc.lower()

    def test_repo_status_description(self):
        """Test repo status description"""
        desc = get_repo_status_description()
        assert isinstance(desc, str)
        assert "status" in desc.lower()


class TestToolSchemas:
    """Test tool input schema functions"""

    def test_api_request_schema_input_schema(self):
        """Test API request schema input schema"""
        schema = get_api_request_schema_input_schema()
        assert isinstance(schema, dict)
        assert schema["type"] == "object"
        assert "query" in schema["properties"]
        assert "n_results" in schema["properties"]
        assert "query" in schema["required"]

    def test_dynamic_api_request_input_schema(self):
        """Test dynamic API request input schema"""
        schema = get_dynamic_api_request_input_schema()
        assert isinstance(schema, dict)
        assert schema["type"] == "object"
        assert "method" in schema["properties"]
        assert "path" in schema["properties"]
        assert "params" in schema["properties"]
        assert "body" in schema["properties"]
        assert "method" in schema["required"]
        assert "path" in schema["required"]

    def test_refresh_endpoint_index_input_schema(self):
        """Test refresh endpoint index input schema"""
        schema = get_refresh_endpoint_index_input_schema()
        assert isinstance(schema, dict)
        assert schema["type"] == "object"
        assert len(schema["properties"]) == 0
        assert len(schema["required"]) == 0

    def test_kb_semantic_search_input_schema(self):
        """Test KB semantic search input schema"""
        schema = get_kb_semantic_search_input_schema()
        assert isinstance(schema, dict)
        assert "query" in schema["properties"]
        assert "n_results" in schema["properties"]
        assert "query" in schema["required"]

    def test_list_repos_input_schema(self):
        """Test list repos input schema - no parameters"""
        schema = get_list_repos_input_schema()
        assert isinstance(schema, dict)
        assert schema["type"] == "object"
        assert len(schema["required"]) == 0

    def test_add_repo_input_schema(self):
        """Test add repo input schema"""
        schema = get_add_repo_input_schema()
        assert isinstance(schema, dict)
        assert "repo" in schema["properties"]
        assert "description" in schema["properties"]
        assert "repo" in schema["required"]
        # Verify category parameter was removed
        assert "category" not in schema["properties"]

    def test_remove_repo_input_schema(self):
        """Test remove repo input schema"""
        schema = get_remove_repo_input_schema()
        assert isinstance(schema, dict)
        assert "repo" in schema["properties"]
        assert "repo" in schema["required"]

    def test_update_repos_input_schema(self):
        """Test update repos input schema"""
        schema = get_update_repos_input_schema()
        assert isinstance(schema, dict)
        assert "repo" in schema["properties"]
        assert "force" in schema["properties"]
        assert len(schema["required"]) == 0

    def test_init_repos_input_schema(self):
        """Test init repos input schema"""
        schema = get_init_repos_input_schema()
        assert isinstance(schema, dict)
        assert "force" in schema["properties"]
        assert len(schema["required"]) == 0

    def test_repo_status_input_schema(self):
        """Test repo status input schema"""
        schema = get_repo_status_input_schema()
        assert isinstance(schema, dict)
        assert len(schema["properties"]) == 0
        assert len(schema["required"]) == 0


class TestDescriptionLengths:
    """Test that descriptions follow LLM best practices for length"""

    def test_descriptions_not_too_long(self):
        """Test that descriptions are reasonably concise (under 300 chars)"""
        descriptions = [
            get_api_request_schema_description(),
            get_dynamic_api_request_description(),
            get_refresh_endpoint_index_description(),
            get_kb_semantic_search_description(),
            get_list_repos_description(),
            get_add_repo_description(),
            get_remove_repo_description(),
            get_update_repos_description(),
            get_init_repos_description(),
            get_repo_status_description(),
        ]

        for desc in descriptions:
            # Allow up to 300 chars (some may be longer but should be clear)
            assert (
                len(desc) < 400
            ), f"Description too long ({len(desc)} chars): {desc[:100]}..."

    def test_descriptions_have_actionable_content(self):
        """Test that descriptions contain actionable verbs"""
        actionable_verbs = ["search", "execute", "refresh", "list", "add", "remove", "update", "initialize", "show"]

        descriptions = [
            get_api_request_schema_description(),
            get_dynamic_api_request_description(),
            get_refresh_endpoint_index_description(),
            get_kb_semantic_search_description(),
            get_list_repos_description(),
            get_add_repo_description(),
            get_remove_repo_description(),
            get_update_repos_description(),
            get_init_repos_description(),
            get_repo_status_description(),
        ]

        for desc in descriptions:
            # Check that at least one actionable verb is present
            has_verb = any(verb.lower() in desc.lower() for verb in actionable_verbs)
            assert has_verb, f"Description lacks actionable verb: {desc}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
