#!/usr/bin/env python3
"""
Test script for the HTTP MCP server
This demonstrates how to interact with the FastMCP HTTP endpoints
"""

import asyncio
import json

import httpx
import pytest


@pytest.mark.asyncio
async def test_mcp_server():
    """Test the HTTP MCP server endpoints"""
    base_url = "http://localhost:8000"

    async with httpx.AsyncClient() as client:
        # Test server info
        print("🔍 Testing server info...")
        try:
            response = await client.get(f"{base_url}/")
            print(f"Server response: {response.status_code}")
            if response.status_code == 200:
                print(f"Server info: {response.json()}")
        except Exception as e:
            print(f"Error connecting to server: {e}")
            return

        # Test tools list
        print("\n🛠️  Testing tools list...")
        try:
            response = await client.post(f"{base_url}/tools")
            if response.status_code == 200:
                tools = response.json()
                print(f"Available tools: {len(tools)} tools found")
                for tool in tools:
                    print(
                        f"  - {tool.get('name', 'Unknown')}: {tool.get('description', 'No description')[:100]}..."
                    )
            else:
                print(f"Error getting tools: {response.status_code}")
        except Exception as e:
            print(f"Error getting tools: {e}")

        # Test API schema search
        print("\n🔍 Testing API schema search...")
        try:
            payload = {
                "name": "mcp_nautobot_openapi_api_request_schema",
                "arguments": {"query": "get devices", "n_results": 3},
            }
            response = await client.post(f"{base_url}/tools/call", json=payload)
            if response.status_code == 200:
                result = response.json()
                print("API schema search successful!")
                # Parse the response content
                content = json.loads(result.get("content", [{}])[0].get("text", "{}"))
                endpoints = content.get("matching_endpoints", [])
                print(f"Found {len(endpoints)} matching endpoints")
                for endpoint in endpoints[:2]:  # Show first 2
                    print(
                        f"  - {endpoint.get('method', 'GET')} {endpoint.get('path', 'Unknown path')}"
                    )
            else:
                print(f"Error in API schema search: {response.status_code}")
                print(f"Response: {response.text}")
        except Exception as e:
            print(f"Error testing API schema search: {e}")

        # Test knowledge base search
        print("\n📚 Testing knowledge base search...")
        try:
            payload = {
                "name": "mcp_nautobot_kb_semantic_search",
                "arguments": {"query": "Nautobot Job example", "n_results": 2},
            }
            response = await client.post(f"{base_url}/tools/call", json=payload)
            if response.status_code == 200:
                result = response.json()
                print("Knowledge base search successful!")
                # Parse the response content
                content = json.loads(result.get("content", [{}])[0].get("text", "{}"))
                results = content.get("results", [])
                print(f"Found {len(results)} knowledge base results")
                for result_item in results[:1]:  # Show first result
                    print(f"  - Source: {result_item.get('source', 'Unknown')}")
                    print(
                        f"    Content: {result_item.get('content', 'No content')[:100]}..."
                    )
            else:
                print(f"Error in knowledge base search: {response.status_code}")
                print(f"Response: {response.text}")
        except Exception as e:
            print(f"Error testing knowledge base search: {e}")


if __name__ == "__main__":
    print("🚀 Starting MCP HTTP Server Test")
    print("Make sure to run 'python server_http.py' in another terminal first!")
    print("=" * 60)

    asyncio.run(test_mcp_server())

    print("\n" + "=" * 60)
    print("✅ Test completed!")
    print("\nTo interact with the server manually:")
    print("  - Server runs at: http://localhost:8000")
    print("  - API docs at: http://localhost:8000/docs")
    print("  - Tools list: POST http://localhost:8000/tools")
    print("  - Call tool: POST http://localhost:8000/tools/call")
