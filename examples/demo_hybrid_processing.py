#!/usr/bin/env python3
"""
Simple verification test to demonstrate hybrid processing on specific file types.
"""


def test_file_type_processing():
    """Test that demonstrates hybrid processing working on different file types."""
    try:
        print("🔍 HYBRID PROCESSING DEMONSTRATION")
        print("=" * 50)

        from nb_kb_enhanced import EnhancedNautobotKnowledge

        kb = EnhancedNautobotKnowledge()
        print("✓ Knowledge base initialized")

        # Test cases targeting specific content
        demos = [
            {
                "name": "🐍 PYTHON CODE PROCESSING",
                "query": "class Device def create python model",
                "target": "Python files with classes and functions",
                "look_for": ["py"],
            },
            {
                "name": "📖 DOCUMENTATION PROCESSING",
                "query": "installation setup guide tutorial markdown",
                "target": "Markdown documentation files",
                "look_for": ["md", "rst"],
            },
            {
                "name": "⚙️ CONFIGURATION PROCESSING",
                "query": "settings config yaml json configuration",
                "target": "Configuration files",
                "look_for": ["yaml", "yml", "json", "toml"],
            },
        ]

        total_old_size = 0
        total_new_size = 0

        for demo in demos:
            print(f"\n{demo['name']}")
            print("-" * 40)
            print(f"Target: {demo['target']}")
            print(f"Query: '{demo['query']}'")

            # Get hybrid results
            results = kb.search_optimized_for_llm(
                query=demo["query"], n_results=2, max_content_length=350
            )

            if results:
                print(f"Found {len(results)} results:")

                for i, result in enumerate(results, 1):
                    file_type = result["source"]["type"]
                    method = result["processing"]["method"]
                    original = result["processing"]["original_size"]
                    final = len(result["content"])
                    ratio = result["processing"]["compression_ratio"]

                    print(f"\n  📄 Result {i}: {result['source']['file']}")
                    print(f"      Type: .{file_type} | Method: {method}")
                    print(f"      Size: {original} → {final} chars (ratio: {ratio})")
                    print(f"      Relevance: {result['relevance_score']}")

                    # Show processing effectiveness
                    if file_type in demo["look_for"]:
                        print(f"      ✅ Found target file type: .{file_type}")
                    else:
                        print(f"      ℹ️  Found different type: .{file_type}")

                    if method == "structured_code" and file_type == "py":
                        print(
                            "      🎯 Perfect: Code file processed with structure extraction"
                        )
                    elif method in [
                        "semantic_selection",
                        "structured_documentation",
                    ] and file_type in ["md", "rst"]:
                        print(
                            "      🎯 Perfect: Documentation processed with semantic selection"
                        )
                    elif method == "structured_config" and file_type in [
                        "yaml",
                        "yml",
                        "json",
                        "toml",
                    ]:
                        print(
                            "      🎯 Perfect: Config processed with structure extraction"
                        )
                    elif method == "full_document":
                        print("      📋 Short document: No processing needed")
                    else:
                        print(f"      🔄 Fallback: {method} used")

                    # Show content sample
                    content_preview = result["content"][:120].replace("\n", " ").strip()
                    print(f"      📝 Content: {content_preview}...")

                # Compare with old method
                old_results = kb.search(demo["query"], n_results=2)
                if old_results:
                    old_size = sum(len(r.get("document", "")) for r in old_results)
                    new_size = sum(len(r["content"]) for r in results)

                    total_old_size += old_size
                    total_new_size += new_size

                    reduction = (
                        ((old_size - new_size) / old_size * 100) if old_size > 0 else 0
                    )
                    print(
                        f"\n  📊 Efficiency: {old_size} → {new_size} chars ({reduction:.1f}% reduction)"
                    )
            else:
                print("  ⚠️  No results found")

        # Overall summary
        print(f"\n{'=' * 50}")
        print("🏆 OVERALL PERFORMANCE SUMMARY")
        print(f"{'=' * 50}")

        if total_old_size > 0:
            overall_reduction = (total_old_size - total_new_size) / total_old_size * 100
            token_savings = (
                total_old_size - total_new_size
            ) // 4  # Rough token estimate

            print("📈 Total content processed:")
            print(f"   Before: {total_old_size:,} characters")
            print(f"   After:  {total_new_size:,} characters")
            print(f"   Reduction: {overall_reduction:.1f}%")
            print(f"   Estimated token savings: ~{token_savings:,} tokens")

            if overall_reduction > 50:
                print("🎉 Excellent compression! Over 50% reduction achieved.")
            elif overall_reduction > 25:
                print("✅ Good compression! 25%+ reduction achieved.")
            else:
                print("📝 Moderate compression. Focus on highly relevant content.")

        print("\n✅ Hybrid processing demonstration complete!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_file_type_processing()
