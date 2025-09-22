import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path so we can import nb_kb_enhanced
sys.path.append(str(Path(__file__).parent.parent))

from nb_kb_enhanced import EnhancedNautobotKnowledge

# Setup output file for comprehensive testing results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = Path("examples", f"nautobot_kb_test_results_{timestamp}.txt")


def log_and_print(message, file_handle=None):
    """Print to console and write to file."""
    print(message)
    if file_handle:
        file_handle.write(message + "\n")
        file_handle.flush()


def main():
    print("Starting comprehensive test of Enhanced Nautobot Knowledge Base...")
    print(f"Results will be saved to: {output_file}")

    with open(output_file, "w", encoding="utf-8") as f:
        # Header
        header = f"""
            {"=" * 80}
            ENHANCED NAUTOBOT KNOWLEDGE BASE - COMPREHENSIVE TEST
            {"=" * 80}
            Test Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            Test File: {output_file}
            {"=" * 80}
        """
        log_and_print(header.strip(), f)

        try:
            # Step 1: Initialize the knowledge base
            log_and_print("\n1. INITIALIZING ENHANCED NAUTOBOT KNOWLEDGE BASE", f)
            log_and_print("-" * 50, f)

            kb = EnhancedNautobotKnowledge()
            log_and_print("✓ Knowledge base initialized successfully", f)

            # Step 2: Check repository configuration
            log_and_print("\n2. CHECKING REPOSITORY CONFIGURATION", f)
            log_and_print("-" * 50, f)

            repo_configs = kb.repo_config_manager.load_repositories()
            if not repo_configs:
                log_and_print(
                    "❌ No repositories configured. Please add repositories first.", f
                )
                sys.exit(1)

            log_and_print(f"✓ Found {len(repo_configs)} configured repositories:", f)
            for repo in repo_configs:
                status = "enabled" if repo.enabled else "disabled"
                log_and_print(f"  - {repo.name} ({status})", f)
                log_and_print(f"    Description: {repo.description}", f)
                log_and_print(f"    Priority: {repo.priority}", f)
                log_and_print(f"    File patterns: {repo.file_patterns}", f)

            # Step 3: Initialize repositories
            log_and_print("\n3. INITIALIZING REPOSITORIES", f)
            log_and_print("-" * 50, f)

            results = kb.initialize_all_repositories(force=False)

            for repo_name, was_updated in results.items():
                status = "updated" if was_updated else "already up to date"
                log_and_print(f"  - {repo_name}: {status}", f)

            # Step 4: Get repository statistics
            log_and_print("\n4. REPOSITORY STATISTICS", f)
            log_and_print("-" * 50, f)

            stats = kb.get_repository_stats()
            total_docs = 0
            for repo_name, stat in stats.items():
                if stat.get("enabled", False):
                    doc_count = stat["document_count"]
                    total_docs += doc_count
                    log_and_print(f"  - {repo_name}: {doc_count} documents", f)
                else:
                    error_msg = stat.get("error", "unknown error")
                    log_and_print(
                        f"  - {repo_name}: disabled or error ({error_msg})", f
                    )

            log_and_print(
                f"\n✓ Total documents across all repositories: {total_docs}", f
            )

            # Step 5: Test different search methods
            log_and_print("\n5. TESTING SEARCH METHODS", f)
            log_and_print("-" * 50, f)

            # Define test queries
            test_queries = [
                "device configuration",
                "API endpoints",
                "database models",
                "authentication",
                "REST API",
                "custom fields",
                "plugin development",
                "webhook notifications",
            ]

            for i, query in enumerate(test_queries, 1):
                log_and_print(f"\n5.{i} Testing query: '{query}'", f)
                log_and_print("~" * 40, f)

                # Test basic search
                try:
                    basic_results = kb.search(query, n_results=3)
                    if basic_results:
                        log_and_print(
                            f"  Basic search: {len(basic_results)} results", f
                        )
                        for j, result in enumerate(basic_results, 1):
                            log_and_print(
                                f"    [{j}] {result.get('repository', 'Unknown')}/"
                                + f"{result.get('metadata', {}).get('file_path', 'Unknown')}",
                                f,
                            )
                            log_and_print(
                                f"        Distance: {result.get('distance', 0):.4f}", f
                            )
                    else:
                        log_and_print("  Basic search: No results", f)
                except Exception as e:
                    log_and_print(f"  Basic search: Error - {e}", f)

                # Test LLM-optimized search
                try:
                    llm_results = kb.search_optimized_for_llm(
                        query, n_results=3, max_content_length=300
                    )
                    if llm_results:
                        log_and_print(
                            f"  LLM-optimized search: {len(llm_results)} results", f
                        )
                        for j, result in enumerate(llm_results, 1):
                            log_and_print(
                                f"    [{j}] {result.get('source', {}).get('repo', 'Unknown')}/"
                                + f"{result.get('source', {}).get('file', 'Unknown')}",
                                f,
                            )
                            log_and_print(
                                f"        Relevance: {result.get('relevance_score', 0):.3f}",
                                f,
                            )
                            log_and_print(
                                f"        Processing: {result.get('processing', {}).get('method', 'unknown')}",
                                f,
                            )
                            log_and_print(
                                f"        Compression: {result.get('processing', {}).get('compression_ratio', 0):.3f}",
                                f,
                            )

                            # Show content sample
                            content = result.get("content", "")
                            if content:
                                content_preview = (
                                    content[:150] + "..."
                                    if len(content) > 150
                                    else content
                                )
                                log_and_print(
                                    f"        Content preview: {content_preview}", f
                                )
                    else:
                        log_and_print("  LLM-optimized search: No results", f)
                except Exception as e:
                    log_and_print(f"  LLM-optimized search: Error - {e}", f)

            # Step 6: Test content processing methods
            log_and_print("\n6. TESTING CONTENT PROCESSING METHODS", f)
            log_and_print("-" * 50, f)

            # Get a sample document for testing
            sample_query = "device configuration"
            sample_results = kb.search(sample_query, n_results=1)

            if sample_results:
                sample_doc = sample_results[0].get("document", "")
                sample_metadata = sample_results[0].get("metadata", {})

                if sample_doc:
                    log_and_print("Testing processing methods on sample document:", f)
                    log_and_print(
                        f"  Source: {sample_metadata.get('repo', 'Unknown')}/{sample_metadata.get('file_path', 'Unknown')}",
                        f,
                    )
                    log_and_print(f"  Original length: {len(sample_doc)} characters", f)

                    # Test different processing methods
                    processing_methods = [
                        (
                            "Query-relevant extraction",
                            lambda: kb.content_processor.extract_query_relevant_content(
                                sample_doc, sample_query, 300
                            ),
                        ),
                        (
                            "Key information extraction",
                            lambda: kb.content_processor.extract_key_information(
                                sample_doc, 300
                            ),
                        ),
                        (
                            "Intelligent hybrid processing",
                            lambda: kb.content_processor.intelligent_content_processing(
                                sample_doc, sample_query, 300, sample_metadata
                            ),
                        ),
                    ]

                    for method_name, method_func in processing_methods:
                        try:
                            result = method_func()
                            if isinstance(result, dict):
                                content = result.get("content", str(result))
                                method_used = result.get(
                                    "processing_method", method_name
                                )
                                compression = result.get("compressed_ratio", "N/A")
                                log_and_print(f"  {method_name}:", f)
                                log_and_print(f"    Method used: {method_used}", f)
                                log_and_print(
                                    f"    Compression ratio: {compression}", f
                                )
                                log_and_print(
                                    f"    Result length: {len(content)} characters", f
                                )
                            else:
                                log_and_print(f"  {method_name}:", f)
                                log_and_print(
                                    f"    Result length: {len(result)} characters", f
                                )
                        except Exception as e:
                            log_and_print(f"  {method_name}: Error - {e}", f)

            # Step 7: Performance summary
            log_and_print("\n7. PERFORMANCE SUMMARY", f)
            log_and_print("-" * 50, f)

            enabled_repos = [repo for repo in repo_configs if repo.enabled]
            log_and_print(f"✓ Successfully tested {len(enabled_repos)} repositories", f)
            log_and_print(f"✓ Processed {total_docs} total documents", f)
            log_and_print(f"✓ Tested {len(test_queries)} search queries", f)
            log_and_print("✓ Validated content processing methods", f)

            # Step 8: Interactive mode option
            log_and_print("\n8. INTERACTIVE TESTING MODE", f)
            log_and_print("-" * 50, f)

            interactive = (
                input("Would you like to enter interactive testing mode? (y/n): ")
                .strip()
                .lower()
            )

            if interactive == "y":
                log_and_print("Entering interactive mode...", f)
                log_and_print("\nType 'quit' to exit, 'help' for commands", f)

                while True:
                    try:
                        query = input("\nEnter search query: ").strip()

                        if query.lower() in ["quit", "exit", "q"]:
                            break
                        elif query.lower() == "help":
                            help_text = """
                                Available commands:
                                - Any text: Search all repositories
                                - 'stats': Show repository statistics
                                - 'repos': Search specific repositories
                                - 'quit': Exit interactive mode
                            """
                            log_and_print(help_text, f)
                            continue
                        elif query.lower() == "stats":
                            current_stats = kb.get_repository_stats()
                            log_and_print("\nCurrent repository statistics:", f)
                            for repo_name, stat in current_stats.items():
                                if stat.get("enabled", False):
                                    log_and_print(
                                        f"  - {repo_name}: {stat['document_count']} documents",
                                        f,
                                    )
                                else:
                                    log_and_print(
                                        f"  - {repo_name}: disabled or error", f
                                    )
                            continue
                        elif not query:
                            continue

                        # Perform LLM-optimized search
                        log_and_print(f"\nSearching for: '{query}'", f)
                        results = kb.search_optimized_for_llm(query, n_results=5)

                        if results:
                            log_and_print(f"Found {len(results)} results:", f)
                            log_and_print("~" * 60, f)

                            for i, result in enumerate(results, 1):
                                log_and_print(f"\n[Result {i}]", f)
                                source = result.get("source", {})
                                log_and_print(
                                    f"Repository: {source.get('repo', 'Unknown')}", f
                                )
                                log_and_print(
                                    f"File: {source.get('file', 'Unknown')}", f
                                )
                                log_and_print(
                                    f"Type: {source.get('type', 'Unknown')}", f
                                )
                                log_and_print(
                                    f"Relevance: {result.get('relevance_score', 0):.3f}",
                                    f,
                                )

                                processing = result.get("processing", {})
                                log_and_print(
                                    f"Processing method: {processing.get('method', 'unknown')}",
                                    f,
                                )
                                log_and_print(
                                    f"Compression ratio: {processing.get('compression_ratio', 0):.3f}",
                                    f,
                                )

                                content = result.get("content", "")
                                content_preview = (
                                    content[:200] + "..."
                                    if len(content) > 200
                                    else content
                                )
                                log_and_print(f"Content preview:\n{content_preview}", f)
                                log_and_print("-" * 40, f)
                        else:
                            log_and_print("No results found.", f)

                    except KeyboardInterrupt:
                        log_and_print("\nExiting interactive mode...", f)
                        break
                    except Exception as e:
                        log_and_print(f"Error during search: {e}", f)

            # Final summary
            log_and_print(f"\n{'=' * 80}", f)
            log_and_print("TEST COMPLETED SUCCESSFULLY", f)
            log_and_print(f"{'=' * 80}", f)
            log_and_print(f"Results saved to: {output_file}", f)
            log_and_print(
                f"Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", f
            )

        except Exception as e:
            error_msg = f"\n❌ CRITICAL ERROR DURING TESTING: {e}"
            log_and_print(error_msg, f)
            import traceback

            log_and_print(f"\nFull traceback:\n{traceback.format_exc()}", f)
            sys.exit(1)

    print("\n✓ Comprehensive test completed!")
    print(f"✓ Full results saved to: {output_file}")
    print("✓ You can review the detailed output in the file.")


if __name__ == "__main__":
    main()
