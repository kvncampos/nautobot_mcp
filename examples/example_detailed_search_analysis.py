#!/usr/bin/env python3
"""
Detailed analysis of content quality between search methods.
"""

import time

from nb_kb_enhanced import EnhancedNautobotKnowledge


def detailed_analysis():
    """Perform detailed analysis of the two search methods."""

    kb = EnhancedNautobotKnowledge()

    test_queries = [
        "How to create a custom Nautobot job",
        "Nautobot database migrations",
        "API authentication in Nautobot",
        "Creating custom models in Nautobot",
    ]

    results_summary = {
        "optimized": {"total_time": 0, "avg_compression": 0, "processing_methods": {}},
        "simple": {"total_time": 0, "avg_length": 0},
    }

    for i, query in enumerate(test_queries):
        print(f"\n{'=' * 60}")
        print(f"Query {i + 1}: {query}")
        print("=" * 60)

        # Test optimized method
        start_time = time.time()
        optimized = kb.search_optimized_for_llm(
            query=query, n_results=2, max_content_length=400
        )
        opt_time = time.time() - start_time
        results_summary["optimized"]["total_time"] += opt_time

        # Test simple method
        start_time = time.time()
        simple = kb.search_for_llm(query=query, n_results=2)
        simple_time = time.time() - start_time
        results_summary["simple"]["total_time"] += simple_time

        if optimized and simple:
            print(f"\nOptimized method (time: {opt_time:.3f}s):")
            for j, result in enumerate(optimized):
                processing = result.get("processing", {})
                method = processing.get("method", "unknown")
                compression = processing.get("compression_ratio", 0)

                # Track processing methods
                if method in results_summary["optimized"]["processing_methods"]:
                    results_summary["optimized"]["processing_methods"][method] += 1
                else:
                    results_summary["optimized"]["processing_methods"][method] = 1

                results_summary["optimized"]["avg_compression"] += compression

                print(
                    f"  [{j + 1}] Method: {method} | Compression: {compression:.3f} | Length: {len(result.get('content', ''))}"
                )
                print(f"      Content: {result.get('content', '')[:120]}...")

            print(f"\nSimple method (time: {simple_time:.3f}s):")
            for j, result in enumerate(simple):
                content_len = len(result.get("content", ""))
                results_summary["simple"]["avg_length"] += content_len
                print(f"  [{j + 1}] Length: {content_len}")
                print(f"      Content: {result.get('content', '')[:120]}...")

    # Calculate averages
    num_queries = len(test_queries)
    total_results = num_queries * 2  # 2 results per query

    print(f"\n{'=' * 60}")
    print("SUMMARY ANALYSIS")
    print("=" * 60)

    print("\nPerformance:")
    print(f"  Optimized total time: {results_summary['optimized']['total_time']:.3f}s")
    print(f"  Simple total time:    {results_summary['simple']['total_time']:.3f}s")
    print(
        f"  Speed difference:     {results_summary['optimized']['total_time'] / results_summary['simple']['total_time']:.1f}x slower"
    )

    print("\nContent Processing:")
    avg_compression = results_summary["optimized"]["avg_compression"] / total_results
    avg_length = results_summary["simple"]["avg_length"] / total_results
    print(f"  Average compression ratio: {avg_compression:.3f}")
    print(f"  Average simple length:     {avg_length:.0f} chars")

    print("\nProcessing Methods Used:")
    for method, count in results_summary["optimized"]["processing_methods"].items():
        percentage = (count / total_results) * 100
        print(f"  {method}: {count}/{total_results} ({percentage:.1f}%)")

    print("\nKey Benefits of search_optimized_for_llm:")
    print("  1. Content is intelligently filtered and relevant to query")
    print(f"  2. Compression ratio of {avg_compression:.1f} means more focused content")
    print("  3. Different processing strategies for different file types")
    print("  4. Semantic selection ensures query-relevant content is preserved")

    print("\nKey Benefits of search_for_llm:")
    print(
        f"  1. Much faster execution ({results_summary['simple']['total_time']:.3f}s vs {results_summary['optimized']['total_time']:.3f}s)"
    )
    print("  2. Simpler, more predictable output")
    print("  3. Full content preservation (just truncated if too long)")


if __name__ == "__main__":
    detailed_analysis()
