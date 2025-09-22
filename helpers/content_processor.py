"""
Intelligent content processing for LLM-optimized search results.

This module provides multiple approaches for processing search results:
1. Query-focused extraction
2. Sentence-level semantic similarity
3. Key information extraction
4. Structured summarization
"""

import logging
import re
from typing import Any, Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class ContentProcessor:
    """Intelligent content processor for search results."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the content processor.

        Args:
            model_name: Name of the sentence transformer model to use
        """
        # Use local model cache only - no internet access
        import os
        from pathlib import Path

        # Set environment variables to force offline mode
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

        # Use the existing model cache location - go up one level from helpers to root, then to backend/models
        model_cache_dir = str(Path(__file__).parent.parent / "backend" / "models")

        # Initialize with local_files_only to prevent internet access
        self.model = SentenceTransformer(
            model_name, cache_folder=model_cache_dir, local_files_only=True
        )
        self.model_name = model_name

    def extract_query_relevant_content(
        self, document: str, query: str, max_chars: int = 500, min_sentences: int = 2
    ) -> str:
        """Extract content most relevant to the query using semantic similarity.

        Args:
            document: Full document content
            query: Original search query
            max_chars: Maximum characters to return
            min_sentences: Minimum number of sentences to include

        Returns:
            Extracted relevant content
        """
        try:
            # Split document into sentences
            sentences = self._split_into_sentences(document)
            if len(sentences) <= min_sentences:
                return document[:max_chars]

            # Encode query and sentences
            query_embedding = self.model.encode([query])
            sentence_embeddings = self.model.encode(sentences)

            # Calculate similarity scores
            similarities = np.dot(sentence_embeddings, query_embedding.T).flatten()

            # Sort sentences by relevance
            sentence_scores = list(zip(sentences, similarities))
            sentence_scores.sort(key=lambda x: x[1], reverse=True)

            # Select top sentences that fit within character limit
            selected_sentences = []
            current_length = 0

            for sentence, score in sentence_scores:
                sentence_length = len(sentence) + 1  # +1 for space
                if current_length + sentence_length <= max_chars:
                    selected_sentences.append((sentence, score))
                    current_length += sentence_length
                elif len(selected_sentences) >= min_sentences:
                    break

            # Ensure minimum sentences are included
            if len(selected_sentences) < min_sentences:
                for sentence, score in sentence_scores[:min_sentences]:
                    if (sentence, score) not in selected_sentences:
                        selected_sentences.append((sentence, score))

            # Re-order sentences to maintain document flow
            selected_text = [s[0] for s in selected_sentences]
            original_order = []
            for sentence in sentences:
                if sentence in selected_text:
                    original_order.append(sentence)

            result = " ".join(original_order)
            return result[:max_chars] if len(result) > max_chars else result

        except Exception as e:
            logger.warning(
                f"Query-relevant extraction failed: {e}, falling back to truncation"
            )
            return self._smart_truncate(document, max_chars)

    def extract_key_information(
        self, document: str, max_chars: int = 500
    ) -> Dict[str, Any]:
        """Extract structured key information from document.

        Args:
            document: Full document content
            max_chars: Maximum characters for extracted content

        Returns:
            Dictionary with structured information
        """
        try:
            # Extract code blocks
            code_blocks = re.findall(r"```[\s\S]*?```", document)

            # Extract headers/titles
            headers = re.findall(r"^#+\s+(.+)$", document, re.MULTILINE)

            # Extract bullet points/lists
            lists = re.findall(r"^\s*[-*+]\s+(.+)$", document, re.MULTILINE)

            # Extract key-value patterns (common in docs)
            key_values = re.findall(r"^\s*(\w+):\s*(.+)$", document, re.MULTILINE)

            # Get first paragraph for context
            paragraphs = [p.strip() for p in document.split("\n\n") if p.strip()]
            first_paragraph = paragraphs[0] if paragraphs else ""

            # Create structured response
            structured_info = {
                "summary": first_paragraph[:200] + "..."
                if len(first_paragraph) > 200
                else first_paragraph,
                "headers": headers[:3],  # Top 3 headers
                "code_examples": len(code_blocks),
                "key_points": lists[:5],  # Top 5 bullet points
                "config_items": dict(key_values[:3]) if key_values else {},
            }

            # If structured extraction doesn't yield much, fall back to smart truncation
            total_extracted = sum(
                len(str(v))
                for v in structured_info.values()
                if isinstance(v, (str, list))
            )
            if total_extracted < max_chars // 2:
                structured_info["additional_content"] = self._smart_truncate(
                    document, max_chars - total_extracted
                )

            return structured_info

        except Exception as e:
            logger.warning(f"Structured extraction failed: {e}")
            return {"content": self._smart_truncate(document, max_chars)}

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using simple heuristics."""
        # Simple sentence splitting - could be improved with more sophisticated NLP
        sentences = re.split(r"[.!?]+\s+", text)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

    def _smart_truncate(self, text: str, max_chars: int) -> str:
        """Intelligently truncate text at word/sentence boundaries."""
        if len(text) <= max_chars:
            return text

        # Try to cut at sentence boundary
        truncated = text[:max_chars]
        last_sentence_end = max(
            truncated.rfind("."), truncated.rfind("!"), truncated.rfind("?")
        )

        if last_sentence_end > max_chars * 0.7:  # If we can preserve most content
            return text[: last_sentence_end + 1]

        # Fall back to word boundary
        last_space = truncated.rfind(" ")
        if last_space > max_chars * 0.8:
            return text[:last_space] + "..."

        return text[:max_chars] + "..."


class HybridContentProcessor(ContentProcessor):
    """Hybrid content processor that combines multiple intelligent processing strategies."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the hybrid content processor."""
        super().__init__(model_name)

        # File type patterns for different processing strategies
        self.code_extensions = {
            ".py",
            ".js",
            ".ts",
            ".java",
            ".cpp",
            ".c",
            ".go",
            ".rs",
            ".php",
            ".rb",
            ".swift",
            ".kt",
        }
        self.config_extensions = {
            ".json",
            ".yaml",
            ".yml",
            ".toml",
            ".ini",
            ".conf",
            ".cfg",
            ".xml",
        }
        self.documentation_extensions = {".md", ".rst", ".txt", ".adoc"}

    def intelligent_content_processing(
        self,
        document: str,
        query: str,
        target_length: int = 500,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Hybrid approach that combines the best of all content processing strategies.

        Args:
            document: Full document content
            query: Original search query
            target_length: Target character length for processed content
            metadata: Document metadata for processing decisions

        Returns:
            Dictionary with processed content and processing metadata
        """
        if not document.strip():
            return self._create_result("", "empty_document", 0, 1.0)

        doc_length = len(document)

        # Step 1: If document is already short enough, return as-is
        if doc_length <= target_length:
            return self._create_result(document, "full_document", doc_length, 1.0)

        # Step 2: Determine file type for processing strategy
        file_ext = self._get_file_extension(metadata)
        processing_strategy = self._determine_processing_strategy(file_ext, document)

        # Step 3: Apply appropriate processing strategy
        if processing_strategy == "structured_code":
            return self._process_code_file(document, query, target_length, metadata)
        elif processing_strategy == "structured_config":
            return self._process_config_file(document, query, target_length, metadata)
        elif processing_strategy == "semantic_selection":
            return self._process_with_semantic_selection(document, query, target_length)
        else:  # documentation or fallback
            return self._process_documentation(document, query, target_length)

    def _determine_processing_strategy(self, file_ext: str, document: str) -> str:
        """Determine the best processing strategy based on file type and content."""
        if file_ext in self.code_extensions:
            # Check if it actually contains code patterns
            if self._contains_code_patterns(document):
                return "structured_code"

        if file_ext in self.config_extensions:
            return "structured_config"

        if file_ext in self.documentation_extensions or self._looks_like_documentation(
            document
        ):
            return "semantic_selection"

        # Fallback to semantic selection for unknown types
        return "semantic_selection"

    def _process_code_file(
        self,
        document: str,
        query: str,
        target_length: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Process code files with structured extraction."""
        try:
            # Extract code structure
            structure = self._extract_code_structure(document)

            # Get query-relevant code sections
            relevant_sections = self._get_relevant_code_sections(
                structure, query, target_length
            )

            if relevant_sections:
                processed_content = self._format_code_sections(relevant_sections)
                return self._create_result(
                    processed_content,
                    "structured_code",
                    len(document),
                    len(processed_content) / len(document),
                )

            # Fallback to semantic selection if structured extraction fails
            return self._process_with_semantic_selection(document, query, target_length)

        except Exception as e:
            logger.warning(
                f"Code processing failed: {e}, falling back to semantic selection"
            )
            return self._process_with_semantic_selection(document, query, target_length)

    def _process_config_file(
        self,
        document: str,
        query: str,
        target_length: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Process configuration files with key-value extraction."""
        try:
            # Extract configuration structure
            config_data = self._extract_config_structure(document)

            # Filter relevant configuration based on query
            relevant_config = self._filter_relevant_config(
                config_data, query, target_length
            )

            if relevant_config:
                processed_content = self._format_config_data(relevant_config)
                return self._create_result(
                    processed_content,
                    "structured_config",
                    len(document),
                    len(processed_content) / len(document),
                )

            # Fallback
            return self._process_with_semantic_selection(document, query, target_length)

        except Exception as e:
            logger.warning(
                f"Config processing failed: {e}, falling back to semantic selection"
            )
            return self._process_with_semantic_selection(document, query, target_length)

    def _process_with_semantic_selection(
        self, document: str, query: str, target_length: int
    ) -> Dict[str, Any]:
        """Process using semantic sentence selection and ranking."""
        try:
            # Extract query-relevant content using semantic similarity
            processed_content = self.extract_query_relevant_content(
                document, query, target_length, min_sentences=1
            )

            return self._create_result(
                processed_content,
                "semantic_selection",
                len(document),
                len(processed_content) / len(document),
            )

        except Exception as e:
            logger.warning(
                f"Semantic processing failed: {e}, falling back to smart truncation"
            )
            truncated = self._smart_truncate(document, target_length)
            return self._create_result(
                truncated,
                "smart_truncation",
                len(document),
                len(truncated) / len(document),
            )

    def _process_documentation(
        self, document: str, query: str, target_length: int
    ) -> Dict[str, Any]:
        """Process documentation with focus on structure and query relevance."""
        try:
            # Combine structured extraction with semantic selection
            structured_info = self.extract_key_information(document, target_length // 2)

            # If structured extraction provides good content, use it
            if self._is_good_structured_content(structured_info):
                content = self._format_structured_content(structured_info)
                return self._create_result(
                    content,
                    "structured_documentation",
                    len(document),
                    len(content) / len(document),
                )

            # Otherwise, use semantic selection
            return self._process_with_semantic_selection(document, query, target_length)

        except Exception as e:
            logger.warning(f"Documentation processing failed: {e}")
            return self._process_with_semantic_selection(document, query, target_length)

    # Helper methods for the hybrid approach
    def _get_file_extension(self, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Get file extension from metadata."""
        if metadata:
            return metadata.get("file_extension", "").lower()
        return ""

    def _contains_code_patterns(self, document: str) -> bool:
        """Check if document contains typical code patterns."""
        code_indicators = [
            "def ",
            "function ",
            "class ",
            "import ",
            "from ",
            "#!/",
            "{",
            "}",
            "()",
            "=>",
            "var ",
            "let ",
            "const ",
            "public ",
            "private ",
        ]
        return any(indicator in document for indicator in code_indicators)

    def _looks_like_documentation(self, document: str) -> bool:
        """Check if document looks like documentation."""
        doc_indicators = ["#", "##", "###", "**", "*", "- ", "1. ", "+ "]
        return any(
            document.startswith(indicator) or f"\n{indicator}" in document
            for indicator in doc_indicators
        )

    def _extract_code_structure(self, document: str) -> Dict[str, Any]:
        """Extract code structure (functions, classes, imports, etc.)."""
        structure = {
            "imports": [],
            "functions": [],
            "classes": [],
            "comments": [],
            "docstrings": [],
        }

        lines = document.split("\n")
        current_function = None
        current_class = None

        for line_num, line in enumerate(lines):
            stripped = line.strip()

            # Imports
            if stripped.startswith(("import ", "from ")):
                structure["imports"].append({"line": line_num, "content": stripped})

            # Function definitions
            elif stripped.startswith("def "):
                func_name = stripped.split("(")[0].replace("def ", "").strip()
                current_function = {
                    "name": func_name,
                    "line": line_num,
                    "content": [line],
                }
                structure["functions"].append(current_function)

            # Class definitions
            elif stripped.startswith("class "):
                class_name = (
                    stripped.split("(")[0].replace("class ", "").strip().rstrip(":")
                )
                current_class = {
                    "name": class_name,
                    "line": line_num,
                    "content": [line],
                }
                structure["classes"].append(current_class)

            # Comments and docstrings
            elif stripped.startswith("#"):
                structure["comments"].append({"line": line_num, "content": stripped})
            elif '"""' in stripped or "'''" in stripped:
                structure["docstrings"].append({"line": line_num, "content": stripped})

            # Add content to current function/class
            else:
                if current_function and line.startswith("    "):
                    current_function["content"].append(line)
                elif current_class and line.startswith("    "):
                    current_class["content"].append(line)
                else:
                    current_function = None
                    current_class = None

        return structure

    def _get_relevant_code_sections(
        self, structure: Dict[str, Any], query: str, target_length: int
    ) -> List[Dict[str, Any]]:
        """Get code sections most relevant to the query."""
        relevant_sections = []

        # Score each section based on query relevance
        for section_type in ["functions", "classes"]:
            for item in structure.get(section_type, []):
                content = "\n".join(item["content"])

                # Simple relevance scoring based on name and content
                relevance_score = 0
                query_words = query.lower().split()

                for word in query_words:
                    if word in item["name"].lower():
                        relevance_score += 2
                    if word in content.lower():
                        relevance_score += 1

                if relevance_score > 0:
                    relevant_sections.append(
                        {
                            "type": section_type[
                                :-1
                            ],  # Remove 's' from functions/classes
                            "name": item["name"],
                            "content": content,
                            "relevance": relevance_score,
                            "length": len(content),
                        }
                    )

        # Sort by relevance and fit within target length
        relevant_sections.sort(key=lambda x: x["relevance"], reverse=True)

        selected_sections = []
        current_length = 0

        for section in relevant_sections:
            if current_length + section["length"] <= target_length:
                selected_sections.append(section)
                current_length += section["length"]
            elif not selected_sections:  # Include at least one section
                # Truncate the section to fit
                remaining_length = target_length - current_length
                section["content"] = section["content"][:remaining_length]
                section["length"] = len(section["content"])
                selected_sections.append(section)
                break

        return selected_sections

    def _format_code_sections(self, sections: List[Dict[str, Any]]) -> str:
        """Format selected code sections into readable content."""
        formatted_parts = []

        for section in sections:
            header = f"# {section['type'].title()}: {section['name']}"
            formatted_parts.append(header)
            formatted_parts.append(section["content"])
            formatted_parts.append("")  # Empty line separator

        return "\n".join(formatted_parts).strip()

    def _extract_config_structure(self, document: str) -> Dict[str, Any]:
        """Extract configuration structure from config files."""
        config_data = {}

        lines = document.split("\n")
        current_section = "root"

        for line in lines:
            stripped = line.strip()

            # Skip empty lines and comments
            if not stripped or stripped.startswith("#") or stripped.startswith("//"):
                continue

            # Section headers (like [section] in INI files)
            if stripped.startswith("[") and stripped.endswith("]"):
                current_section = stripped[1:-1]
                config_data[current_section] = {}
                continue

            # Key-value pairs
            if "=" in stripped or ":" in stripped:
                separator = "=" if "=" in stripped else ":"
                parts = stripped.split(separator, 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()

                    if current_section not in config_data:
                        config_data[current_section] = {}
                    config_data[current_section][key] = value

        return config_data

    def _filter_relevant_config(
        self, config_data: Dict[str, Any], query: str, target_length: int
    ) -> Dict[str, Any]:
        """Filter configuration data based on query relevance."""
        query_words = set(query.lower().split())
        relevant_config = {}

        for section, items in config_data.items():
            relevant_items = {}

            # Check section name relevance
            section_relevant = any(word in section.lower() for word in query_words)

            for key, value in items.items():
                # Check key/value relevance
                key_relevant = any(word in key.lower() for word in query_words)
                value_relevant = any(word in str(value).lower() for word in query_words)

                if section_relevant or key_relevant or value_relevant:
                    relevant_items[key] = value

            if relevant_items:
                relevant_config[section] = relevant_items

        return relevant_config

    def _format_config_data(self, config_data: Dict[str, Any]) -> str:
        """Format configuration data into readable content."""
        formatted_parts = []

        for section, items in config_data.items():
            if section != "root":
                formatted_parts.append(f"[{section}]")

            for key, value in items.items():
                formatted_parts.append(f"{key}: {value}")

            formatted_parts.append("")  # Empty line separator

        return "\n".join(formatted_parts).strip()

    def _is_good_structured_content(self, structured_info: Dict[str, Any]) -> bool:
        """Check if structured extraction provided good content."""
        total_content = 0
        for key, value in structured_info.items():
            if isinstance(value, str):
                total_content += len(value)
            elif isinstance(value, list):
                total_content += sum(len(str(item)) for item in value)

        return total_content > 100  # Arbitrary threshold for "good" content

    def _format_structured_content(self, structured_info: Dict[str, Any]) -> str:
        """Format structured information into readable text."""
        parts = []

        if structured_info.get("summary"):
            parts.append(f"Summary: {structured_info['summary']}")

        if structured_info.get("headers"):
            parts.append(f"Sections: {', '.join(structured_info['headers'])}")

        if structured_info.get("key_points"):
            points = "\n- ".join(structured_info["key_points"])
            parts.append(f"Key Points:\n- {points}")

        if structured_info.get("config_items"):
            config = "\n".join(
                [f"{k}: {v}" for k, v in structured_info["config_items"].items()]
            )
            parts.append(f"Configuration:\n{config}")

        if structured_info.get("code_examples"):
            parts.append(f"Code Examples: {structured_info['code_examples']}")

        if structured_info.get("additional_content"):
            parts.append(structured_info["additional_content"])

        return "\n\n".join(parts)

    def _create_result(
        self, content: str, method: str, original_length: int, compression_ratio: float
    ) -> Dict[str, Any]:
        """Create a standardized result dictionary."""
        return {
            "content": content,
            "processing_method": method,
            "original_length": original_length,
            "compressed_ratio": round(compression_ratio, 3),
            "final_length": len(content),
        }
