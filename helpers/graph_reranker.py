"""
Graph-based re-ranking using Neo4j and Graphiti.

This module enhances ChromaDB search results by applying temporal knowledge graph
intelligence to re-rank and filter endpoints based on:
- Historical usage patterns
- Endpoint relationships and workflows
- Temporal context (recency, frequency)
- Success/failure patterns

The re-ranker reduces context bloat by returning only the most relevant 2-3 endpoints
instead of the default 5+ from vector search alone.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from graphiti_core import Graphiti
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode, EpisodeType
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError

from utils.config import config

logger = logging.getLogger("graph_reranker")
logger.setLevel(logging.INFO)


class GraphReranker:
    """
    Re-ranks ChromaDB results using Neo4j + Graphiti temporal knowledge graphs.

    This class maintains a temporal knowledge graph of endpoint usage patterns
    and relationships. It learns from successful API interactions and uses this
    knowledge to improve search precision over time.
    """

    def __init__(self) -> None:
        """Initialize GraphReranker with Neo4j and Graphiti connections."""
        self.neo4j_uri: str = config.NEO4J_URI
        self.neo4j_user: str = config.NEO4J_USER
        self.neo4j_password: str = config.NEO4J_PASSWORD
        self.neo4j_database: str = config.NEO4J_DATABASE
        self.enabled: bool = config.GRAPHITI_ENABLED

        self.driver: Optional[Any] = None
        self.graphiti: Optional[Graphiti] = None

        if self.enabled:
            self._initialize_connections()

    def _initialize_connections(self) -> None:
        """Initialize Neo4j driver and Graphiti client."""
        try:
            # Initialize Neo4j driver
            self.driver = GraphDatabase.driver(
                self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password)
            )
            # Verify connectivity
            self.driver.verify_connectivity()
            logger.info("Neo4j connection established successfully")

            # Initialize Graphiti
            try:
                self.graphiti = Graphiti(
                    neo4j_uri=self.neo4j_uri,
                    neo4j_user=self.neo4j_user,
                    neo4j_password=self.neo4j_password,
                    neo4j_database=self.neo4j_database,
                )
                logger.info("Graphiti initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Graphiti: {e}")
                # Close driver since Graphiti failed
                if self.driver:
                    self.driver.close()
                self.enabled = False
                self.driver = None
                self.graphiti = None

        except Neo4jError as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            self.enabled = False
            self.driver = None
            self.graphiti = None

    def close(self) -> None:
        """Close Neo4j driver connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")

    def rerank(
        self, query: str, chroma_results: List[Dict[str, Any]], n_results: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Re-rank ChromaDB results using graph intelligence.

        Args:
            query: Original search query
            chroma_results: List of results from ChromaDB
            n_results: Number of results to return (default: 3)

        Returns:
            Re-ranked and filtered list of results
        """
        if not self.enabled or not chroma_results:
            logger.debug("Graphiti disabled or no results to rerank")
            return chroma_results[:n_results]

        if not self.graphiti:
            logger.warning("Graphiti not initialized, returning original results")
            return chroma_results[:n_results]

        try:
            # Score each result based on graph context
            scored_results = []
            for result in chroma_results:
                score = self._calculate_graph_score(query, result)
                scored_results.append({"result": result, "graph_score": score})

            # Sort by graph score (higher is better)
            scored_results.sort(key=lambda x: x["graph_score"], reverse=True)

            # Return top n_results
            reranked = [item["result"] for item in scored_results[:n_results]]

            logger.info(
                f"Re-ranked {len(chroma_results)} results to top {len(reranked)}"
            )
            return reranked

        except Exception as e:
            logger.error(f"Error during re-ranking: {e}")
            return chroma_results[:n_results]

    def _calculate_graph_score(self, query: str, result: Dict[str, Any]) -> float:
        """
        Calculate graph-based relevance score for a result.

        The score is based on:
        - Usage frequency (how often this endpoint is used)
        - Recency (when it was last used successfully)
        - Relationship strength (connections to other endpoints in workflows)
        - Query similarity (contextual relevance from graph)

        Args:
            query: Original search query
            result: Single ChromaDB result

        Returns:
            Float score (higher is better)
        """
        if not self.graphiti:
            return 0.0

        try:
            metadata = result.get("metadata", {})
            endpoint_id = f"{metadata.get('method', '')} {metadata.get('path', '')}"

            # Base score starts at 0
            score = 0.0

            # Factor 1: Usage frequency (0-40 points)
            frequency_score = self._get_usage_frequency(endpoint_id)
            score += min(frequency_score, 40.0)

            # Factor 2: Recency (0-30 points)
            recency_score = self._get_recency_score(endpoint_id)
            score += min(recency_score, 30.0)

            # Factor 3: Relationship strength (0-20 points)
            relationship_score = self._get_relationship_score(endpoint_id, query)
            score += min(relationship_score, 20.0)

            # Factor 4: Success rate (0-10 points)
            success_score = self._get_success_rate(endpoint_id)
            score += min(success_score, 10.0)

            logger.debug(
                f"Endpoint {endpoint_id} scored {score:.2f} "
                f"(freq: {frequency_score:.1f}, "
                f"recency: {recency_score:.1f}, "
                f"rel: {relationship_score:.1f}, "
                f"success: {success_score:.1f})"
            )

            return score

        except Exception as e:
            logger.error(f"Error calculating graph score: {e}")
            return 0.0

    def _get_usage_frequency(self, endpoint_id: str) -> float:
        """
        Get usage frequency score for an endpoint.

        Args:
            endpoint_id: Unique endpoint identifier (method + path)

        Returns:
            Frequency score (0-40)
        """
        if not self.driver:
            return 0.0

        try:
            with self.driver.session(database=self.neo4j_database) as session:
                result = session.run(
                    """
                    MATCH (e:Endpoint {id: $endpoint_id})
                    RETURN COALESCE(e.usage_count, 0) AS count
                    """,
                    endpoint_id=endpoint_id,
                )
                record = result.single()
                if record:
                    count = record["count"]
                    # Logarithmic scaling: 1 use = 5 pts, 10 uses = 15 pts, 100 uses = 25 pts
                    import math

                    return min(5 * math.log10(count + 1), 40.0)
                return 0.0
        except Exception as e:
            logger.debug(f"Error getting frequency for {endpoint_id}: {e}")
            return 0.0

    def _get_recency_score(self, endpoint_id: str) -> float:
        """
        Get recency score for an endpoint based on last successful use.

        Args:
            endpoint_id: Unique endpoint identifier

        Returns:
            Recency score (0-30)
        """
        if not self.driver:
            return 0.0

        try:
            with self.driver.session(database=self.neo4j_database) as session:
                result = session.run(
                    """
                    MATCH (e:Endpoint {id: $endpoint_id})
                    RETURN e.last_used_timestamp AS last_used
                    """,
                    endpoint_id=endpoint_id,
                )
                record = result.single()
                if record and record["last_used"]:
                    last_used = datetime.fromisoformat(record["last_used"])
                    age_hours = (datetime.now() - last_used).total_seconds() / 3600

                    # Decay function: full points if <1 hour, half at 24 hours, min at 168 hours (1 week)
                    if age_hours < 1:
                        return 30.0
                    elif age_hours < 24:
                        return 30.0 * (1 - (age_hours - 1) / 46)  # Linear decay to 15
                    elif age_hours < 168:
                        return 15.0 * (
                            1 - (age_hours - 24) / 288
                        )  # Further decay to 0
                    else:
                        return 0.0
                return 0.0
        except Exception as e:
            logger.debug(f"Error getting recency for {endpoint_id}: {e}")
            return 0.0

    def _get_relationship_score(self, endpoint_id: str, query: str) -> float:
        """
        Get relationship strength score based on endpoint connections.

        Args:
            endpoint_id: Unique endpoint identifier
            query: Original search query

        Returns:
            Relationship score (0-20)
        """
        if not self.driver:
            return 0.0

        try:
            with self.driver.session(database=self.neo4j_database) as session:
                # Count relationships to other frequently used endpoints
                result = session.run(
                    """
                    MATCH (e:Endpoint {id: $endpoint_id})-[r:USED_WITH]-(other:Endpoint)
                    RETURN COUNT(DISTINCT other) AS related_count,
                           SUM(r.weight) AS total_weight
                    """,
                    endpoint_id=endpoint_id,
                )
                record = result.single()
                if record:
                    related_count = record["related_count"] or 0
                    total_weight = record["total_weight"] or 0

                    # Base score from relationship count
                    base_score = min(related_count * 2, 10.0)

                    # Bonus from relationship strength
                    weight_score = min(total_weight / 10, 10.0)

                    return base_score + weight_score
                return 0.0
        except Exception as e:
            logger.debug(f"Error getting relationships for {endpoint_id}: {e}")
            return 0.0

    def _get_success_rate(self, endpoint_id: str) -> float:
        """
        Get success rate score for an endpoint.

        Args:
            endpoint_id: Unique endpoint identifier

        Returns:
            Success rate score (0-10)
        """
        if not self.driver:
            return 0.0

        try:
            with self.driver.session(database=self.neo4j_database) as session:
                result = session.run(
                    """
                    MATCH (e:Endpoint {id: $endpoint_id})
                    RETURN COALESCE(e.success_count, 0) AS success,
                           COALESCE(e.failure_count, 0) AS failure
                    """,
                    endpoint_id=endpoint_id,
                )
                record = result.single()
                if record:
                    success = record["success"]
                    failure = record["failure"]
                    total = success + failure

                    if total == 0:
                        return 5.0  # Neutral score for untested endpoints

                    success_rate = success / total
                    return success_rate * 10.0
                return 5.0  # Neutral score
        except Exception as e:
            logger.debug(f"Error getting success rate for {endpoint_id}: {e}")
            return 5.0

    async def record_usage(
        self,
        endpoint: Dict[str, Any],
        query: str,
        success: bool = True,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record endpoint usage in the temporal knowledge graph.

        This method updates the graph with:
        - Endpoint usage count
        - Success/failure tracking
        - Temporal context
        - Relationships to other endpoints used in the same session

        Args:
            endpoint: Endpoint metadata dict
            query: Original search query
            success: Whether the usage was successful
            context: Optional additional context (related endpoints, params, etc.)
        """
        if not self.enabled or not self.graphiti:
            return

        try:
            metadata = endpoint.get("metadata", {})
            endpoint_id = f"{metadata.get('method', '')} {metadata.get('path', '')}"

            # Update endpoint node
            await self._update_endpoint_node(endpoint_id, success)

            # Create/update episode in Graphiti for temporal context
            await self._create_episode(endpoint_id, query, success, context)

            # Update relationships if context provided
            if context and context.get("related_endpoints"):
                await self._update_relationships(
                    endpoint_id, context["related_endpoints"]
                )

            logger.debug(f"Recorded usage for {endpoint_id} (success: {success})")

        except Exception as e:
            logger.error(f"Error recording usage: {e}")

    async def _update_endpoint_node(self, endpoint_id: str, success: bool) -> None:
        """Update endpoint node with usage statistics."""
        if not self.driver:
            return

        try:
            with self.driver.session(database=self.neo4j_database) as session:
                session.run(
                    """
                    MERGE (e:Endpoint {id: $endpoint_id})
                    SET e.usage_count = COALESCE(e.usage_count, 0) + 1,
                        e.last_used_timestamp = $timestamp,
                        e.success_count = COALESCE(e.success_count, 0) + $success_inc,
                        e.failure_count = COALESCE(e.failure_count, 0) + $failure_inc
                    """,
                    endpoint_id=endpoint_id,
                    timestamp=datetime.now().isoformat(),
                    success_inc=1 if success else 0,
                    failure_inc=0 if success else 1,
                )
        except Exception as e:
            logger.error(f"Error updating endpoint node: {e}")

    async def _create_episode(
        self,
        endpoint_id: str,
        query: str,
        success: bool,
        context: Optional[Dict[str, Any]],
    ) -> None:
        """Create temporal episode in Graphiti."""
        if not self.graphiti:
            return

        try:
            episode_content = (
                f"User searched for: '{query}' and used endpoint {endpoint_id}. "
                f"Result: {'success' if success else 'failure'}."
            )

            if context:
                episode_content += f" Context: {str(context)}"

            await self.graphiti.add_episode(
                name=f"endpoint_usage_{datetime.now().isoformat()}",
                episode_body=episode_content,
                episode_type=EpisodeType.json,
                reference_time=datetime.now(),
                source_description="Nautobot MCP endpoint usage tracking",
            )
        except Exception as e:
            logger.error(f"Error creating episode: {e}")

    async def _update_relationships(
        self, endpoint_id: str, related_endpoints: List[str]
    ) -> None:
        """Update relationships between endpoints used together."""
        if not self.driver:
            return

        try:
            with self.driver.session(database=self.neo4j_database) as session:
                for related_id in related_endpoints:
                    session.run(
                        """
                        MERGE (e1:Endpoint {id: $endpoint_id})
                        MERGE (e2:Endpoint {id: $related_id})
                        MERGE (e1)-[r:USED_WITH]-(e2)
                        SET r.weight = COALESCE(r.weight, 0) + 1,
                            r.last_used = $timestamp
                        """,
                        endpoint_id=endpoint_id,
                        related_id=related_id,
                        timestamp=datetime.now().isoformat(),
                    )
        except Exception as e:
            logger.error(f"Error updating relationships: {e}")

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on Neo4j and Graphiti connections.

        Returns:
            Dict with health status information
        """
        health_status = {
            "enabled": self.enabled,
            "neo4j_connected": False,
            "graphiti_initialized": False,
            "errors": [],
        }

        if not self.enabled:
            return health_status

        # Check Neo4j
        if self.driver:
            try:
                self.driver.verify_connectivity()
                health_status["neo4j_connected"] = True
            except Exception as e:
                health_status["errors"].append(f"Neo4j connection error: {str(e)}")

        # Check Graphiti
        if self.graphiti:
            health_status["graphiti_initialized"] = True
        else:
            health_status["errors"].append("Graphiti not initialized")

        return health_status
