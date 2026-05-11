"""BFS-style graph exploration of a Qdrant collection from a query prompt."""
from __future__ import annotations

from collections import deque

from qdrant_client import QdrantClient, models

from eigenmind.config import MAX_CHUNKS_FOR_CONTEXT, NEIGHBORS_TO_FETCH


def explore_graph_for_context(
    collection_name: str,
    prompt: str,
    client: QdrantClient,
    embedding_model,
    max_chunks: int = MAX_CHUNKS_FOR_CONTEXT,
    neighbors_to_fetch: int = NEIGHBORS_TO_FETCH,
) -> tuple[list, list[dict]]:
    """BFS graph exploration from a prompt.

    Returns (retrieved_points_with_vectors, top_sources_details).
    """
    query_vector = embedding_model.encode(prompt).tolist()

    collected_ids: set = set()
    frontier: deque = deque()
    top_sources_details: list[dict] = []

    initial = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=neighbors_to_fetch,
        with_payload=True,
    )

    for point in initial.points:
        if point.id not in collected_ids:
            collected_ids.add(point.id)
            frontier.append(point.id)
            top_sources_details.append({
                "filename": point.payload.get("filename", "N/A"),
                "text": point.payload.get("text", ""),
            })

    while frontier and len(collected_ids) < max_chunks:
        current_id = frontier.popleft()
        try:
            neighbors = client.query_points(
                collection_name=collection_name,
                query=current_id,
                limit=neighbors_to_fetch,
                query_filter=models.Filter(
                    must_not=[models.HasIdCondition(has_id=list(collected_ids))]
                ),
            )
            for point in neighbors.points:
                if point.id not in collected_ids and len(collected_ids) < max_chunks:
                    collected_ids.add(point.id)
                    frontier.append(point.id)
        except Exception:
            pass

    retrieved = client.retrieve(
        collection_name=collection_name,
        ids=list(collected_ids),
        with_vectors=True,
        with_payload=True,
    )
    return retrieved, top_sources_details


def explore_graph_with_initial_set(
    collection_name: str,
    query_vector: list[float],
    client: QdrantClient,
    max_chunks: int,
    neighbors_to_fetch: int = NEIGHBORS_TO_FETCH,
) -> tuple[list, set]:
    """Variant used by the graph explorer page: returns (retrieved_points, initial_point_ids)."""
    collected_ids: set = set()
    frontier: deque = deque()

    initial = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=neighbors_to_fetch,
        with_payload=False,
    )
    for point in initial.points:
        if point.id not in collected_ids:
            collected_ids.add(point.id)
            frontier.append(point.id)
    initial_point_ids = collected_ids.copy()

    while frontier and len(collected_ids) < max_chunks:
        current_id = frontier.popleft()
        try:
            neighbors = client.query_points(
                collection_name=collection_name,
                query=current_id,
                limit=neighbors_to_fetch,
                query_filter=models.Filter(
                    must_not=[models.HasIdCondition(has_id=list(collected_ids))]
                ),
            )
            for point in neighbors.points:
                if point.id not in collected_ids and len(collected_ids) < max_chunks:
                    collected_ids.add(point.id)
                    frontier.append(point.id)
        except Exception:
            pass

    retrieved = client.retrieve(
        collection_name=collection_name,
        ids=list(collected_ids),
        with_vectors=True,
        with_payload=True,
    )
    return retrieved, initial_point_ids
