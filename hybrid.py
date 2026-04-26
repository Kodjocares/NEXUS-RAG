from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from rank_bm25 import BM25Okapi
from ingestion.embedder import embed_query
from config import get_settings
import logging

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class RetrievedChunk:
    chunk_id: str
    document_id: str
    content: str
    title: str
    source: str
    doc_type: str
    category: str
    score: float
    chunk_index: int


async def vector_search(
    db: AsyncSession,
    query_embedding: List[float],
    top_k: int = None,
    category_filter: str = None,
) -> List[Tuple[str, float]]:
    """pgvector cosine similarity search. Returns (chunk_id, score) pairs."""
    top_k = top_k or settings.TOP_K_VECTOR
    vec_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

    category_clause = ""
    params: Dict[str, Any] = {"vec": vec_str, "top_k": top_k}
    if category_filter:
        category_clause = "AND d.category = :category"
        params["category"] = category_filter

    sql = text(f"""
        SELECT c.id::text, 1 - (c.embedding <=> :vec::vector) AS score
        FROM chunks c
        JOIN documents d ON c.document_id = d.id
        WHERE c.embedding IS NOT NULL
        {category_clause}
        ORDER BY c.embedding <=> :vec::vector
        LIMIT :top_k
    """)
    result = await db.execute(sql, params)
    rows = result.fetchall()
    return [(row[0], float(row[1])) for row in rows]


async def bm25_search(
    db: AsyncSession,
    query: str,
    top_k: int = None,
    category_filter: str = None,
) -> List[Tuple[str, float]]:
    """BM25 full-text search using in-memory BM25Okapi on fetched chunks."""
    top_k = top_k or settings.TOP_K_BM25

    category_clause = ""
    params: Dict[str, Any] = {}
    if category_filter:
        category_clause = "WHERE d.category = :category"
        params["category"] = category_filter

    # Fetch all chunks (for production: use pg full-text search instead)
    sql = text(f"""
        SELECT c.id::text, c.content
        FROM chunks c
        JOIN documents d ON c.document_id = d.id
        {category_clause}
        ORDER BY c.created_at DESC
        LIMIT 5000
    """)
    result = await db.execute(sql, params)
    rows = result.fetchall()

    if not rows:
        return []

    chunk_ids = [row[0] for row in rows]
    corpus = [row[1] for row in rows]

    # Tokenize
    tokenized_corpus = [doc.lower().split() for doc in corpus]
    tokenized_query = query.lower().split()

    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenized_query)

    # Get top-k indices
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [(chunk_ids[i], float(scores[i])) for i in top_indices if scores[i] > 0]


def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[str, float]]],
    k: int = None,
) -> List[Tuple[str, float]]:
    """
    Combine multiple ranked lists using RRF.
    RRF score = sum(1 / (k + rank)) across lists.
    """
    k = k or settings.RRF_K
    rrf_scores: Dict[str, float] = {}

    for ranked_list in ranked_lists:
        for rank, (doc_id, _) in enumerate(ranked_list, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results


async def hybrid_search(
    db: AsyncSession,
    query: str,
    top_k: int = None,
    category_filter: str = None,
) -> List[RetrievedChunk]:
    """
    Full hybrid search: vector + BM25 fused via RRF.
    Returns enriched RetrievedChunk objects.
    """
    top_k = top_k or settings.TOP_K_FINAL

    # Run both searches concurrently
    query_embedding = embed_query(query)

    vector_results = await vector_search(db, query_embedding, category_filter=category_filter)
    bm25_results = await bm25_search(db, query, category_filter=category_filter)

    # Fuse
    fused = reciprocal_rank_fusion([vector_results, bm25_results])
    top_chunk_ids = [chunk_id for chunk_id, _ in fused[:top_k]]
    fused_scores = dict(fused)

    if not top_chunk_ids:
        return []

    # Fetch full chunk details
    placeholders = ", ".join(f":id{i}" for i in range(len(top_chunk_ids)))
    params = {f"id{i}": cid for i, cid in enumerate(top_chunk_ids)}

    sql = text(f"""
        SELECT
            c.id::text,
            c.document_id::text,
            c.content,
            c.chunk_index,
            d.title,
            d.source,
            d.doc_type,
            COALESCE(d.category, 'general') as category
        FROM chunks c
        JOIN documents d ON c.document_id = d.id
        WHERE c.id::text IN ({placeholders})
    """)
    result = await db.execute(sql, params)
    rows = result.fetchall()

    # Map to RetrievedChunk, preserving fused ranking order
    row_map = {row[0]: row for row in rows}
    chunks = []
    for chunk_id in top_chunk_ids:
        if chunk_id in row_map:
            row = row_map[chunk_id]
            chunks.append(RetrievedChunk(
                chunk_id=row[0],
                document_id=row[1],
                content=row[2],
                chunk_index=row[3],
                title=row[4],
                source=row[5],
                doc_type=row[6],
                category=row[7],
                score=fused_scores.get(chunk_id, 0.0),
            ))

    return chunks
