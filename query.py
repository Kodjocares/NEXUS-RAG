from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from pydantic import BaseModel
from typing import Optional, List
import anthropic, json, time, uuid, logging
from database import get_db, QueryLog
from retrieval.hybrid import hybrid_search, RetrievedChunk
from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter(prefix="/query", tags=["query"])

client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)


SYSTEM_PROMPT = """You are NEXUS RAG — an intelligent document analyst. Answer questions strictly based on the provided context chunks. Follow these rules:

1. Cite sources inline using [Doc: <title>, Chunk #<index>] format.
2. If the context doesn't contain enough information, say so clearly — do not hallucinate.
3. For cybersecurity questions, be precise about threats, CVEs, and mitigations.
4. For business questions, be analytical and structured.
5. Keep answers concise but complete. Use markdown formatting when helpful.
6. At the end of your answer, list the sources you used under a "## Sources" heading."""


def _build_context(chunks: List[RetrievedChunk]) -> str:
    parts = []
    for i, chunk in enumerate(chunks):
        parts.append(
            f"--- [Context {i+1}] ---\n"
            f"Title: {chunk.title}\n"
            f"Source: {chunk.source}\n"
            f"Category: {chunk.category}\n"
            f"Chunk #{chunk.chunk_index}\n\n"
            f"{chunk.content}"
        )
    return "\n\n".join(parts)


from memory.conversation import session_store, rewrite_query_for_context
import uuid as _uuid

class QueryRequest(BaseModel):
    query: str
    category: Optional[str] = None
    top_k: Optional[int] = None
    stream: bool = True
    rerank: bool = False
    session_id: Optional[str] = None  # pass to maintain multi-turn memory


class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    query: str
    latency_ms: float


@router.post("/stream")
async def query_stream(req: QueryRequest, db: AsyncSession = Depends(get_db)):
    """Stream answer via SSE. Frontend connects with EventSource."""
    t0 = time.time()

    # Session memory — resolve or create
    sid = req.session_id or str(_uuid.uuid4())
    session = session_store.get_or_create(sid)

    # Rewrite follow-up questions as standalone search queries
    search_query = rewrite_query_for_context(session, req.query)

    chunks = await hybrid_search(db, search_query, top_k=req.top_k, category_filter=req.category)
    if req.rerank and chunks:
        chunks = rerank(search_query, chunks, top_k=req.top_k or settings.TOP_K_FINAL)

    if not chunks:
        async def empty_stream():
            yield f"data: {json.dumps({'type': 'error', 'content': 'No relevant documents found. Please ingest documents first.'})}\n\n"
        return StreamingResponse(empty_stream(), media_type="text/event-stream")

    context = _build_context(chunks)
    sources = [
        {
            "title": c.title,
            "source": c.source,
            "doc_type": c.doc_type,
            "category": c.category,
            "chunk_index": c.chunk_index,
            "score": round(c.score, 4),
            "preview": c.content[:200] + "..." if len(c.content) > 200 else c.content,
        }
        for c in chunks
    ]

    async def event_generator():
        # Send sources + session_id first
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources, 'session_id': sid})}\n\n"

        full_answer = []
        # Build messages including prior turns for multi-turn coherence
        prior_messages = session.build_messages_context()
        current_message = {
            "role": "user",
            "content": f"Context:\n{context}\n\n---\nQuestion: {req.query}",
        }
        all_messages = prior_messages + [current_message] if prior_messages else [current_message]
        try:
            with client.messages.stream(
                model=settings.LLM_MODEL,
                max_tokens=settings.MAX_TOKENS,
                temperature=settings.TEMPERATURE,
                system=SYSTEM_PROMPT,
                messages=all_messages,
            ) as stream:
                for text_chunk in stream.text_stream:
                    full_answer.append(text_chunk)
                    yield f"data: {json.dumps({'type': 'token', 'content': text_chunk})}\n\n"

        except Exception as e:
            logger.error(f"LLM stream error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
            return

        latency = round((time.time() - t0) * 1000, 1)
        answer_text = "".join(full_answer)

        # Save to session memory
        session.add_turn("user", req.query)
        session.add_turn("assistant", answer_text[:600])  # truncate for memory efficiency

        yield f"data: {json.dumps({'type': 'done', 'latency_ms': latency, 'session_id': sid})}\n\n"

        # Log query
        try:
            log = QueryLog(
                id=uuid.uuid4(),
                query=req.query,
                answer=answer_text,
                sources_used=sources,
                latency_ms=latency,
            )
            db.add(log)
            await db.commit()
        except Exception as e:
            logger.warning(f"Failed to log query: {e}")

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/sync", response_model=QueryResponse)
async def query_sync(req: QueryRequest, db: AsyncSession = Depends(get_db)):
    """Non-streaming query. Returns full answer at once."""
    t0 = time.time()

    chunks = await hybrid_search(db, req.query, top_k=req.top_k, category_filter=req.category)
    if not chunks:
        raise HTTPException(404, "No relevant documents found.")

    context = _build_context(chunks)
    sources = [
        {
            "title": c.title,
            "source": c.source,
            "doc_type": c.doc_type,
            "category": c.category,
            "chunk_index": c.chunk_index,
            "score": round(c.score, 4),
            "preview": c.content[:200] + "..." if len(c.content) > 200 else c.content,
        }
        for c in chunks
    ]

    response = client.messages.create(
        model=settings.LLM_MODEL,
        max_tokens=settings.MAX_TOKENS,
        temperature=settings.TEMPERATURE,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Context:\n{context}\n\n---\nQuestion: {req.query}"}],
    )

    answer = response.content[0].text
    latency = round((time.time() - t0) * 1000, 1)

    return QueryResponse(answer=answer, sources=sources, query=req.query, latency_ms=latency)
