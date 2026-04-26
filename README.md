# NEXUS RAG

> Hybrid Retrieval-Augmented Generation — Cybersecurity × Business × General  
> FastAPI · pgvector · sentence-transformers · BM25 · Claude Sonnet · React/Vite

---

## Architecture

```
                  ┌─────────────────────────────────────────────┐
                  │              NEXUS RAG SYSTEM               │
                  └─────────────────────────────────────────────┘

  ┌──────────┐    ┌────────────────────────────────────────────┐
  │  Ingest  │───▶│  Loaders: PDF · DOCX · TXT · URL scraper  │
  └──────────┘    │  Chunker: Recursive split + overlap        │
                  │  Embedder: all-MiniLM-L6-v2 (local)       │
                  │  Storage: PostgreSQL + pgvector HNSW        │
                  └────────────────────────────────────────────┘

  ┌──────────┐    ┌────────────────────────────────────────────┐
  │  Query   │───▶│  Vector search (pgvector cosine)           │
  └──────────┘    │  BM25 search (rank-bm25)                   │
                  │  RRF Fusion → Cross-encoder Re-rank        │
                  │  Query rewrite (multi-turn memory)         │
                  │  Claude Sonnet → SSE streaming             │
                  └────────────────────────────────────────────┘

  ┌──────────┐    ┌────────────────────────────────────────────┐
  │   Eval   │───▶│  RAGAS-style: Faithfulness                 │
  └──────────┘    │               Answer Relevance             │
                  │               Context Precision            │
                  │               Context Recall               │
                  │  LLM-as-judge using Claude Haiku           │
                  └────────────────────────────────────────────┘
```

---

## Quick Start

### Option A — Docker (recommended)

```bash
git clone <repo>
cd nexus-rag/backend
cp .env.example .env
# Edit .env — add your ANTHROPIC_API_KEY
cd ..
docker-compose up --build
```

- Frontend: http://localhost:5173  
- API docs: http://localhost:8000/docs  

### Option B — Local dev

```bash
# 1. Start PostgreSQL with pgvector
docker run -d --name nexus_pg \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=nexus_rag \
  -p 5432:5432 \
  pgvector/pgvector:pg16

# 2. Backend
cd backend
cp .env.example .env   # fill in ANTHROPIC_API_KEY
pip install -r requirements.txt
uvicorn main:app --reload

# 3. Frontend
cd ../frontend
npm install
npm run dev
```

---

## API Reference

### Ingest

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/ingest/file` | Upload PDF, DOCX, TXT, MD |
| POST | `/ingest/url` | Scrape and ingest a URL |

**Form fields:** `file` / `url`, `category` (cybersecurity \| business \| general)

### Query

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/query/stream` | SSE streaming response |
| POST | `/query/sync` | Synchronous response |

**Body:**
```json
{
  "query": "What are NIS2 requirements for SMEs?",
  "category": "cybersecurity",
  "top_k": 6,
  "rerank": true,
  "session_id": "optional-uuid-for-multi-turn"
}
```

### Documents

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/documents/` | List all documents |
| DELETE | `/documents/{id}` | Delete document + chunks |
| GET | `/documents/stats` | System statistics |

### Evaluation

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/eval/single` | Evaluate one question |
| POST | `/eval/batch` | Batch evaluation with report |

---

## Configuration (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | required | Anthropic API key |
| `DATABASE_URL` | `postgresql+asyncpg://...` | Async DB URL |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local embedding model |
| `EMBEDDING_DIMENSION` | `384` | Must match model output |
| `CHUNK_SIZE` | `512` | Tokens per chunk |
| `CHUNK_OVERLAP` | `64` | Token overlap between chunks |
| `TOP_K_VECTOR` | `10` | Candidates from vector search |
| `TOP_K_BM25` | `10` | Candidates from BM25 |
| `TOP_K_FINAL` | `6` | Final chunks after RRF |
| `RRF_K` | `60` | RRF fusion constant |
| `LLM_MODEL` | `claude-sonnet-4-20250514` | Generation model |
| `TEMPERATURE` | `0.1` | LLM temperature |

---

## Retrieval Pipeline

```
Query
  │
  ├─── embed_query() ──────▶ vector_search()  ──┐
  │     (all-MiniLM-L6-v2)   (pgvector HNSW)    │
  │                                              ├── RRF Fusion
  └─── tokenize() ─────────▶ bm25_search()   ──┘
        (whitespace)          (rank-bm25)        │
                                                 │
                            [optional] rerank() ─┘
                             (cross-encoder/ms-marco)
                                                 │
                                           Claude Sonnet
                                           (SSE streaming)
```

---

## Embedding Models

| Model | Dim | Speed | Quality | Use case |
|-------|-----|-------|---------|----------|
| `all-MiniLM-L6-v2` | 384 | ⚡ Fast | Good | Default — balanced |
| `all-mpnet-base-v2` | 768 | Moderate | Better | Higher quality |
| `multi-qa-MiniLM-L6-cos-v1` | 384 | Fast | Good | Q&A focused |

Change `EMBEDDING_MODEL` + `EMBEDDING_DIMENSION` in `.env` and re-ingest.

---

## RAGAS Metrics

| Metric | What it measures | Needs ground truth? |
|--------|-----------------|---------------------|
| Faithfulness | Are all claims in the answer backed by context? | No |
| Answer Relevance | Does the answer actually address the question? | No |
| Context Precision | What fraction of retrieved chunks is useful? | No |
| Context Recall | Does the context cover the ground truth? | Yes |

Evaluation uses Claude Haiku as the judge (fast + low cost).
