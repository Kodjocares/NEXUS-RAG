from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging, sys
from contextlib import asynccontextmanager
from config import get_settings
from database import init_db
from api.ingest import router as ingest_router
from api.query import router as query_router
from api.documents import router as documents_router
from api.eval import router as eval_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("NEXUS RAG starting up...")
    await init_db()
    logger.info("Database initialized.")
    yield
    logger.info("NEXUS RAG shutting down.")


app = FastAPI(
    title="NEXUS RAG API",
    description="Hybrid RAG system — Cybersecurity × Business × General",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest_router)
app.include_router(query_router)
app.include_router(documents_router)
app.include_router(eval_router)


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0", "system": "NEXUS RAG"}


@app.get("/")
async def root():
    return {
        "system": "NEXUS RAG",
        "endpoints": {
            "ingest_file": "POST /ingest/file",
            "ingest_url": "POST /ingest/url",
            "query_stream": "POST /query/stream",
            "query_sync": "POST /query/sync",
            "documents": "GET /documents/",
            "stats": "GET /documents/stats",
            "docs": "/docs",
        }
    }
