"""
CodeLens — FastAPI Application
Main entry point for the backend API.
"""

import uuid
import time
import asyncio
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from app.config import CORS_ORIGINS, HOST, PORT
from app.storage.schemas import (
    QueryRequest, QueryResponse, IndexRequest, IndexStatus,
    SearchResult, AgentTraceStep,
)
from app.storage.sqlite_store import SQLiteStore
from app.storage.faiss_store import FAISSStore
from app.agent.planner import Planner
from app.agent.searcher import Searcher
from app.agent.reader import Reader
from app.agent.refiner import Refiner
from app.indexer.indexer import get_indexer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("codelens")

# ── In-memory state ───────────────────────────────────
# Stores agent trace events for SSE streaming
_query_traces: dict[str, list[dict]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup & shutdown."""
    logger.info("CodeLens starting up...")
    yield
    logger.info("CodeLens shutting down...")


app = FastAPI(
    title="CodeLens",
    description="Agentic Code Intelligence — retrieve and rank code with natural language",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "codelens"}


# ── Search ─────────────────────────────────────────────
@app.post("/api/search", response_model=QueryResponse)
async def query(req: QueryRequest):
    """
    Execute the full agentic pipeline:
    Plan → Search → Read → Refine
    """
    query_id = str(uuid.uuid4())[:8]
    total_start = time.time()
    trace_steps: list[AgentTraceStep] = []

    indexer = get_indexer()

    # Get FAISS store for the requested version
    # We need a repo_path — for now, check the latest indexed repo
    sqlite = SQLiteStore()
    # Find any indexed repo
    conn = sqlite._get_conn()
    row = conn.execute(
        "SELECT DISTINCT repo_path FROM chunks WHERE version = ? LIMIT 1",
        (req.version,)
    ).fetchone()
    conn.close()

    if not row:
        # Try without version filter
        conn = sqlite._get_conn()
        row = conn.execute("SELECT DISTINCT repo_path, version FROM chunks LIMIT 1").fetchone()
        conn.close()

        if not row:
            return QueryResponse(
                query_id=query_id,
                query=req.query,
                results=[],
                trace=[AgentTraceStep(
                    phase="error",
                    detail="No indexed repository found. Please index a repository first.",
                    duration_ms=0,
                )],
                total_ms=0,
            )

        repo_path = row["repo_path"]
        version = row["version"]
    else:
        repo_path = row["repo_path"]
        version = req.version

    faiss_store = indexer.get_faiss_store(repo_path, version)

    # ── PLAN ──────────────────────────────────────────
    planner = Planner()
    sub_queries, plan_trace = planner.plan(req.query, req.mode)
    trace_steps.append(plan_trace)

    # ── SEARCH ────────────────────────────────────────
    searcher = Searcher(faiss_store, sqlite)
    candidates, search_trace = searcher.search(sub_queries, version=version)
    trace_steps.append(search_trace)

    # ── READ ──────────────────────────────────────────
    reader = Reader()
    enriched, read_trace = reader.read(candidates)
    trace_steps.append(read_trace)

    # ── REFINE ────────────────────────────────────────
    refiner = Refiner(use_cross_encoder=False)  # CPU-friendly default
    results, refine_trace = refiner.refine(
        query=req.query,
        candidates=enriched,
        max_results=req.max_results,
    )
    trace_steps.append(refine_trace)

    total_ms = (time.time() - total_start) * 1000

    # Store trace for SSE streaming
    _query_traces[query_id] = [t.model_dump() for t in trace_steps]

    return QueryResponse(
        query_id=query_id,
        query=req.query,
        results=results,
        trace=trace_steps if req.include_trace else [],
        total_ms=round(total_ms, 2),
    )


# ── SSE Stream for Agent Trace ────────────────────────
@app.get("/api/query/{query_id}/stream")
async def query_stream(query_id: str):
    """Stream agent trace events via SSE."""
    async def event_generator():
        trace = _query_traces.get(query_id, [])
        for step in trace:
            yield {"event": "trace", "data": str(step)}
        yield {"event": "done", "data": "complete"}

    return EventSourceResponse(event_generator())


# ── Indexing ──────────────────────────────────────────
from fastapi import UploadFile, Form, File
import shutil
import os

@app.post("/api/index_upload")
async def index_upload(background_tasks: BackgroundTasks, files: list[UploadFile] = File(...), paths: list[str] = Form(...)):
    """Accepts uploaded repository files and indexes them."""
    # Create a unique upload directory
    import uuid
    repo_name = f"uploaded_repo_{uuid.uuid4().hex[:8]}"
    
    # Place uploads in backend/uploads directory
    base_dir = Path(__file__).parent.parent / "uploads" / repo_name
    base_dir.mkdir(parents=True, exist_ok=True)
    
    for file, path in zip(files, paths):
        # path is the relative path from the user's browser, e.g., "my-repo/src/index.js"
        # We need to sanitize it to avoid directory traversal just in case
        safe_path = path.lstrip('/')
        if '..' in safe_path:
            continue
            
        file_path = base_dir / safe_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
    indexer = get_indexer()
    req_version = "main"
    
    # Determine the inner directory if there's a single top-level folder
    # webkitdirectory usually sends paths like "repo-name/src/main.js"
    # We want to index the whole thing, so we can just index base_dir.
    
    indexer.sqlite.set_index_status(str(base_dir), req_version, "pending")
    background_tasks.add_task(_run_indexing, str(base_dir), req_version, False)
    
    return {
        "status": "started",
        "repo_path": str(base_dir),
        "version": req_version,
    }
@app.post("/api/index")
async def index_repo(req: IndexRequest, background_tasks: BackgroundTasks):
    """Trigger repository indexing as a background task."""
    indexer = get_indexer()

    # Validate path exists
    repo_path = Path(req.repo_path).resolve()
    if not repo_path.exists():
        raise HTTPException(status_code=400, detail=f"Path does not exist: {req.repo_path}")

    # Set initial status
    indexer.sqlite.set_index_status(str(repo_path), req.version, "pending")

    # Run indexing in background
    background_tasks.add_task(
        _run_indexing, str(repo_path), req.version, req.force_reindex
    )

    return {
        "status": "started",
        "repo_path": str(repo_path),
        "version": req.version,
    }


def _run_indexing(repo_path: str, version: str, force: bool):
    """Background indexing task."""
    indexer = get_indexer()
    try:
        result = indexer.index_repository(
            repo_path=repo_path,
            version=version,
            force_reindex=force,
        )
        logger.info(f"Indexing complete: {result}")
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        indexer.sqlite.set_index_status(repo_path, version, "error", error_message=str(e))


@app.get("/api/index/status")
async def index_status(repo_path: str, version: str = "main"):
    """Check indexing status."""
    indexer = get_indexer()
    status = indexer.get_index_status(repo_path, version)
    if not status:
        raise HTTPException(status_code=404, detail="No indexing job found")
    return status


# ── Versions ──────────────────────────────────────────
@app.get("/api/versions")
async def list_versions(repo_path: str):
    """List all indexed versions for a repository."""
    indexer = get_indexer()
    versions = indexer.get_versions(repo_path)
    return {"repo_path": repo_path, "versions": versions}


# ── File Content ──────────────────────────────────────
@app.get("/api/code")
async def get_file(file: str, start_line: int = 1, end_line: int = -1, version: str = "main"):
    """Fetch file content for display."""
    sqlite = SQLiteStore()
    chunks = sqlite.get_file_content(file, version)
    if not chunks:
        raise HTTPException(status_code=404, detail="File not found in index")

    # Reconstruct file content from chunks
    # Read actual file if possible
    if chunks:
        repo_path = Path(chunks[0].repo_path)
        full_path = repo_path / file
        try:
            content = full_path.read_text(encoding="utf-8", errors="replace")
            return {
                "file": file,
                "start_line": start_line,
                "end_line": end_line,
                "code": content,
            }
        except FileNotFoundError:
            pass

    # Fallback: return concatenated chunks
    content = "\n".join(c.content for c in chunks)
    return {
        "file": file,
        "start_line": start_line,
        "end_line": end_line,
        "code": content,
    }


# ── Entry point ───────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=True)
