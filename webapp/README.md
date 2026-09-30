# 🧠 CodeLens — Agentic Code Intelligence

> Natural-language code search and retrieval with an agentic pipeline.
> Samsung PRISM GenAI Hackathon Y2026

---

## Retrieval model (PRISM)
CodeLens uses the PRISM pipeline from this repository: **pretrained embedding models (Qwen3-Embedding-0.6B +
EmbeddingGemma-300m, fused) with execution-based re-ranking**. Nothing is trained or fine-tuned. The PRISM
pipeline runs as its own local server (Python 3.11 venv at the repo root); CodeLens talks to it over HTTP
(`backend/app/agent/prism_client.py`, frozen API in `../docs/INTEGRATION.md`), so the two virtual
environments never share dependencies.

- **Datasets** (picker next to the search box): the APPS corpus (8,765 Python solutions) and the PRISM demo
  repository at versions v1..v4 (or all versions v1..v4, one result per file with its history).
- **Stage 2** (execution re-ranking): when the query contains sample Input/Output, the top 20 Python candidates
  are run on the sample input in PRISM's Windows sandbox; results show *passed sample tests* / *wrong output* /
  *runtime error*. Otherwise a notice says why Stage 2 did not run (results are fusion-only).
- **Benchmark vs app**: the benchmark submission executes the top 50 (fusion + Stage 2 k=50, AppsRetrieval test
  NDCG@10 0.956); the app executes the top 20 for interactivity (same configuration at k=20: 0.952).

## Start (Windows, three terminals, in this order)
```powershell
# 1. PRISM model server (repo root; 3.11 venv, demo data in demo_data/, sandbox set up) - ready in ~35 s
cd <repo root>
.venv\Scripts\python -m codeintel.app.server --port 8765
# 2. CodeLens backend (its own 3.11 venv)
cd <repo root>\webappackend
py -3.11 -m venv .venv ; .venv\Scripts\python -m pip install -r requirements.txt   # first time only
.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
# 3. Frontend (Node 18+)
cd <repo root>\webapprontend
npm install        # first time only
npm run dev        # open http://localhost:5173
```
Checks: `curl http://127.0.0.1:8765/health` shows `"ready": true`; `curl http://127.0.0.1:8000/api/model_status`
shows `"ready": true` and the datasets. `PRISM_API_URL` overrides the PRISM server address.

### Demo queries
1. **APPS with sample I/O (Stage 2 runs)**: the "APPS problem with sample I/O" example button (a+b with two
   samples): the top results are executed and marked *passed sample tests*.
2. **Plain**: "Given a string, reverse it and print the reversed string." (APPS) - fusion-only notice.
3. **Version**: dataset "Demo repo @ v2", query "parse sample input and output pairs from a problem statement".
4. **All versions**: dataset "Demo repo, all versions v1..v4", query "run an untrusted program in a sandbox
   with a time limit" - one result per file with its version history.

## What is CodeLens?

CodeLens takes **natural-language queries** and returns **ranked, line-precise JavaScript/TypeScript code snippets** from large codebases. It operates through an agentic loop — **Plan → Search → Read → Refine** — and supports semantic, structural, and hybrid retrieval.

**It is not a chatbot.** It's a code research tool with a beautiful cream UI, designed specifically to locate code.

---

## Quick Start

### 1. Clone & Install

```bash
git clone <repository_url>
cd prism-app

# Backend Setup
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
pip install -r requirements.txt

# Frontend Setup
cd ../frontend
npm install
```

### 2. Run

```bash
# Terminal 1 — Backend
cd backend
venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000

# Terminal 2 — Frontend
cd frontend
npm run dev
```

Open **http://localhost:5173**

### 3. Index a Repository

1. Click **"Index Repo"** in the top-right header of the UI.
2. Your operating system's native folder browser will open.
3. Select the local folder/repository you want to index.
4. Wait for the upload and indexing completion status.
5. Start searching! (e.g., "Where is authentication handled?")

---

## Architecture

```
User Query
    │
    ▼
┌─ PLANNER ─┐   Decomposes NL query into sub-queries
│           │   (semantic + structural)
└────┬──────┘
     │
     ▼
┌─ SEARCHER ─┐   Semantic: RetrievalModel Adapter
│            │   Structural: SQLite AST index
│            │   Merged via Reciprocal Rank Fusion
└────┬───────┘
     │
     ▼
  Ranked Results
```

### Tech Stack

| Component | Technology |
|---|---|
| Frontend | HTML, Vanilla JavaScript, CSS |
| Backend | Python 3.13, FastAPI |
| Vector Search | FAISS |
| AST Parsing | Tree-sitter (JavaScript) |
| Metadata | SQLite |

---

## API Reference

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/search` | Submit search query with mode parameter |
| `POST` | `/api/index_upload` | Upload a directory for indexing |
| `GET` | `/api/index/status` | Check indexing progress |
| `GET` | `/api/code` | Fetch raw file content |

---

## Model Integration Guide (For Team)

The final trained model should be integrated into:
**`backend/app/agent/model_adapter.py`**

Currently, this file mocks embeddings using random `numpy` arrays. Once the final `.pt` or `.bin` weights are ready:
1. Update `requirements.txt` to include your heavy ML libraries (`torch`, `transformers`, etc.).
2. Modify `model_adapter.py` to load the actual model weights.
3. Replace the `search()` and `embed()` functions in the adapter to use real inferences.
4. Place model files in `backend/models/` (which is gitignored).

---

## License

MIT
