# 🧠 CodeLens — Agentic Code Intelligence

> Natural-language code search and retrieval with an agentic pipeline.
> Samsung PRISM GenAI Hackathon Y2026

---

## ⚠️ Important Note About the Model
**The final trained retrieval model is not included in this repository yet.** 
The backend contains a clean model integration interface (`backend/app/agent/model_adapter.py`) and is currently using a lightweight, mocked numpy implementation to allow the frontend and backend integration to function instantly. It can be effortlessly connected to the final model when provided.

---

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
