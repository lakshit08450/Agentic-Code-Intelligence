"""
PRISM — Agentic Code Intelligence: Live Retrieval Demo
=====================================================
Gradio app that loads CoIR-Retrieval/apps (test split) via the MTEB API,
encodes the corpus using CodeLens's Embedder (all-MiniLM-L6-v2),
and serves a live search UI with syntax-highlighted code results.

The MTEB API is used to ensure identical data loading as the official
evaluation — same dataset, same revision, same split.

Usage:
    cd demo
    pip install gradio mteb
    python app.py
"""

import sys
import os
import time
import json
import hashlib

import numpy as np
import gradio as gr

# ── Add backend to path so we can import the Embedder ────────────
BACKEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend")
sys.path.insert(0, BACKEND_DIR)

from app.indexer.embedder import Embedder
from app.config import EMBEDDING_MODEL_NAME, EMBEDDING_DIM

# ── Constants ────────────────────────────────────────────────────
MAX_RESULTS = 20
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Known dataset revision from mteb source
DATASET_REVISION = "f22508f96b7a36c2415181ed8bb76f76e04ae2d5"

# ── Results paths (optional — for the official scores panel) ─────
RESULTS_CANDIDATES = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "appsretrieval_results.json"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval", "results", "eval_results.json"),
]


# ════════════════════════════════════════════════════════════════
#  1. LOAD DATASET VIA MTEB (same source as official evaluation)
# ════════════════════════════════════════════════════════════════
print("=" * 60)
print("  PRISM Demo - Loading AppsRetrieval via MTEB")
print("=" * 60)

import mteb

print("  Loading AppsRetrieval task + test split from HuggingFace...")
task = mteb.get_tasks(tasks=["AppsRetrieval"])[0]
task.load_data()
test_data = task.dataset["default"]["test"]

# test_data is a dict with keys: corpus, queries, relevant_docs, top_ranked
# corpus and queries are HF Dataset objects with columns: id, text, language, ...
# relevant_docs is a dict: {query_id: {doc_id: relevance_score}}

corpus_ds = test_data["corpus"]       # HF Dataset: id, text, ...
queries_ds = test_data["queries"]     # HF Dataset: id, text, ...
qrels = test_data["relevant_docs"]    # dict: {qid: {did: score}}

# Build lookup dicts from HF Datasets
queries = {}  # id -> text
for row in queries_ds:
    queries[row["id"]] = row["text"]

corpus = {}   # id -> text
corpus_ids = []
corpus_texts = []
for row in corpus_ds:
    doc_id = row["id"]
    doc_text = row["text"]
    corpus[doc_id] = doc_text
    corpus_ids.append(doc_id)
    corpus_texts.append(doc_text)

print(f"  [OK] Loaded {len(queries):,} queries, {len(corpus):,} corpus docs")
print(f"  [OK] {len(qrels):,} query-relevance entries")
print(f"  [OK] Dataset revision: {DATASET_REVISION[:12]}...")


# ════════════════════════════════════════════════════════════════
#  2. LOAD MODEL & ENCODE CORPUS (with disk cache)
# ════════════════════════════════════════════════════════════════
print("\n  Loading embedding model...")
embedder = Embedder(model_name=EMBEDDING_MODEL_NAME)

# Cache key based on corpus hash + model
corpus_hash = hashlib.md5(
    json.dumps(corpus_ids[:50] + [str(len(corpus_ids))]).encode()
).hexdigest()[:12]
cache_file = os.path.join(CACHE_DIR, f"corpus_vecs_{corpus_hash}_{EMBEDDING_DIM}.npy")

if os.path.exists(cache_file):
    print(f"  Loading cached corpus embeddings...")
    corpus_vecs = np.load(cache_file)
    print(f"  [OK] Loaded {corpus_vecs.shape[0]:,} vectors from cache")
else:
    print(f"  Encoding {len(corpus_texts):,} corpus documents...")
    print(f"  (First run may take 5-15 minutes on CPU; subsequent runs use cache)")
    start_enc = time.time()
    corpus_vecs = embedder.embed_batch(corpus_texts, batch_size=64)
    enc_time = time.time() - start_enc
    print(f"  [OK] Encoded {corpus_vecs.shape[0]:,} docs in {enc_time:.1f}s")
    np.save(cache_file, corpus_vecs)
    print(f"  [OK] Saved embeddings cache")

# Normalize for cosine similarity
norms = np.linalg.norm(corpus_vecs, axis=1, keepdims=True)
norms[norms == 0] = 1.0
corpus_vecs = (corpus_vecs / norms).astype(np.float32)


# ════════════════════════════════════════════════════════════════
#  3. LOAD OFFICIAL SCORES (optional)
# ════════════════════════════════════════════════════════════════
official_scores = {}

for rpath in RESULTS_CANDIDATES:
    if os.path.exists(rpath):
        try:
            with open(rpath) as f:
                official = json.load(f)
            if "scores" in official:
                official_scores = official["scores"].get("test", [{}])[0]
            elif "ndcg@10" in official:
                official_scores = official
            print(f"  [OK] Loaded official results from {rpath}")
        except Exception as e:
            print(f"  [WARN] Could not load {rpath}: {e}")
        break


# ════════════════════════════════════════════════════════════════
#  4. PREPARE EXAMPLE QUERIES
# ════════════════════════════════════════════════════════════════
example_qids = []
for qid in list(qrels.keys()):
    q_text = queries.get(qid, "")
    if 100 < len(q_text) < 1200 and len(qrels[qid]) > 0:
        example_qids.append(qid)
    if len(example_qids) >= 8:
        break

if len(example_qids) < 3:
    example_qids = list(queries.keys())[:8]

example_choices = []
example_lookup = {}
for qid in example_qids:
    q_text = queries[qid]
    preview = q_text[:80].replace("\n", " ").replace("\r", "").strip()
    label = f"[{qid}] {preview}..."
    example_choices.append(label)
    example_lookup[label] = q_text

print(f"  [OK] {len(example_choices)} example queries prepared")


# ════════════════════════════════════════════════════════════════
#  5. SEARCH + GROUND-TRUTH FUNCTIONS
# ════════════════════════════════════════════════════════════════
def search(query_text: str, top_k: int):
    """Encode query, compute cosine similarity, return top-k results."""
    if not query_text or not query_text.strip():
        return "⚠️ Enter a query.", []

    start = time.time()

    # Encode with the asymmetric "search:" prefix
    q_vec = embedder.embed_query(query_text)
    q_vec = q_vec.reshape(1, -1).astype(np.float32)
    q_norm = np.linalg.norm(q_vec)
    if q_norm > 0:
        q_vec = q_vec / q_norm

    # Cosine similarity (corpus already L2-normalized)
    sims = (q_vec @ corpus_vecs.T)[0]
    top_idx = np.argsort(-sims)[:int(top_k)]

    latency_ms = (time.time() - start) * 1000

    # Check if this query matches a known benchmark query — if so, find ground truth
    gt_doc_ids = set()
    for qid, q_text in queries.items():
        if q_text.strip() == query_text.strip() and qid in qrels:
            gt_doc_ids = set(qrels[qid].keys())
            break

    status = (
        f"✅ Retrieved top **{int(top_k)}** results in **{latency_ms:.1f} ms** · "
        f"Corpus: {len(corpus_ids):,} docs · Model: `{EMBEDDING_MODEL_NAME}`"
    )
    if gt_doc_ids:
        status += f" · 🎯 Ground truth available ({len(gt_doc_ids)} relevant doc{'s' if len(gt_doc_ids) > 1 else ''})"

    outputs = []
    for rank, idx in enumerate(top_idx, 1):
        doc_id = corpus_ids[idx]
        score = float(sims[idx])
        code = corpus_texts[idx]
        is_relevant = doc_id in gt_doc_ids
        outputs.append((rank, doc_id, score, code, is_relevant))

    return status, outputs


# ════════════════════════════════════════════════════════════════
#  6. GRADIO UI — PREMIUM DARK THEME
# ════════════════════════════════════════════════════════════════

# Build info panel markdown
ndcg = official_scores.get("ndcg@10", official_scores.get("ndcg_at_10", "—"))
mrr_val = official_scores.get("mrr", official_scores.get("mrr_at_10", "—"))
p5 = official_scores.get("precision@5", "—")
r20 = official_scores.get("recall@20", "—")

info_md = f"""
### 📊 Dataset Info
| | |
|---|---|
| **Source** | `CoIR-Retrieval/apps` |
| **Split** | test |
| **Revision** | `{DATASET_REVISION[:12]}…` |
| **Queries** | {len(queries):,} |
| **Corpus** | {len(corpus):,} |
| **Model** | `{EMBEDDING_MODEL_NAME}` |
| **Dim** | {EMBEDDING_DIM} |

### 🏆 Official Scores
| Metric | Score |
|--------|-------|
| NDCG@10 | **{ndcg}** |
| MRR | **{mrr_val}** |
| Precision@5 | **{p5}** |
| Recall@20 | **{r20}** |
"""

custom_css = """
/* ── Import premium fonts ────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Global overrides ────────────────────────────────── */
.gradio-container {
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    max-width: 1440px !important;
    background: #050510 !important;
}

/* ── Hero header ─────────────────────────────────────── */
#hero-header {
    background: linear-gradient(135deg, #0c0c24 0%, #12122e 40%, #1a0f33 70%, #0d0d26 100%);
    border: 1px solid rgba(99, 102, 241, 0.12);
    border-radius: 20px;
    padding: 32px 40px;
    margin-bottom: 16px;
    position: relative;
    overflow: hidden;
    box-shadow:
        0 0 60px rgba(99, 102, 241, 0.06),
        0 4px 24px rgba(0, 0, 0, 0.4);
}
#hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(99, 102, 241, 0.08) 0%, transparent 70%);
    pointer-events: none;
}
#hero-header h1 {
    color: #e0e7ff !important;
    font-size: 30px !important;
    font-weight: 800 !important;
    letter-spacing: -0.8px;
    margin: 0 0 6px 0 !important;
    background: linear-gradient(135deg, #e0e7ff, #a5b4fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
#hero-header p {
    color: #64748b !important;
    font-size: 14px !important;
    margin: 0 !important;
    font-weight: 400;
}
#hero-header .badge {
    display: inline-block;
    background: rgba(99, 102, 241, 0.15);
    color: #a5b4fc;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin-top: 8px;
}

/* ── Info panel ──────────────────────────────────────── */
.info-card {
    background: linear-gradient(160deg, #0e1225 0%, #0a0f1e 100%) !important;
    border: 1px solid rgba(99, 102, 241, 0.12) !important;
    border-radius: 16px !important;
    padding: 20px 24px !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.2) !important;
}
.info-card h3 {
    color: #818cf8 !important;
    font-size: 13px !important;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 700;
    margin-top: 14px !important;
    margin-bottom: 8px !important;
}
.info-card table {
    width: 100%;
    border-collapse: collapse;
}
.info-card td {
    padding: 3px 6px !important;
    color: #94a3b8 !important;
    font-size: 12.5px !important;
    border: none !important;
}
.info-card strong {
    color: #c7d2fe !important;
}
.info-card code {
    background: rgba(99, 102, 241, 0.1) !important;
    color: #a5b4fc !important;
    padding: 1px 6px !important;
    border-radius: 4px !important;
    font-size: 11px !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── Search button ───────────────────────────────────── */
.search-btn {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a78bfa 100%) !important;
    border: none !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    border-radius: 12px !important;
    padding: 14px 28px !important;
    letter-spacing: 0.3px;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow:
        0 4px 16px rgba(99, 102, 241, 0.25),
        0 0 0 1px rgba(99, 102, 241, 0.1) !important;
}
.search-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow:
        0 8px 28px rgba(99, 102, 241, 0.4),
        0 0 0 1px rgba(99, 102, 241, 0.2) !important;
}
.search-btn:active {
    transform: translateY(0) !important;
}

/* ── Status bar ──────────────────────────────────────── */
.status-bar p {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.06), rgba(16, 185, 129, 0.02)) !important;
    border: 1px solid rgba(16, 185, 129, 0.15) !important;
    border-radius: 10px !important;
    padding: 10px 16px !important;
    color: #6ee7b7 !important;
    font-size: 13px !important;
    font-weight: 500;
}
.status-bar code {
    background: rgba(16, 185, 129, 0.1) !important;
    color: #a7f3d0 !important;
    padding: 1px 5px !important;
    border-radius: 3px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
}

/* ── Result header cards ─────────────────────────────── */
.result-hdr {
    margin-top: 10px !important;
}
.result-hdr p {
    background: linear-gradient(135deg, #111827 0%, #1a1f35 100%) !important;
    border: 1px solid rgba(99, 102, 241, 0.12) !important;
    border-bottom: none !important;
    border-radius: 12px 12px 0 0 !important;
    padding: 10px 16px !important;
    font-size: 13px !important;
    color: #94a3b8 !important;
    font-weight: 500;
    margin: 0 !important;
}
.result-hdr strong {
    color: #e0e7ff !important;
}
.result-hdr code {
    background: rgba(99, 102, 241, 0.1) !important;
    color: #a5b4fc !important;
    padding: 1px 5px !important;
    border-radius: 3px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11.5px !important;
}

/* Ground-truth hit highlight */
.result-hdr-hit p {
    background: linear-gradient(135deg, #052e16 0%, #0a2e1f 100%) !important;
    border-color: rgba(16, 185, 129, 0.25) !important;
    border-bottom: none !important;
    border-radius: 12px 12px 0 0 !important;
    padding: 10px 16px !important;
}

/* ── Code blocks ─────────────────────────────────────── */
.code-blk {
    border-radius: 0 0 12px 12px !important;
    border: 1px solid rgba(99, 102, 241, 0.1) !important;
    border-top: none !important;
    margin-bottom: 2px !important;
    overflow: hidden;
}
.code-blk textarea, .code-blk pre, .code-blk code {
    font-size: 12px !important;
    line-height: 1.55 !important;
    font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace !important;
    background: #0d1117 !important;
}

/* ── Input styling ───────────────────────────────────── */
.gradio-container textarea {
    font-family: 'Inter', system-ui, sans-serif !important;
    background: #0e1225 !important;
    border-color: rgba(99, 102, 241, 0.15) !important;
    color: #e2e8f0 !important;
    border-radius: 10px !important;
    transition: border-color 0.2s ease !important;
}
.gradio-container textarea:focus {
    border-color: rgba(99, 102, 241, 0.4) !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.08) !important;
}

/* ── Results heading ─────────────────────────────────── */
.results-heading h3 {
    color: #818cf8 !important;
    font-size: 15px !important;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    font-weight: 700;
}
"""

with gr.Blocks(
    title="PRISM — Live Code Retrieval Demo",
    theme=gr.themes.Base(
        primary_hue="indigo",
        secondary_hue="violet",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ).set(
        body_background_fill="#050510",
        body_background_fill_dark="#050510",
        block_background_fill="#0a0f1e",
        block_background_fill_dark="#0a0f1e",
        block_border_color="rgba(99, 102, 241, 0.1)",
        block_border_color_dark="rgba(99, 102, 241, 0.1)",
        block_label_text_color="#64748b",
        block_label_text_color_dark="#64748b",
        block_title_text_color="#c7d2fe",
        block_title_text_color_dark="#c7d2fe",
        body_text_color="#e2e8f0",
        body_text_color_dark="#e2e8f0",
        border_color_accent="rgba(99, 102, 241, 0.2)",
        border_color_accent_dark="rgba(99, 102, 241, 0.2)",
        button_primary_background_fill="linear-gradient(135deg, #6366f1, #8b5cf6)",
        button_primary_background_fill_dark="linear-gradient(135deg, #6366f1, #8b5cf6)",
        button_primary_text_color="white",
        input_background_fill="#0e1225",
        input_background_fill_dark="#0e1225",
        input_border_color="rgba(99, 102, 241, 0.15)",
        input_border_color_dark="rgba(99, 102, 241, 0.15)",
    ),
    css=custom_css,
) as demo:

    # ── Hero Header ──────────────────────────────────
    with gr.Column(elem_id="hero-header"):
        gr.HTML(f"""
            <h1>🔬 PRISM — Agentic Code Intelligence</h1>
            <p>Live retrieval demo · CoIR-Retrieval/apps benchmark · Samsung PRISM GenAI Hackathon Y2026</p>
            <span class="badge">MTEB · {len(queries):,} queries · {len(corpus):,} corpus docs · revision {DATASET_REVISION[:8]}</span>
        """)

    # ── Main layout ──────────────────────────────────
    with gr.Row():
        # Left panel — controls + info
        with gr.Column(scale=1, min_width=360):
            gr.Markdown(info_md, elem_classes=["info-card"])

            example_dropdown = gr.Dropdown(
                choices=example_choices,
                label="🔖 Try a benchmark query",
                info="Real queries from the CoIR-Retrieval/apps test split",
            )

            query_box = gr.Textbox(
                lines=8,
                label="🔍 Query",
                placeholder="Describe the programming problem to find a solution for...\n\nExample: Given n integers, find the maximum subarray sum.",
                info="Natural-language problem description — same format used in the official evaluation",
            )

            top_k_slider = gr.Slider(
                1, MAX_RESULTS, value=10, step=1,
                label="Top K results",
            )

            search_btn = gr.Button(
                "⚡ Search",
                variant="primary",
                elem_classes=["search-btn"],
            )

            status_box = gr.Markdown(
                value="Ready. Select an example query or type your own.",
                elem_classes=["status-bar"],
            )

        # Right panel — results
        with gr.Column(scale=2):
            gr.Markdown("### 📋 Retrieved Code Snippets", elem_classes=["results-heading"])

            result_blocks = []
            for i in range(MAX_RESULTS):
                header = gr.Markdown(
                    visible=False,
                    elem_classes=["result-hdr"],
                )
                code = gr.Code(
                    language="python",
                    visible=False,
                    elem_classes=["code-blk"],
                    label=f"Result {i+1}",
                    show_label=False,
                )
                result_blocks.append((header, code))

    # ── Event handlers ───────────────────────────────
    def on_example_select(choice):
        return example_lookup.get(choice, "")

    example_dropdown.change(
        on_example_select,
        inputs=example_dropdown,
        outputs=query_box,
    )

    def on_search(q, k):
        status, outputs = search(q, k)
        updates = []
        for i in range(MAX_RESULTS):
            if i < len(outputs):
                rank, doc_id, score, code_text, is_relevant = outputs[i]

                # Score badge
                if is_relevant:
                    badge = "🎯"
                elif score >= 0.7:
                    badge = "🟢"
                elif score >= 0.4:
                    badge = "🟡"
                else:
                    badge = "🔴"

                lines = len(code_text.splitlines())
                gt_label = " · **✓ GROUND TRUTH**" if is_relevant else ""

                header_text = (
                    f"**{badge} #{rank}** · `{doc_id}` · "
                    f"similarity: **{score:.4f}** ({score*100:.1f}%) · "
                    f"{lines} lines{gt_label}"
                )
                updates.append(gr.update(value=header_text, visible=True))
                updates.append(gr.update(value=code_text, visible=True))
            else:
                updates.append(gr.update(visible=False))
                updates.append(gr.update(visible=False))
        return [status] + updates

    flat_outputs = [status_box]
    for header, code in result_blocks:
        flat_outputs += [header, code]

    search_btn.click(
        on_search,
        inputs=[query_box, top_k_slider],
        outputs=flat_outputs,
    )

    # Enter key triggers search
    query_box.submit(
        on_search,
        inputs=[query_box, top_k_slider],
        outputs=flat_outputs,
    )


# ════════════════════════════════════════════════════════════════
#  7. LAUNCH
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Launching Gradio demo at http://127.0.0.1:7860")
    print("=" * 60 + "\n")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )
