"""Search a versioned store (plan Section 12.4).

- `search_rev` (P1): revisions alive at version R, windows collapsed to units, top-k.
- `search_range` (Bonus): all revisions alive in A..B, one result per unit with its best revision
  (ties -> most recent), versions matched, and the unit's history in the range.
Scores are cosine similarities of query and snippet embeddings only (R1); optional `verify` re-ranks
with Stage 2 when the query carries parseable sample I/O.
"""

from __future__ import annotations

import time

import numpy as np

from codeintel.versioned.store import Store


def _encode(store: Store, query: str) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    q = store.encoder.embed([query], is_query=True)[0]
    return q, 1000 * (time.perf_counter() - t0)


def _verify(query: str, results: list[dict], store: Store) -> float:
    """Stage 2 re-rank of the returned units (only if the query has stdin samples)."""
    from codeintel.stage2.reranker import Stage2Cfg, stage2_score, load_stage2_cfg
    from codeintel.stage2.sandbox import ExecCache
    from codeintel.stage2.verifier import samples_for, verify

    t0 = time.perf_counter()
    if samples_for(query).kind != "stdin":
        return 0.0
    try:
        cfg = load_stage2_cfg("configs/stage2_final.yaml")
    except FileNotFoundError:
        cfg = Stage2Cfg()
    cache = ExecCache()
    for r in results:
        v = verify(query, r["snippet"], cache)
        r["verify"] = v.outcome
        r["score"] = stage2_score(r["score"], v.outcome, query, cfg)
    results.sort(key=lambda r: -r["score"])
    for i, r in enumerate(results, 1):
        r["rank"] = i
    return 1000 * (time.perf_counter() - t0)


def search_rev(store: Store, query: str, rev: str | int, k: int = 10, verify: bool = False) -> dict:
    n = store.ordinal(rev)
    q, t_enc = _encode(store, query)
    t0 = time.perf_counter()
    rows = store.alive_at(n)
    if not rows:
        return {"rev": n, "results": [], "ms": {"encode": t_enc, "search": 0.0}}
    vec_rows = np.array([r[5] for r in rows])
    scores = store.vectors[vec_rows] @ q
    best: dict[int, int] = {}
    for i, r in enumerate(rows):
        uid = r[1]
        if uid not in best or scores[i] > scores[best[uid]]:
            best[uid] = i
    top = sorted(best.values(), key=lambda i: -scores[i])[:k]
    results = [
        {"rank": j + 1, "unit_id": rows[i][1], "path": rows[i][2], "lines": [rows[i][3], rows[i][4]],
         "score": float(scores[i]), "version": {"from": rows[i][6], "to": rows[i][7]}, "snippet": store.snippet(rows[i][8])}
        for j, i in enumerate(top)
    ]
    t_search = 1000 * (time.perf_counter() - t0)
    t_ver = _verify(query, results, store) if verify else 0.0
    labels = {o: l for o, l, _ in store.versions()}
    return {"rev": labels.get(n, n), "results": results, "ms": {"encode": t_enc, "search": t_search, "verify": t_ver}}


def search_range(store: Store, query: str, a: str | int, b: str | int, k: int = 10, verify: bool = False) -> dict:
    lo, hi = store.ordinal(a), store.ordinal(b)
    q, t_enc = _encode(store, query)
    t0 = time.perf_counter()
    rows = store.alive_in_range(lo, hi)
    if not rows:
        return {"range": [lo, hi], "results": [], "ms": {"encode": t_enc, "search": 0.0}}
    scores = store.vectors[np.array([r[5] for r in rows])] @ q
    per_unit: dict[int, list[int]] = {}
    for i, r in enumerate(rows):
        per_unit.setdefault(r[1], []).append(i)
    labels = {o: l for o, l, _ in store.versions()}

    def best_idx(idxs):  # best score; ties -> most recent revision
        return max(idxs, key=lambda i: (round(float(scores[i]), 6), rows[i][6]))

    ranked = sorted(per_unit, key=lambda u: -scores[best_idx(per_unit[u])])[:k]
    results = []
    for j, uid in enumerate(ranked, 1):
        idxs = per_unit[uid]
        bi = best_idx(idxs)
        # revision history inside [lo, hi]: group windows by v_from (one revision = one v_from)
        revs: dict[int, dict] = {}
        for i in idxs:
            vf, vt = rows[i][6], rows[i][7]
            e = revs.setdefault(vf, {"v_from": vf, "v_to": vt, "keys": set(), "score": -1.0})
            e["keys"].add(rows[i][8])
            e["score"] = max(e["score"], float(scores[i]))
        history, prev = [], None
        for vf in sorted(revs):
            e = revs[vf]
            status = "introduced" if prev is None and vf > lo else ("present" if prev is None else "modified")
            end = e["v_to"] if e["v_to"] is not None else None
            history.append({
                "status": status, "from": labels.get(max(vf, lo), max(vf, lo)),
                "to": labels.get(min(end, hi), min(end, hi)) if end is not None else labels.get(hi, hi),
                "score": e["score"],
            })
            prev = e
        last = revs[max(revs)]
        if last["v_to"] is not None and last["v_to"] < hi:
            history.append({"status": "removed", "at": labels.get(last["v_to"] + 1, last["v_to"] + 1)})
        matched = sorted({labels.get(v, v) for i in idxs for v in range(max(rows[i][6], lo), min(rows[i][7] if rows[i][7] is not None else hi, hi) + 1)},
                         key=lambda x: store.ordinal(x))
        results.append({
            "rank": j, "unit_id": uid, "path": rows[bi][2], "lines": [rows[bi][3], rows[bi][4]], "score": float(scores[bi]),
            "best_revision": {"from": labels.get(rows[bi][6], rows[bi][6]), "to": labels.get(rows[bi][7], rows[bi][7])},
            "versions_matched": matched, "history": history, "snippet": store.snippet(rows[bi][8]),
        })
    t_search = 1000 * (time.perf_counter() - t0)
    t_ver = _verify(query, results, store) if verify else 0.0
    return {"range": [labels.get(lo, lo), labels.get(hi, hi)], "results": results,
            "ms": {"encode": t_enc, "search": t_search, "verify": t_ver}}
