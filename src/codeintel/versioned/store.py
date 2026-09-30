"""Versioned index store (plan Sections 12.2-12.4): index.sqlite + vectors.npy.

- emb_key = sha1 of the exact Stage 1 model text, so identical content across versions, files or
  renames maps to one vector and is embedded once.
- Incremental update per version: embed only missing keys; revisions of changed or vanished units are
  closed (v_to = n - 1) and new revisions opened (v_from = n). Unchanged units keep their open revision.
- Unit identity: same key (path / JSONL id), carried across git renames; otherwise a vanished unit
  whose content (set of emb_keys) reappears under a new key; otherwise a new unit.
Only texts are embedded; IDs/paths are bookkeeping, never scoring features (R1).
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import numpy as np

from codeintel.common.hashing import sha1_text
from codeintel.stage1.encoder import PrePostPipelineEncoder
from codeintel.versioned.ingest import Version
from codeintel.versioned.units import Piece, split

SCHEMA = """
CREATE TABLE IF NOT EXISTS versions(version_id INTEGER PRIMARY KEY, ordinal INTEGER UNIQUE, label TEXT, source_ref TEXT);
CREATE TABLE IF NOT EXISTS units(unit_id INTEGER PRIMARY KEY, key TEXT, first_version INTEGER, last_version INTEGER);
CREATE TABLE IF NOT EXISTS revisions(rev_id INTEGER PRIMARY KEY, unit_id INTEGER, path TEXT, start_line INTEGER,
    end_line INTEGER, emb_key TEXT, v_from INTEGER, v_to INTEGER);
CREATE TABLE IF NOT EXISTS embeddings(emb_key TEXT PRIMARY KEY, row INTEGER);
CREATE TABLE IF NOT EXISTS snippets(emb_key TEXT PRIMARY KEY, text TEXT);
CREATE INDEX IF NOT EXISTS ix_rev_unit ON revisions(unit_id);
CREATE INDEX IF NOT EXISTS ix_rev_emb ON revisions(emb_key);
CREATE INDEX IF NOT EXISTS ix_rev_range ON revisions(v_from, v_to);
"""


class Store:
    def __init__(self, root: str | Path, encoder: PrePostPipelineEncoder) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: the app server loads stores in one thread and serves in others (calls are serialized)
        self.db = sqlite3.connect(str(self.root / "index.sqlite"), check_same_thread=False)
        self.db.executescript(SCHEMA)
        self.encoder = encoder
        vec = self.root / "vectors.npy"
        self.vectors = np.load(vec) if vec.exists() else None

    # ---------- keys and embeddings ----------
    def emb_key(self, text: str) -> str:
        parts = [self.encoder.model_texts([text], False, i)[0] for i in range(len(self.encoder.cfg.models))]
        return sha1_text("\x1f".join(parts))

    def _ensure_embeddings(self, pieces: list[Piece]) -> tuple[int, int]:
        """Embed pieces whose emb_key is not in the store yet. Returns (embedded, reused)."""
        known = {k for (k,) in self.db.execute("SELECT emb_key FROM embeddings")}
        todo: dict[str, str] = {}
        for p in pieces:
            k = self.emb_key(p.text)
            if k not in known and k not in todo:
                todo[k] = p.text
        reused = len({self.emb_key(p.text) for p in pieces}) - len(todo)
        if todo:
            new = self.encoder.embed(list(todo.values()), is_query=False)
            start = 0 if self.vectors is None else len(self.vectors)
            self.vectors = new if self.vectors is None else np.concatenate([self.vectors, new])
            self.db.executemany("INSERT INTO embeddings VALUES (?, ?)", [(k, start + i) for i, k in enumerate(todo)])
            self.db.executemany("INSERT INTO snippets VALUES (?, ?)", list(todo.items()))
            np.save(self.root / "vectors.npy", self.vectors)
        return len(todo), reused

    # ---------- versions ----------
    def n_versions(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM versions").fetchone()[0]

    def versions(self) -> list[tuple[int, str, str]]:
        return self.db.execute("SELECT ordinal, label, source_ref FROM versions ORDER BY ordinal").fetchall()

    def ordinal(self, label_or_ordinal: str | int) -> int:
        row = self.db.execute("SELECT ordinal FROM versions WHERE label = ?", (str(label_or_ordinal),)).fetchone()
        if row:
            return row[0]
        try:
            return int(label_or_ordinal)
        except ValueError:
            raise KeyError(f"unknown version {label_or_ordinal!r}") from None

    def add_version(self, v: Version) -> dict:
        t0 = time.perf_counter()
        n = self.n_versions()
        pieces_by_key = {key: split(key, path, text) for key, (path, text) in v.items.items()}
        t_split = time.perf_counter()
        embedded, reused = self._ensure_embeddings([p for ps in pieces_by_key.values() for p in ps])
        t_embed = time.perf_counter()

        alive = {}  # unit key -> (unit_id, frozenset(emb_keys))
        for uid, key in self.db.execute("SELECT unit_id, key FROM units WHERE last_version = ?", (n - 1,)):
            keys = frozenset(k for (k,) in self.db.execute(
                "SELECT emb_key FROM revisions WHERE unit_id = ? AND v_to IS NULL", (uid,)))
            alive[key] = (uid, keys)
        by_content = {keys: (key, uid) for key, (uid, keys) in alive.items()}
        claimed: set[int] = set()
        stats = {"units": len(pieces_by_key), "unchanged": 0, "modified": 0, "added": 0, "renamed_or_moved": 0, "removed": 0}

        for key, pieces in pieces_by_key.items():
            ekeys = [self.emb_key(p.text) for p in pieces]
            old_key = next((o for o, nk in v.renames.items() if nk == key), None)
            uid, prev_keys = None, None
            if key in alive and alive[key][0] not in claimed:
                uid, prev_keys = alive[key]
            elif old_key in alive and alive[old_key][0] not in claimed:
                uid, prev_keys = alive[old_key]
                stats["renamed_or_moved"] += 1
            elif frozenset(ekeys) in by_content and by_content[frozenset(ekeys)][0] not in pieces_by_key:
                uid, prev_keys = by_content[frozenset(ekeys)][1], frozenset(ekeys)
                if uid in claimed:
                    uid = None
                else:
                    stats["renamed_or_moved"] += 1
            if uid is None:
                uid = self.db.execute("INSERT INTO units(key, first_version, last_version) VALUES (?, ?, ?)", (key, n, n)).lastrowid
                stats["added"] += 1
            else:
                claimed.add(uid)
                self.db.execute("UPDATE units SET key = ?, last_version = ? WHERE unit_id = ?", (key, n, uid))
                same = prev_keys == frozenset(ekeys)
                paths_same = self.db.execute(
                    "SELECT COUNT(*) FROM revisions WHERE unit_id = ? AND v_to IS NULL AND path != ?", (uid, pieces[0].path)).fetchone()[0] == 0
                if same and paths_same:
                    stats["unchanged"] += 1
                    continue
                stats["modified"] += not same
                self.db.execute("UPDATE revisions SET v_to = ? WHERE unit_id = ? AND v_to IS NULL", (n - 1, uid))
            self.db.executemany(
                "INSERT INTO revisions(unit_id, path, start_line, end_line, emb_key, v_from, v_to) VALUES (?,?,?,?,?,?,NULL)",
                [(uid, p.path, p.start_line, p.end_line, k, n) for p, k in zip(pieces, ekeys)],
            )
        for key, (uid, _) in alive.items():
            if uid not in claimed:
                self.db.execute("UPDATE revisions SET v_to = ? WHERE unit_id = ? AND v_to IS NULL", (n - 1, uid))
                stats["removed"] += 1
        self.db.execute("INSERT INTO versions(ordinal, label, source_ref) VALUES (?, ?, ?)", (n, v.label, v.source_ref))
        self.db.commit()
        t_end = time.perf_counter()
        return stats | {
            "version": n, "label": v.label, "embedded": embedded, "reused": reused,
            "reused_share": reused / max(1, reused + embedded),
            "ms": {"split": 1000 * (t_split - t0), "embed": 1000 * (t_embed - t_split), "index": 1000 * (t_end - t_embed),
                   "total": 1000 * (t_end - t0)},
        }

    # ---------- queries ----------
    def snippet(self, emb_key: str) -> str:
        row = self.db.execute("SELECT text FROM snippets WHERE emb_key = ?", (emb_key,)).fetchone()
        return row[0] if row else ""

    def alive_at(self, n: int) -> list[tuple]:
        return self.db.execute(
            "SELECT r.rev_id, r.unit_id, r.path, r.start_line, r.end_line, e.row, r.v_from, r.v_to, r.emb_key "
            "FROM revisions r JOIN embeddings e ON e.emb_key = r.emb_key "
            "WHERE r.v_from <= ? AND (r.v_to IS NULL OR r.v_to >= ?)", (n, n)).fetchall()

    def alive_in_range(self, a: int, b: int) -> list[tuple]:
        return self.db.execute(
            "SELECT r.rev_id, r.unit_id, r.path, r.start_line, r.end_line, e.row, r.v_from, r.v_to, r.emb_key "
            "FROM revisions r JOIN embeddings e ON e.emb_key = r.emb_key "
            "WHERE r.v_from <= ? AND (r.v_to IS NULL OR r.v_to >= ?)", (b, a)).fetchall()
