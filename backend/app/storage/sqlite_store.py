"""
SQLite storage layer for CodeLens.
Stores code chunk metadata and AST structural index.
"""

import sqlite3
import json
from pathlib import Path
from typing import Optional
from app.config import DB_PATH
from app.storage.schemas import CodeChunk


class SQLiteStore:
    """Manages SQLite database for chunk metadata and AST nodes."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self):
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                content TEXT NOT NULL,
                language TEXT DEFAULT 'javascript',
                version TEXT NOT NULL,
                commit_sha TEXT NOT NULL,
                repo_path TEXT NOT NULL,
                node_type TEXT,
                node_name TEXT,
                is_exported BOOLEAN DEFAULT 0,
                is_async BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_path);
            CREATE INDEX IF NOT EXISTS idx_chunks_version ON chunks(version);
            CREATE INDEX IF NOT EXISTS idx_chunks_node_type ON chunks(node_type);
            CREATE INDEX IF NOT EXISTS idx_chunks_node_name ON chunks(node_name);
            CREATE INDEX IF NOT EXISTS idx_chunks_repo_version ON chunks(repo_path, version);

            CREATE TABLE IF NOT EXISTS index_status (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                repo_path TEXT NOT NULL,
                version TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                total_files INTEGER DEFAULT 0,
                processed_files INTEGER DEFAULT 0,
                total_chunks INTEGER DEFAULT 0,
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(repo_path, version)
            );
        """)
        conn.commit()
        conn.close()

    def upsert_chunk(self, chunk: CodeChunk):
        """Insert or update a code chunk."""
        conn = self._get_conn()
        conn.execute("""
            INSERT OR REPLACE INTO chunks
            (id, file_path, start_line, end_line, content, language, version, commit_sha,
             repo_path, node_type, node_name, is_exported, is_async)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            chunk.id, chunk.file_path, chunk.start_line, chunk.end_line,
            chunk.content, chunk.language, chunk.version, chunk.commit_sha,
            chunk.repo_path, chunk.node_type, chunk.node_name,
            chunk.is_exported, chunk.is_async
        ))
        conn.commit()
        conn.close()

    def upsert_chunks_batch(self, chunks: list[CodeChunk]):
        """Batch insert/update chunks for performance."""
        conn = self._get_conn()
        conn.executemany("""
            INSERT OR REPLACE INTO chunks
            (id, file_path, start_line, end_line, content, language, version, commit_sha,
             repo_path, node_type, node_name, is_exported, is_async)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            (c.id, c.file_path, c.start_line, c.end_line,
             c.content, c.language, c.version, c.commit_sha,
             c.repo_path, c.node_type, c.node_name,
             c.is_exported, c.is_async)
            for c in chunks
        ])
        conn.commit()
        conn.close()

    def get_chunk_by_id(self, chunk_id: str) -> Optional[CodeChunk]:
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
        conn.close()
        if row:
            return CodeChunk(**dict(row))
        return None

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[CodeChunk]:
        """Fetch multiple chunks by their IDs."""
        if not chunk_ids:
            return []
        conn = self._get_conn()
        placeholders = ",".join("?" * len(chunk_ids))
        rows = conn.execute(
            f"SELECT * FROM chunks WHERE id IN ({placeholders})", chunk_ids
        ).fetchall()
        conn.close()
        return [CodeChunk(**dict(r)) for r in rows]

    def search_structural(
        self,
        version: str,
        node_type: Optional[str] = None,
        node_name_pattern: Optional[str] = None,
        is_exported: Optional[bool] = None,
        is_async: Optional[bool] = None,
        limit: int = 30
    ) -> list[CodeChunk]:
        """
        Structural search against AST metadata.
        Supports filtering by node type, name pattern (LIKE), export/async flags.
        """
        conn = self._get_conn()
        conditions = ["version = ?"]
        params: list = [version]

        if node_type:
            conditions.append("node_type = ?")
            params.append(node_type)
        if node_name_pattern:
            conditions.append("node_name LIKE ?")
            params.append(f"%{node_name_pattern}%")
        if is_exported is not None:
            conditions.append("is_exported = ?")
            params.append(is_exported)
        if is_async is not None:
            conditions.append("is_async = ?")
            params.append(is_async)

        where = " AND ".join(conditions)
        rows = conn.execute(
            f"SELECT * FROM chunks WHERE {where} LIMIT ?",
            params + [limit]
        ).fetchall()
        conn.close()
        return [CodeChunk(**dict(r)) for r in rows]

    def get_file_content(self, file_path: str, version: str) -> list[CodeChunk]:
        """Get all chunks for a specific file and version, ordered by line number."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM chunks WHERE file_path = ? AND version = ? ORDER BY start_line",
            (file_path, version)
        ).fetchall()
        conn.close()
        return [CodeChunk(**dict(r)) for r in rows]

    def get_versions(self, repo_path: str) -> list[str]:
        """List all indexed versions for a repo."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT DISTINCT version FROM chunks WHERE repo_path = ? ORDER BY version",
            (repo_path,)
        ).fetchall()
        conn.close()
        return [r["version"] for r in rows]

    def delete_version(self, repo_path: str, version: str):
        """Remove all chunks for a specific version."""
        conn = self._get_conn()
        conn.execute(
            "DELETE FROM chunks WHERE repo_path = ? AND version = ?",
            (repo_path, version)
        )
        conn.commit()
        conn.close()

    def get_chunk_count(self, repo_path: str, version: str) -> int:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM chunks WHERE repo_path = ? AND version = ?",
            (repo_path, version)
        ).fetchone()
        conn.close()
        return row["cnt"] if row else 0

    # ── Index Status ──────────────────────────────────
    def set_index_status(self, repo_path: str, version: str, status: str,
                         total_files: int = 0, processed_files: int = 0,
                         total_chunks: int = 0, error_message: str = None):
        conn = self._get_conn()
        conn.execute("""
            INSERT INTO index_status (repo_path, version, status, total_files, processed_files, total_chunks, error_message, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(repo_path, version) DO UPDATE SET
                status=excluded.status,
                total_files=excluded.total_files,
                processed_files=excluded.processed_files,
                total_chunks=excluded.total_chunks,
                error_message=excluded.error_message,
                updated_at=CURRENT_TIMESTAMP
        """, (repo_path, version, status, total_files, processed_files, total_chunks, error_message))
        conn.commit()
        conn.close()

    def get_index_status(self, repo_path: str, version: str) -> Optional[dict]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM index_status WHERE repo_path = ? AND version = ?",
            (repo_path, version)
        ).fetchone()
        conn.close()
        return dict(row) if row else None
