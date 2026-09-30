"""
Main Indexer — orchestrates the full indexing pipeline.
Coordinates: file discovery → chunking → embedding → FAISS + SQLite storage.
"""

import time
import logging
from pathlib import Path
from typing import Optional

from app.config import SEMANTIC_TOP_K
from app.indexer.chunker import chunk_file
from app.indexer.embedder import get_embedder
from app.indexer.version_manager import VersionManager
from app.storage.sqlite_store import SQLiteStore
from app.storage.faiss_store import FAISSStore
from app.storage.schemas import CodeChunk

logger = logging.getLogger("codelens.indexer")


class Indexer:
    """
    Orchestrates the full indexing pipeline for a code repository.
    """

    def __init__(self):
        self.sqlite = SQLiteStore()
        self.faiss: Optional[FAISSStore] = None
        self.embedder = None  # Lazy-loaded

    def _ensure_embedder(self):
        if self.embedder is None:
            self.embedder = get_embedder()

    def _ensure_faiss(self, index_name: str = "default"):
        if self.faiss is None or self.faiss.index_name != index_name:
            self.faiss = FAISSStore(index_name=index_name)

    def index_repository(
        self,
        repo_path: str,
        version: str = "main",
        force_reindex: bool = False,
        progress_callback=None,
    ) -> dict:
        """
        Index a repository: discover files → chunk → embed → store.

        Args:
            repo_path: Absolute path to the repository
            version: Branch/tag to index
            force_reindex: If True, delete existing index first
            progress_callback: Optional callback(processed, total) for progress

        Returns:
            Summary dict with stats
        """
        start_time = time.time()
        self._ensure_embedder()

        # Normalize path
        repo_path = str(Path(repo_path).resolve())
        index_name = self._make_index_name(repo_path, version)
        self._ensure_faiss(index_name)

        # Update status
        self.sqlite.set_index_status(repo_path, version, "indexing")

        vm = VersionManager(repo_path)

        # Checkout version if git repo
        if vm.is_git_repo:
            try:
                commit_sha = vm.checkout_version(version)
            except Exception as e:
                commit_sha = vm.get_current_sha()
                logger.warning(f"Could not checkout {version}: {e}, using current HEAD")
        else:
            commit_sha = "no-git"

        # Handle force reindex
        if force_reindex:
            self.sqlite.delete_version(repo_path, version)
            self.faiss.clear()

        # Discover files
        js_files = vm.get_js_files()
        total_files = len(js_files)
        logger.info(f"Found {total_files} JS/TS files in {repo_path}")

        self.sqlite.set_index_status(
            repo_path, version, "indexing", total_files=total_files
        )

        # Process files in batches
        all_chunks: list[CodeChunk] = []
        all_texts: list[str] = []

        for i, rel_path in enumerate(js_files):
            content = vm.read_file(rel_path)
            if content is None:
                continue

            try:
                chunks = chunk_file(
                    file_path=rel_path,
                    content=content,
                    version=version,
                    commit_sha=commit_sha,
                    repo_path=repo_path,
                )

                for chunk in chunks:
                    all_chunks.append(chunk)
                    # Build embedding text with context
                    parts = [f"File: {chunk.file_path}"]
                    if chunk.node_name:
                        parts.append(f"Name: {chunk.node_name}")
                    parts.append(chunk.content)
                    all_texts.append("\n".join(parts))

            except Exception as e:
                logger.warning(f"Error chunking {rel_path}: {e}")

            if progress_callback and (i + 1) % 10 == 0:
                progress_callback(i + 1, total_files)

            self.sqlite.set_index_status(
                repo_path, version, "indexing",
                total_files=total_files, processed_files=i + 1
            )

        if not all_chunks:
            self.sqlite.set_index_status(
                repo_path, version, "complete",
                total_files=total_files, processed_files=total_files,
                total_chunks=0
            )
            return {"total_files": total_files, "total_chunks": 0, "duration_s": time.time() - start_time}

        # Batch embed
        logger.info(f"Embedding {len(all_chunks)} chunks...")
        embeddings = self.embedder.embed_batch(all_texts, batch_size=64)

        # Store in FAISS
        chunk_ids = [c.id for c in all_chunks]
        self.faiss.add(embeddings, chunk_ids)
        self.faiss.rebuild_ivf()  # Upgrade to IVF if large enough
        self.faiss.save()

        # Store metadata in SQLite
        self.sqlite.upsert_chunks_batch(all_chunks)

        duration = time.time() - start_time
        self.sqlite.set_index_status(
            repo_path, version, "complete",
            total_files=total_files, processed_files=total_files,
            total_chunks=len(all_chunks)
        )

        logger.info(f"Indexing complete: {len(all_chunks)} chunks from {total_files} files in {duration:.1f}s")

        return {
            "total_files": total_files,
            "total_chunks": len(all_chunks),
            "duration_s": round(duration, 2),
            "version": version,
            "commit_sha": commit_sha,
        }

    def _make_index_name(self, repo_path: str, version: str) -> str:
        """Create a unique FAISS index name from repo + version."""
        import hashlib
        key = f"{repo_path}:{version}"
        return hashlib.sha256(key.encode()).hexdigest()[:12]

    def get_faiss_store(self, repo_path: str, version: str) -> FAISSStore:
        """Get the FAISS store for a specific repo+version."""
        index_name = self._make_index_name(str(Path(repo_path).resolve()), version)
        return FAISSStore(index_name=index_name)

    def get_index_status(self, repo_path: str, version: str) -> Optional[dict]:
        return self.sqlite.get_index_status(repo_path, version)

    def get_versions(self, repo_path: str) -> list[str]:
        return self.sqlite.get_versions(repo_path)


# Module-level singleton
_indexer: Indexer | None = None

def get_indexer() -> Indexer:
    global _indexer
    if _indexer is None:
        _indexer = Indexer()
    return _indexer
