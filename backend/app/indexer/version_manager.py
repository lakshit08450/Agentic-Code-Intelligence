"""
Version Manager — handles git operations for multi-version indexing.
Supports branch/tag listing, checkout, and diff-aware file discovery.
"""

import os
from pathlib import Path
from typing import Optional
from git import Repo, InvalidGitRepositoryError


class VersionManager:
    """Manages git operations for code version tracking."""

    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path).resolve()
        self.repo: Optional[Repo] = None
        self._init_repo()

    def _init_repo(self):
        """Try to open the git repo."""
        try:
            self.repo = Repo(str(self.repo_path))
        except InvalidGitRepositoryError:
            self.repo = None

    @property
    def is_git_repo(self) -> bool:
        return self.repo is not None

    def get_current_branch(self) -> str:
        """Get the current active branch name."""
        if not self.repo:
            return "unknown"
        try:
            return self.repo.active_branch.name
        except TypeError:
            # Detached HEAD
            return self.repo.head.commit.hexsha[:8]

    def get_current_sha(self) -> str:
        """Get the current HEAD commit SHA."""
        if not self.repo:
            return "unknown"
        return self.repo.head.commit.hexsha

    def list_branches(self) -> list[str]:
        """List all local branches."""
        if not self.repo:
            return []
        return [b.name for b in self.repo.branches]

    def list_tags(self) -> list[str]:
        """List all tags."""
        if not self.repo:
            return []
        return [t.name for t in self.repo.tags]

    def list_versions(self) -> list[dict]:
        """
        List all indexable versions (branches + tags) with metadata.
        Returns list of {name, type, sha}.
        """
        versions = []
        if not self.repo:
            return versions

        for branch in self.repo.branches:
            versions.append({
                "name": branch.name,
                "type": "branch",
                "sha": branch.commit.hexsha,
            })

        for tag in self.repo.tags:
            versions.append({
                "name": tag.name,
                "type": "tag",
                "sha": tag.commit.hexsha,
            })

        return versions

    def checkout_version(self, version: str) -> str:
        """
        Checkout a specific branch or tag.
        Returns the commit SHA of the checked-out version.
        """
        if not self.repo:
            raise ValueError("Not a git repository")

        self.repo.git.checkout(version)
        return self.repo.head.commit.hexsha

    def get_changed_files(self, from_sha: str, to_sha: str = "HEAD") -> list[str]:
        """
        Get list of files changed between two commits.
        Used for diff-aware re-indexing.
        """
        if not self.repo:
            return []

        try:
            diff = self.repo.git.diff("--name-only", from_sha, to_sha)
            return [f for f in diff.strip().split("\n") if f]
        except Exception:
            return []

    def get_js_files(self) -> list[str]:
        """
        Get all JavaScript/TypeScript files in the repository.
        Returns relative paths.
        """
        extensions = {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}
        js_files = []

        for root, dirs, files in os.walk(str(self.repo_path)):
            # Skip common non-source directories
            dirs[:] = [d for d in dirs if d not in {
                "node_modules", ".git", "dist", "build", "coverage",
                ".next", "__pycache__", ".cache", "vendor"
            }]

            for f in files:
                if Path(f).suffix in extensions:
                    full_path = Path(root) / f
                    rel_path = full_path.relative_to(self.repo_path)
                    js_files.append(str(rel_path).replace("\\", "/"))

        return sorted(js_files)

    def read_file(self, rel_path: str) -> Optional[str]:
        """Read a file's content by its relative path."""
        full_path = self.repo_path / rel_path
        try:
            return full_path.read_text(encoding="utf-8", errors="replace")
        except (FileNotFoundError, PermissionError):
            return None
