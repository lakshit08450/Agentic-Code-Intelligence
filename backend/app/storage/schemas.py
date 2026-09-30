"""
CodeLens — Pydantic schemas for storage layer.
Defines the data models used across the indexing and retrieval pipeline.
"""

from pydantic import BaseModel, Field
from typing import Optional


class CodeChunk(BaseModel):
    """A single chunk of code extracted from a source file."""
    id: str = Field(description="Unique chunk identifier (hash of file+lines+version)")
    file_path: str = Field(description="Relative path to source file")
    start_line: int = Field(description="1-indexed start line")
    end_line: int = Field(description="1-indexed end line (inclusive)")
    content: str = Field(description="Raw source code text")
    language: str = Field(default="javascript")
    version: str = Field(description="Git branch/tag/commit")
    commit_sha: str = Field(description="Full commit SHA")
    repo_path: str = Field(description="Absolute path to repository root")

    # AST metadata (populated during structural indexing)
    node_type: Optional[str] = Field(default=None, description="e.g. function_declaration, class_declaration")
    node_name: Optional[str] = Field(default=None, description="e.g. function name, class name")
    is_exported: bool = Field(default=False)
    is_async: bool = Field(default=False)


class SearchResult(BaseModel):
    """A single search result returned to the user."""
    rank: int
    score: float
    file: str
    start_line: int
    end_line: int
    snippet: str
    language: str = "javascript"
    type: str = "semantic"
    version: str = ""
    commit_sha: str = ""
    node_type: Optional[str] = None
    node_name: Optional[str] = None
    context: dict = Field(default_factory=dict)


class AgentTraceStep(BaseModel):
    """One step in the agent trace timeline."""
    phase: str  # plan | search | read | refine
    detail: str
    duration_ms: float
    sub_queries: list[str] = Field(default_factory=list)
    candidates_count: int = 0


class QueryRequest(BaseModel):
    """Incoming search query from the frontend."""
    query: str
    mode: str = "hybrid" # semantic | structural | hybrid
    version: str = "main"
    max_results: int = 10
    include_trace: bool = True


class QueryResponse(BaseModel):
    """Full response to a search query."""
    query_id: str
    query: str
    results: list[SearchResult]
    trace: list[AgentTraceStep] = Field(default_factory=list)
    total_ms: float = 0.0


class IndexRequest(BaseModel):
    """Request to index a repository."""
    repo_path: str
    version: str = "main"
    force_reindex: bool = False


class IndexStatus(BaseModel):
    """Status of an indexing job."""
    repo_path: str
    version: str
    status: str  # pending | indexing | complete | error
    total_files: int = 0
    processed_files: int = 0
    total_chunks: int = 0
    error_message: Optional[str] = None
