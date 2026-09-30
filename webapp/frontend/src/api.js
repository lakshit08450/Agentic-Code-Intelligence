const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

/**
 * Submit a natural-language query to the CodeLens agent pipeline.
 */
export async function submitQuery(query, version = 'main', maxResults = 10) {
  const res = await fetch(`${API_BASE}/api/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query,
      version,
      max_results: maxResults,
      include_trace: true,
    }),
  });

  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `HTTP ${res.status}`);
  }

  return res.json();
}

/**
 * Trigger repository indexing.
 */
export async function indexRepository(repoPath, version = 'main', forceReindex = false) {
  const res = await fetch(`${API_BASE}/api/index`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      repo_path: repoPath,
      version,
      force_reindex: forceReindex,
    }),
  });

  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `HTTP ${res.status}`);
  }

  return res.json();
}

/**
 * Check indexing status.
 */
export async function getIndexStatus(repoPath, version = 'main') {
  const res = await fetch(
    `${API_BASE}/api/index/status?repo_path=${encodeURIComponent(repoPath)}&version=${version}`
  );

  if (!res.ok) {
    if (res.status === 404) return null;
    throw new Error(`HTTP ${res.status}`);
  }

  return res.json();
}

/**
 * List indexed versions.
 */
export async function getVersions(repoPath) {
  const res = await fetch(
    `${API_BASE}/api/versions?repo_path=${encodeURIComponent(repoPath)}`
  );

  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

/**
 * Health check.
 */
export async function healthCheck() {
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    return res.ok;
  } catch {
    return false;
  }
}
