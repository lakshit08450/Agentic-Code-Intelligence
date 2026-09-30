import { useState } from 'react';
import { indexRepository, getIndexStatus } from '../api';

/**
 * IndexPanel — repository indexing interface.
 * Only appears when needed — not a permanent fixture.
 */
export default function IndexPanel({ onIndexComplete }) {
  const [repoPath, setRepoPath] = useState('');
  const [version, setVersion] = useState('main');
  const [status, setStatus] = useState(null); // null | pending | indexing | complete | error
  const [progress, setProgress] = useState({ processed: 0, total: 0, chunks: 0 });
  const [error, setError] = useState(null);

  const handleIndex = async () => {
    if (!repoPath.trim()) return;

    setStatus('pending');
    setError(null);

    try {
      await indexRepository(repoPath.trim(), version);
      setStatus('indexing');

      // Poll for status
      const pollInterval = setInterval(async () => {
        try {
          const statusData = await getIndexStatus(repoPath.trim(), version);
          if (!statusData) return;

          setProgress({
            processed: statusData.processed_files || 0,
            total: statusData.total_files || 0,
            chunks: statusData.total_chunks || 0,
          });

          if (statusData.status === 'complete') {
            clearInterval(pollInterval);
            setStatus('complete');
            if (onIndexComplete) onIndexComplete();
          } else if (statusData.status === 'error') {
            clearInterval(pollInterval);
            setStatus('error');
            setError(statusData.error_message || 'Unknown error');
          }
        } catch (err) {
          // Polling error — continue
        }
      }, 1000);

    } catch (err) {
      setStatus('error');
      setError(err.message);
    }
  };

  const progressPct = progress.total > 0
    ? Math.round((progress.processed / progress.total) * 100)
    : 0;

  return (
    <div className="index-panel animate-fade-in-up">
      <h3 className="index-panel__title">Index a Repository</h3>

      <div className="index-panel__input-group">
        <input
          className="index-panel__input"
          type="text"
          value={repoPath}
          onChange={(e) => setRepoPath(e.target.value)}
          placeholder="Absolute path to repository…"
          id="index-path-input"
          disabled={status === 'indexing'}
        />
        <input
          className="index-panel__input"
          type="text"
          value={version}
          onChange={(e) => setVersion(e.target.value)}
          placeholder="Branch"
          style={{ maxWidth: '120px' }}
          id="index-version-input"
          disabled={status === 'indexing'}
        />
        <button
          className="index-panel__submit"
          onClick={handleIndex}
          disabled={!repoPath.trim() || status === 'indexing'}
          id="index-submit"
        >
          {status === 'indexing' ? 'Indexing…' : 'Index'}
        </button>
      </div>

      {status === 'indexing' && (
        <>
          <div className="index-panel__progress">
            <div
              className="index-panel__progress-bar"
              style={{ width: `${progressPct}%` }}
            />
          </div>
          <div className="index-panel__status">
            {progress.processed}/{progress.total} files · {progress.chunks} chunks
          </div>
        </>
      )}

      {status === 'complete' && (
        <div className="index-panel__status" style={{ color: 'var(--color-success)' }}>
          ✓ Indexed {progress.total} files → {progress.chunks} chunks
        </div>
      )}

      {status === 'error' && (
        <div className="index-panel__status" style={{ color: 'var(--color-error)' }}>
          ✕ {error}
        </div>
      )}
    </div>
  );
}
