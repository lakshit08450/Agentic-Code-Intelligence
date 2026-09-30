import { useState, useCallback } from 'react';
import SearchInput from '../components/SearchInput';
import ResultCard from '../components/ResultCard';
import AgentTrace from '../components/AgentTrace';
import InlineFilters from '../components/InlineFilters';
import IndexPanel from '../components/IndexPanel';
import { useQuery } from '../hooks/useQuery';

const EXAMPLE_QUERIES = [
  "How does the auth middleware validate tokens?",
  "Find all async functions that handle HTTP requests",
  "Show exported utility functions",
  "Where is the database connection initialized?",
  "Error handling in the payment module",
];

/**
 * Home page — the entire CodeLens experience.
 *
 * Three states:
 * 1. Idle: centered search, example queries
 * 2. Loading: search moves to top, spinner
 * 3. Results: ranked cards, trace, filters
 */
export default function Home() {
  const { status, query, results, trace, totalMs, error, search, reset } = useQuery();
  const [activeFilter, setActiveFilter] = useState(null);
  const [showIndex, setShowIndex] = useState(false);

  const isIdle = status === 'idle';
  const isLoading = status === 'loading';
  const hasResults = status === 'success' && results.length > 0;
  const hasError = status === 'error';
  const noResults = status === 'success' && results.length === 0;

  const handleSearch = useCallback((q) => {
    setActiveFilter(null);
    search(q);
  }, [search]);

  const handleReset = useCallback(() => {
    setActiveFilter(null);
    reset();
  }, [reset]);

  // Apply inline filter
  const filteredResults = activeFilter
    ? results.filter(r => r.node_type === activeFilter || r.version === activeFilter)
    : results;

  return (
    <>
      {/* ── Header ── */}
      <header className="header">
        <div className="header__logo">
          <svg className="header__logo-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="11" cy="11" r="8" />
            <path d="M21 21l-4.35-4.35" />
            <path d="M8 8l6 6" opacity="0.4" />
          </svg>
          <span className="header__logo-text">CodeLens</span>
        </div>
        <div className="header__meta">
          {hasResults && (
            <span className="version-badge">
              <span className="version-badge__dot" />
              {results[0]?.version || 'main'}
            </span>
          )}
          <button
            className="filter-link"
            onClick={() => setShowIndex(!showIndex)}
            style={{ fontSize: 'var(--text-xs)' }}
          >
            {showIndex ? 'hide indexer' : 'index repo'}
          </button>
        </div>
      </header>

      {/* ── Main Content ── */}
      <main className={`main ${isIdle ? 'main--idle' : 'main--active'}`}>

        {/* ── Index Panel (conditional) ── */}
        {showIndex && (
          <IndexPanel onIndexComplete={() => setShowIndex(false)} />
        )}

        {/* ── Search ── */}
        <SearchInput
          isIdle={isIdle}
          isLoading={isLoading}
          onSearch={handleSearch}
          onReset={handleReset}
          initialValue={query}
        />

        {/* ── Idle State: Example Queries ── */}
        {isIdle && !showIndex && (
          <div className="examples animate-fade-in">
            <div className="examples__title">Try searching for</div>
            <div className="examples__list">
              {EXAMPLE_QUERIES.map((eq, i) => (
                <button
                  key={i}
                  className="examples__item"
                  onClick={() => handleSearch(eq)}
                >
                  {eq}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* ── Agent Trace ── */}
        {(hasResults || noResults || hasError) && trace.length > 0 && (
          <AgentTrace trace={trace} totalMs={totalMs} />
        )}

        {/* ── Error ── */}
        {hasError && (
          <div className="empty-state animate-fade-in-up">
            <div className="empty-state__icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
                <circle cx="12" cy="12" r="10" />
                <path d="M12 8v4M12 16h.01" />
              </svg>
            </div>
            <h3 className="empty-state__title">Something went wrong</h3>
            <p className="empty-state__desc">{error}</p>
          </div>
        )}

        {/* ── No Results ── */}
        {noResults && (
          <div className="empty-state animate-fade-in-up">
            <div className="empty-state__icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
                <circle cx="11" cy="11" r="8" />
                <path d="M21 21l-4.35-4.35" />
              </svg>
            </div>
            <h3 className="empty-state__title">No results found</h3>
            <p className="empty-state__desc">
              Try rephrasing your query or indexing a repository first.
            </p>
          </div>
        )}

        {/* ── Results ── */}
        {hasResults && (
          <>
            <div className="results__header">
              <span className="results__count">
                {filteredResults.length} result{filteredResults.length !== 1 ? 's' : ''}
              </span>
              <span className="results__timing">
                {totalMs.toFixed(0)}ms
              </span>
            </div>

            <InlineFilters
              results={results}
              activeFilter={activeFilter}
              onFilter={setActiveFilter}
            />

            <div className="results stagger">
              {filteredResults.map((result) => (
                <ResultCard key={`${result.file}-${result.start_line}`} result={result} />
              ))}
            </div>
          </>
        )}
      </main>
    </>
  );
}
