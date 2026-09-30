import { useState, useRef, useCallback } from 'react';

/**
 * Unified search input component.
 * Single input — no dropdowns, no mode toggles.
 * Adapts size based on idle vs active state.
 */
export default function SearchInput({ isIdle, isLoading, onSearch, onReset, initialValue = '' }) {
  const [value, setValue] = useState(initialValue);
  const inputRef = useRef(null);

  const handleSubmit = useCallback((e) => {
    e.preventDefault();
    if (value.trim() && !isLoading) {
      onSearch(value.trim());
    }
  }, [value, isLoading, onSearch]);

  const handleKeyDown = useCallback((e) => {
    if (e.key === 'Escape') {
      setValue('');
      if (onReset) onReset();
    }
  }, [onReset]);

  return (
    <form className={`search ${isIdle ? 'search--idle' : ''}`} onSubmit={handleSubmit}>
      <div className="search__wrapper">
        <svg className="search__icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="11" cy="11" r="8" />
          <path d="M21 21l-4.35-4.35" />
        </svg>

        <input
          ref={inputRef}
          className="search__input"
          type="text"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={isIdle
            ? "Describe what you're looking for…"
            : "Search code…"
          }
          autoFocus
          id="search-input"
          aria-label="Search code"
        />

        {isLoading && <div className="search__spinner" aria-label="Searching" />}
      </div>

      {isIdle && (
        <div className="search__hint">
          Press <kbd>Enter</kbd> to search · <kbd>Esc</kbd> to clear
        </div>
      )}
    </form>
  );
}
