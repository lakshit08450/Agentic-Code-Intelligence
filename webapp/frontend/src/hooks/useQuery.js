import { useState, useCallback } from 'react';
import { submitQuery } from '../api';

/**
 * Hook for managing search query state and execution.
 */
export function useQuery() {
  const [state, setState] = useState({
    status: 'idle',    // idle | loading | success | error
    query: '',
    results: [],
    trace: [],
    totalMs: 0,
    error: null,
    queryId: null,
  });

  const search = useCallback(async (query, version = 'main', maxResults = 10) => {
    if (!query.trim()) return;

    setState(prev => ({
      ...prev,
      status: 'loading',
      query,
      error: null,
    }));

    try {
      const data = await submitQuery(query, version, maxResults);

      setState({
        status: 'success',
        query,
        results: data.results || [],
        trace: data.trace || [],
        totalMs: data.total_ms || 0,
        error: null,
        queryId: data.query_id,
      });
    } catch (err) {
      setState(prev => ({
        ...prev,
        status: 'error',
        error: err.message,
      }));
    }
  }, []);

  const reset = useCallback(() => {
    setState({
      status: 'idle',
      query: '',
      results: [],
      trace: [],
      totalMs: 0,
      error: null,
      queryId: null,
    });
  }, []);

  return { ...state, search, reset };
}
