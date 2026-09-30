/**
 * InlineFilters — text-link filters that appear AFTER results.
 * Not buttons. Not dropdowns. Just subtle text links.
 */
export default function InlineFilters({ results, activeFilter, onFilter }) {
  if (!results || results.length === 0) return null;

  // Extract filter options from results
  const nodeTypes = [...new Set(results.map(r => r.node_type).filter(Boolean))];
  const versions = [...new Set(results.map(r => r.version).filter(Boolean))];

  if (nodeTypes.length <= 1 && versions.length <= 1) return null;

  return (
    <div className="filters animate-fade-in">
      <span className="filters__label">Narrow by</span>

      <button
        className={`filter-link ${!activeFilter ? 'filter-link--active' : ''}`}
        onClick={() => onFilter(null)}
      >
        all
      </button>

      {nodeTypes.map(type => (
        <button
          key={type}
          className={`filter-link ${activeFilter === type ? 'filter-link--active' : ''}`}
          onClick={() => onFilter(type)}
        >
          {formatType(type)}
        </button>
      ))}

      {versions.length > 1 && versions.map(v => (
        <button
          key={v}
          className={`filter-link ${activeFilter === v ? 'filter-link--active' : ''}`}
          onClick={() => onFilter(v)}
        >
          {v}
        </button>
      ))}
    </div>
  );
}

function formatType(type) {
  const map = {
    'function_declaration': 'functions',
    'class_declaration': 'classes',
    'method_definition': 'methods',
    'arrow_function': 'arrow fns',
    'variable_declaration': 'variables',
    'export_statement': 'exports',
  };
  return map[type] || type;
}
