/**
 * ResultCard — displays a single code search result.
 * Uses variable density: top-ranked results are larger.
 */
export default function ResultCard({ result }) {
  const { rank, score, file, start_line, end_line, snippet, node_type, node_name, version, context } = result;

  // Variable density class
  const rankClass = rank <= 2
    ? `result-card--rank-${rank}`
    : rank > 5
      ? 'result-card--rank-low'
      : '';

  // Format code lines with line numbers
  const codeLines = snippet.split('\n');
  const maxLineDigits = String(end_line).length;

  return (
    <article className={`result-card ${rankClass}`} id={`result-${rank}`}>
      <div className="result-card__header">
        <div className="result-card__meta">
          <span className="result-card__file" title={file}>
            {file}
          </span>
          {node_name && (
            <span className="result-card__name">{node_name}</span>
          )}
          <span className="result-card__line-range">
            L{start_line}–{end_line}
          </span>
        </div>

        <div className="result-card__badges">
          <span className="badge badge--score" title="Relevance score">
            {score.toFixed(2)}
          </span>
          {node_type && (
            <span className="badge badge--type">
              {formatNodeType(node_type)}
            </span>
          )}
          {version && (
            <span className="badge badge--version">{version}</span>
          )}
        </div>
      </div>

      <div className="result-card__code">
        <pre>
          {codeLines.map((line, i) => (
            <div className="result-card__code-line" key={i}>
              <span className="result-card__line-num">
                {String(start_line + i).padStart(maxLineDigits)}
              </span>
              <span className="result-card__line-content">{line}</span>
            </div>
          ))}
        </pre>
      </div>

      {context && (context.enclosing || (context.imports && context.imports.length > 0)) && (
        <div className="result-card__context">
          {context.enclosing && (
            <span>Scope: {context.enclosing}</span>
          )}
          {context.imports && context.imports.length > 0 && (
            <span>Imports: {context.imports.slice(0, 3).join(', ')}{context.imports.length > 3 ? '…' : ''}</span>
          )}
        </div>
      )}
    </article>
  );
}

function formatNodeType(type) {
  const map = {
    'function_declaration': 'function',
    'class_declaration': 'class',
    'method_definition': 'method',
    'arrow_function': 'arrow fn',
    'variable_declaration': 'variable',
    'lexical_declaration': 'const/let',
    'export_statement': 'export',
  };
  return map[type] || type;
}
