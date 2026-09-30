import { useState } from 'react';

/**
 * AgentTrace — collapsible build-log-style timeline.
 * Shows the Plan → Search → Read → Refine steps.
 * Collapsed by default — secondary info, not the main attraction.
 */
export default function AgentTrace({ trace, totalMs }) {
  const [isOpen, setIsOpen] = useState(false);

  if (!trace || trace.length === 0) return null;

  return (
    <div className="trace animate-fade-in">
      <button
        className="trace__toggle"
        onClick={() => setIsOpen(!isOpen)}
        id="trace-toggle"
        aria-expanded={isOpen}
      >
        <span>
          Agent pipeline · {totalMs.toFixed(0)}ms · {trace.length} steps
        </span>
        <svg
          className={`trace__toggle-icon ${isOpen ? 'trace__toggle-icon--open' : ''}`}
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
        >
          <polyline points="6 9 12 15 18 9" />
        </svg>
      </button>

      <div className={`trace__body ${isOpen ? 'trace__body--open' : ''}`}>
        {trace.map((step, i) => (
          <div className="trace__step" key={i}>
            <span className="trace__step-phase">{step.phase}</span>
            <span className="trace__step-detail">{step.detail}</span>
            <span className="trace__step-time">{step.duration_ms.toFixed(0)}ms</span>
          </div>
        ))}
      </div>
    </div>
  );
}
