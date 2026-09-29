# Demo video script (<= 5 min, plan Section 15). Video link goes in `docs/DEMO_VIDEO.md` [P].
Record on CPU (no GPU) with nothing else running. Timings shown on screen come from the CLI output.

**0:00-0:20 Problem.** One slide: statement in, solution out; 8,765 candidates; one correct.

**0:20-1:40 Two live APPS-style queries (Stage 1 then Stage 2).**
- Pick two validation statements with sample I/O (one Codeforces, one AtCoder style) [choose in
  rehearsal from `results/predictions/validation_stage2_exec`; not test queries].
- Index for the demo: `python -m codeintel.versioned.cli index --store stores\apps --jsonl data\demo\apps_corpus.jsonl`
  (one-version JSONL of the corpus; prepared once, embeddings from cache).
- `python -m codeintel.versioned.cli search --store stores\apps --rev apps_corpus "<statement>"`: show top-5, encode/search ms.
- Same with `--verify`: show PASS/WRONG per candidate and the new order, verify ms.

**1:40-2:20 A query Stage 2 fixes.** The walked-through example from PPT slide 10: gold at rank N after
Stage 1, alone PASS, rank 1 after Stage 2.

**2:20-3:10 New commit, incremental vs full rebuild.** Small real git repo with 3-4 commits.
- `index --git <repo> --revs c1 c2 c3`, then commit a change live, `index ... --revs c4`.
- Show the JSON stats: embedded vs reused, ms per step; then the full rebuild time for comparison
  (numbers from `results/versioning_*.json` [pending]).

**3:10-4:00 `--rev` and `--range` on the same query.**
- `search --rev c2 "<query>"` vs `search --rev c4 "<query>"`: different results as code changed.
- `search --range c1..c4 "<query>"`: one result per unit, versions matched, history (introduced/modified/removed).

**4:00-4:40 Results and limitations.** Slide with Output A / B test numbers, validation side by side,
limitations (sandbox is user-space; function-style problems; val/test mix shift).

**4:40-5:00 Close.** Repo, release tag `PRISM_GENAI_HACKATHON_Y2026`.
