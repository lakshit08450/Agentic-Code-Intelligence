# Demo video

**Link:** [to be added by the team before submission]

Contents (≤ 5 min), recorded on the dev laptop (CPU only, Windows 11):
1. Problem and approach (fusion Stage 1 + execution-verified Stage 2).
2. CodeLens UI on the APPS corpus: demo problem q2264 (AtCoder, validation split, `docs/demo_queries/q2264.txt`,
   also the "q2264 AtCoder" button): Stage 1 alone ranks the gold solution d2264 second; Stage 2 runs the top 20
   candidates on the sample input in the sandbox and moves it to #1 ("✓ passed sample tests"), with real timing.
3. Plain query without sample I/O: fusion-only results with the "Stage 2 not run" notice.
4. Versioned search: demo repo `@ v2` and all versions `v1..v4` with per-file history (P1 / Bonus).
5. Uploaded JS repo: indexed through PRISM `/embed`; "Stage 2 not applicable (Python only)".
6. Results: test NDCG@10 0.956 (primary, k=50), 0.870 encoder-only; limitations.

Start commands and query list: `webapp/README.md`.
