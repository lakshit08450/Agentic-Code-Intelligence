# PPT outline (final numbers, plan Section 15). File name per template: `CollegeName_TeamName` [team: fill in].
Numbers come only from `results/` (commands and commits in `docs/LOG.md`).

1. **Title**: Execution-verified code retrieval. Theme 01, Agentic Code Intelligence. Team, college. [team]
2. **The problem in our words**: given a contest problem statement, find the one program (out of 8,765) that
   solves it; the same search must work across versions of a codebase, on CPU.
3. **Why APPS is a top-1 problem**: one relevant solution per query; many problems share topics and I/O
   idioms, so similarity alone confuses them. Reference points: e5-base-v2 0.115, gte-modernbert 0.564
   (published); our harness reproduces both (0.1153 / 0.5774).
4. **Architecture**: Stage 1 = pretrained embedding models, fused (Qwen3-Embedding-0.6B 0.3 +
   EmbeddingGemma-300m 0.7; cosine = weighted sum) for recall -> Stage 2 = run the top-k candidates on the
   statement's sample I/O in a sandbox for precision -> MTEB two-stage JSON. No training. Diagram: README Section 1.
5. **Stage 1 (validation)**: bake-off Qwen3 0.834 NDCG@10 / recall@50 0.977 vs gte-modernbert 0.702 / 0.902 vs
   CodeRankEmbed 0.637 / 0.759. Fusion with EmbeddingGemma (stdin, test-mix weighted): NDCG@10 0.874 vs 0.783,
   recall@50 0.994 vs 0.971 (G3 passed). CPU: 3.65 docs/s (Qwen3).
6. **Stage 2 and the sandbox**: sample parser (Codeforces, AtCoder, CodeChef formats; 97.9% of test statements),
   comparator, five-layer Windows sandbox (own interpreter, firewall block, audit-hook guard, Job Object limits,
   temp dir), 33 hostile-program tests. Refinements: any-answer problems (WRONG neutral), trivial outputs
   (PASS x0.1). Validation Stage 1 0.783 -> Stage 2 0.922 (k=20) / 0.944 (k=50).
7. **Results (test, all runs)**:
   | pipeline | NDCG@10 | MRR@10 |
   |---|---|---|
   | Qwen3 encoder | 0.7464 | 0.7004 |
   | Qwen3 + Stage 2 k=20 | 0.8938 | 0.8812 |
   | Qwen3 + Stage 2 k=50 | 0.9127 | 0.8980 |
   | Fusion encoder (**encoder-only submission**) | 0.8702 | 0.8400 |
   | **Fusion + Stage 2 k=50 (primary submission)** | **0.9565** | **0.9450** |
   | Fusion + Stage 2 k=20 (ablation; = app depth) | 0.9520 | 0.9419 |
   Primary = higher of the two k=50 runs (the one test-based choice, disclosed). The live app runs Stage 2 at
   k=20 for interactivity.
8. **Ceiling and error analysis**: test Stage 1 loss buckets (rank 1: 60.5%, 2-10: 28.5%, 11-20: 4.5%,
   21-100: 4.9%) -> Stage 2 targets ranks 2-10; gold solutions pass their own samples in 82% (Codeforces) /
   80% (AtCoder) of validation cases; failure causes: several valid answers, label noise, Python 2, no samples;
   chance passes on 56% of trivial-output queries vs 11% otherwise.
9. **P1 / Bonus** (scaled simulation, 1,000 units x 5 versions, CPU): incremental update 91-217 s vs full
   rebuild 506 s; ~79.5% of embeddings reused per version; single-version NDCG@10 0.960; all versions
   collapsed per unit 0.958 vs 0.951 without collapse. On this repo's git history an update re-embeds 0-1 files.
   `--rev` and `--range` (per-file history) in the CLI and the app.
10. **One query walked through (validation q2264, AtCoder)**: Stage 1 ranks the gold solution d2264 second;
    Stage 2 executes the top 20 on the sample input; d2264 prints the expected output and moves to #1
    ("passed sample tests"). Live in the demo video.
11. **Speed and reproducibility (CPU)**: Stage 1 p50 1.25 s, Stage 2 (k=50) p50 13 s (Windows process start-up
    bound); app load 34 s, 3.7 GB RAM. CPU re-encode vs GPU cache: NDCG@10 within 0.003, recall@50 identical,
    exact top-10 order on 52% of queries.
12. **Limitations, licenses, disclosure, next steps**: user-space sandbox (weaker than a container);
    function-style problems not executed; val/test source-mix shift; one test-based choice; approximate CPU
    reproduction; licenses Apache-2.0 (Qwen3), Gemma terms (EmbeddingGemma); GPU for development encodes only.
    Next: call-based harness, format-aware checks for any-answer problems, faster sandbox start-up.
