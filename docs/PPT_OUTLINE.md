# PPT outline (draft, plan Section 15). File name per template: `CollegeName_TeamName` [P: fill in].
Numbers come only from `results/`; `[pending]` = run in progress.

1. **Title**: Execution-verified code retrieval. Theme 01, Agentic Code Intelligence. Team, college. [P]
2. **The problem in our words**: given a contest problem statement, find the one program (out of 8,765) that
   solves it; the same search must work across versions of a codebase, on CPU.
3. **Why APPS is a top-1 problem**: one relevant solution per query; many problems share topics and I/O
   idioms, so similarity alone confuses them. Reference points: e5-base-v2 0.115, gte-modernbert 0.564
   (published), our harness reproduces both (0.1153 / 0.5774).
4. **Architecture**: Stage 1 Qwen3-Embedding-0.6B (recall) -> Stage 2 run candidates on the statement's
   sample I/O in a sandbox (precision) -> MTEB two-stage JSON. Diagram from README Section 1.
5. **Stage 1 bake-off (validation)**: Qwen3 0.834 NDCG@10 / recall@50 0.977 vs gte-modernbert 0.702 /
   0.902 vs CodeRankEmbed 0.637 / 0.759; CPU throughput 3.65 docs/s. Pre-declared zero-shot test check:
   0.7464 vs 0.5774.
6. **Stage 2 and the sandbox**: sample parser (formats of Codeforces, AtCoder, CodeChef; 97.9% test
   coverage), comparator, five-layer Windows sandbox, 33 hostile-program tests (fork/child processes,
   network, file writes/deletes, memory bombs, 100 MB output, ctypes). Two refinements: any-answer
   problems, trivial outputs.
7. **Results (test, all runs)**: Qwen3 encoder 0.7464 / MRR 0.7004; Qwen3 + Stage 2 k=20 0.8938 / 0.8812;
   Qwen3 + Stage 2 k=50 0.9127 / 0.8980; fusion (Qwen3 0.3 + EmbeddingGemma 0.7) encoder 0.8702 / 0.8400
   (encoder-only submission); fusion + Stage 2 k=50 [pending]; fusion + Stage 2 k=20 [pending, ablation].
   Primary = higher of the two k=50 runs (the one test-based choice, disclosed). Validation side by side
   (stdin subset, test-mix weighted): Stage 1 0.783 -> Stage 2 k=20 0.922 / k=50 0.944; fusion + Stage 2 k=50 0.965.
8. **Where the ceiling is (G2) and error analysis**: test loss buckets (rank 1: 60.5%, 2-10: 28.5%,
   11-20: 4.5%, 21-100: 4.9%); gold solutions pass their own samples in 82% (Codeforces) / 80% (AtCoder)
   of validation cases; failure causes (several valid answers, label noise, Python 2, no samples);
   wrong programs pass by chance on 56% of trivial-output queries vs 11% otherwise.
9. **P1 / Bonus**: incremental index (embed only changed content, renames keep identity), `--rev`,
   `--range` with per-unit history; numbers [pending: simulate_versions + CPU latency, Wed morning].
10. **One tough query walked through** [pending: pick from Output B vs A predictions on validation, e.g. a
    Codeforces problem where the gold solution moves from rank 4 to 1 because it alone passes].
11. **Limitations, licenses, disclosure, next steps**: user-space sandbox (weaker than a container);
    function-style problems not executed; val/test source-mix shift; process start-up bound; licenses
    Apache-2.0 / MIT; no test data used for selection; GPU for development encodes only; next: call-based
    harness, format-aware checks for any-answer problems, faster sandbox start-up.
