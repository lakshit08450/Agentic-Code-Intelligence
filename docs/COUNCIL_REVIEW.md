# Research council review: Samsung PRISM Theme 01
Revised Mon 28 Sep 2026. This document records why `IMPLEMENTATION_PLAN.md` looks the way it does.

Brief given to the council: produce the plan with the highest probability of exceeding 95% on every required metric (NDCG@10 and MRR on MTEB AppsRetrieval), while preventing leakage and overfitting and generalizing to unseen data, within 2 days. No guarantees.

## Round 1: five analyses

### Contrarian: what prevents >95%, or makes a score meaningless
1. Similarity has a ceiling. Many APPS problems share topics and I/O idioms ("read n integers, print the answer modulo 10^9+7"). Embeddings measure topical similarity; they cannot reliably pick one specific solution among near-identical ones. Published small-model scores on APPS sit mostly in single digits to low teens (E5-base 11.5, BGE-base 6.5, Granite-R2 14.0); the best small model found, gte-modernbert-base, reaches 56.4.
2. Leaks can fake a high score: `meta_information` source URLs [V3: reported to match gold pairs], ID numbering shared between queries and documents, reading qrels, tuning on test, or a public checkpoint trained on APPS test pairs. Any custom search code that receives IDs is a leak surface. Hands-on judges who read the code will catch a leak.
3. Tuning noise. Many configurations evaluated on 1,000 validation queries will produce some that look better by chance.
4. Split shift. APPS train and test may differ in difficulty mix [H], so validation may not transfer exactly.
5. Contaminated checkpoints. Some embedders include CoIR training data; the Granite paper notes it deliberately does not. gte-modernbert's APPS score almost certainly reflects APPS-style training pairs. Acceptable only if the test split was excluded, which cannot be fully verified.

### First Principles: what the problem actually is
- One relevant document per query, target 0.95: this is a top-1 identification problem, not a similarity problem. The question is "which program solves this exact problem?"
- APPS statements usually include sample input/output, and every document is an executable program. Running each top-k candidate on the statement's samples is a verification signal that uses only query text and document text. A solution to a different problem almost never prints the right output for this problem's samples.
- Sound design: Stage 1 embedder for recall into the top-k; Stage 2 execution verification for rank 1.
- The final score is bounded by recall@k × the share of queries where the gold solution passes its own samples and is the only candidate that does. Measure this ceiling before building further.

### Expansionist: practical high-impact techniques
1. gte-modernbert-base as the Stage 1 backbone: 149M, Apache-2.0, 8,192-token input (long queries fit without windowing), roughly 4x the APPS score of the next small model.
2. Execution-verified reranking (First Principles' idea): CPU-only, parallel, deterministic, text-only.
3. Dual-encoder fusion inside `AbsEncoder`: concatenate the unit vectors of two models, scaled by √w and √(1−w); the cosine of the concatenation equals the weighted average of the two cosines. A hybrid score that stays inside the guideline's encoder interface.
4. Qwen3-Embedding-0.6B as a larger but still CPU-plausible candidate (instruction-aware; card reports 1-5% gains with instructions).
5. Fine-tuning on APPS train with a free GPU: likely a large gain, but needs permission, time and a second full encode.
6. LLM reranking: strong but infeasible on CPU for 3,765 queries × top-k in 48 hours.

### Outsider: blind spots and hidden assumptions
1. Screening is competitive, not absolute. The bar is beating other teams cleanly, not 0.95.
2. The MTEB interface decides everything. The guideline template is an `AbsEncoder` scored by MTEB's cosine. A reranker reaches the screening JSON only through MTEB's two-stage/cross-encoder route, and only if organizers accept it. Ask today.
3. The verifier has failure modes to measure, not assume: Python 2 solutions, problems with multiple valid outputs, interactive or call-based problems (`class Solution`), float formatting, statements without samples, and near-duplicate problems whose solutions also pass.
4. Executing 8,765 untrusted programs is a security task. Judges will run the code too. It needs a real sandbox (no network, resource limits, disposable filesystem).
5. MTEB may report `mrr_at_10` rather than full MRR; check the real key names before targeting a number.
6. A rival's "88" may be on validation, a subset, or a leaky setup; only the full-test MTEB JSON is comparable.
7. The guideline snippet is missing `import json`, and mentions a CSV once while every submission instruction says JSON.

### Executor: what fits in 48 hours
- Stage 1 and the verifier are both small codebases. The time goes into CPU/GPU runtime, so long jobs run overnight and code is written while they run.
- GPU for encodes (Kaggle/Colab) saves most of a day; embeddings are small files and move easily.
- Fine-tuning adds a GPU dependency, a permission question, a training loop and a second full encode. It does not fit alongside P1 and the deliverables.
- The gold-solution pass rate can be measured with only 1,000 executions, before Stage 1 is even final. Do that first.

## Round 2: peer review

| Reviewer → target | Critique | Resolution |
|---|---|---|
| Contrarian → Stage 2 | Exploits the benchmark's structure (sample I/O); judges may see it as tailored to APPS | Keep. It uses only query and document text; the guidelines invite post-processing and multiple retrieval passes; hands-on queries resemble the dataset. Always report the encoder-only score beside it and disclose the method |
| Executor → First Principles | Stage 2 is worthless if gold solutions often fail their own samples | Gate G2 measures gold PASS rate and the ceiling before tuning |
| Contrarian → Expansionist (gte-modernbert) | Its APPS score may come from APPS training data | Use and disclose; record stated training data; compare validation vs test gap as a contamination signal |
| Outsider → fine-tuning | GPU dependence, permission, overfitting risk, time | Cut |
| Expansionist → Outsider (interface) | If reranking is disallowed, the strategy collapses | Output A (encoder-only, with fusion) is always produced and is guideline-compliant; Output B only if G4 passes |
| First Principles → everyone | Many preprocessing ablations barely move top-1 accuracy | At most three ablations, each through a bootstrap gate |
| Contrarian → Stage 2 tuning | A weight grid can overfit validation | Tiny grid (54 configs) re-scored from cached executions; bootstrap CI gate; one test run per frozen config |
| Outsider → sandbox | Docker may be missing; macOS lacks some rlimits | Docker container is the default on any OS; Linux-only fallback with `unshare -rn`; never unsandboxed |
| Executor → Qwen3-0.6B | May be too slow on CPU for the judges' run | Measure CPU throughput in the bake-off; drop if impractical |

## Chairman's decisions
1. Two-stage architecture: Stage 1 embedder (bake-off: gte-modernbert-base, CodeRankEmbed, Qwen3-Embedding-0.6B; optional fusion), Stage 2 execution-verified reranking in a Docker sandbox.
2. Two screening outputs: A (encoder-only, always) and B (MTEB two-stage, only if G2, G3 and G4 pass).
3. Four gates: G1 harness reproduction, G2 ceiling measurement, G3 bootstrap-significant gain, G4 organizer acceptance.
4. Leakage controls: text-only scoring with an ID-shuffle test, validation from train only, test qrels untouched outside final MTEB runs, one test run per frozen config.
5. Compute split: GPU (Kaggle/Colab) for encodes; local CPU for code, Stage 2, final MTEB runs from cache, P1 and latency. CPU equivalence check required.
6. P1 kept minimal and required; Bonus last.
7. Cut: fine-tuning, LLM reranking/rewriting, models above ~0.6B, the JavaScript structural track, extra ablations.
8. Honesty: >95% is not promised. Report A and B, validation and test, the ceiling and the error breakdown.
