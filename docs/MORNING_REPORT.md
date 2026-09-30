> **Superseded** by README.md and docs/LOG.md (final state 30 Sep). Kept for the record.

# Morning report: night of 28/29 Sep

**Status: I stopped early at about 01:55 because I hit my usage limit.** Steps 1–4 are done and committed. Step 5 (the call-based harness) is only half built. Step 6 got as far as its stdin part, which is still running in the background. Details and numbers are in `docs/LOG.md`.

## Results
| Item | Result |
|---|---|
| E5 check | e5-base-v2 scores 0.4266 on validation vs 0.1153 on test (3.7x). gte-modernbert: 0.7023 vs 0.5774 (1.22x). Validation is easier for **every** model, so the gap isn't evidence of contamination. Validation numbers are inflated across the board. |
| Qwen3 test run (the single pre-declared one, zero-shot) | **NDCG@10 0.7464**, MRR@10 0.7004, recall@10 0.890, recall@100 0.984. File: `results/test_qwen3_zeroshot.json`. |
| Stage 1 decision | **Qwen3-Embedding-0.6B** (0.7464 > 0.5774; projected CPU run 1 h 50 min < 8 h). |
| Qwen3 CPU throughput | 3.65 docs/s, 0.90 queries/s (8 threads). A full CPU test run without cache takes about 1 h 50 min; from cache about 25 s. |
| Validation recall@k (Qwen3) | @1 0.738 · @10 0.924 · @20 0.956 · @50 0.977 · @100 0.987. NDCG@10 0.834, which beats gte-modernbert by +0.132 (95% CI 0.113 to 0.153). |
| Sandbox | 33/33 hostile-program tests pass, including the firewall ones. Four sandbox bugs were found and fixed, each with a regression test. |
| Sandbox throughput | About 4 runs/s at 12 workers. Starting a Python process costs 0.56 s on this machine (cmd.exe: 4 ms), most likely antivirus scanning. I did not change any Defender settings. |
| Parser coverage (validation) | Statements with stdin samples: 336 before, 344 after. Gold PASS rate among parsed statements: **73.2% before, 86.9% after**. |
| Gold outcome distribution (after the fixes) | PASS 299, WRONG 25, ERROR 4, UNKNOWN 672 (606 of these are call-based programs). |
| Call-based share | **60.6%** (606 of 1,000 validation queries), so the harness is needed. |

## Not done
- **Step 5, the call-based harness, is partly built.** The example parser `src/codeintel/stage2/callspec.py` covers 267 of the 606 call-based queries, but has no tests yet. The runner's `--call` mode exists only as a scratch draft, not in the repo. It still needs:
  - copying the draft into `runner.py`;
  - the `--call` path in `sandbox.py` and the verifier;
  - tests, including hostile call-based programs;
  - the full sandbox suite passing again.
- **Step 6 is only partly running.** The stdin share of the Qwen3 top-50 run started at 01:47 in the background (`logs/stage2_top50_stdin.log`, 17,200 pairs, about 3 runs/s). It should finish around 03:30, with results cached in `cache/exec/`. The call-based share, G2 (`results/ceiling.json`), the Stage 2 grid and the G3 bootstrap haven't been run. That means there is no best Stage 2 config yet, and no Stage 1 vs Stage 2 validation comparison with confidence intervals.
- **Top-50 instead of top-100.** At about 4 runs/s, top-100 couldn't have finished by 07:00. The reasoning is in LOG.md.

## Needs your decision
1. **Defender.** Would you allow a Defender exclusion for `tools/sandbox-python`? That would probably cut Stage 2 run time several-fold. It doesn't weaken any sandbox layer, but the plan says it needs your approval.
2. **Output A.** Should it be regenerated with Qwen3? The committed `results/appsretrieval_results_A.json` is still the gte-modernbert v0 run. That's a Phase 4 action on a frozen config.
3. **Placeholders still missing:** registration yes/no, whether the organizer email was sent, the exact deadline time, and the GitHub URL.

## Next steps for Tuesday
1. Finish the harness: move the draft into `runner.py`, wire up `sandbox.py` and the verifier, add tests, then run the full sandbox suite.
2. Rerun top-50 with `ceiling topk` so the call-based pairs are included. The stdin pairs will come from cache.
3. Compute G2 (`results/ceiling.json`), then run the Stage 2 grid and the G3 bootstrap.
4. Phase 3 ablations, then freeze the configs by 16:00.
