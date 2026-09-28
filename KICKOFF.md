# Kickoff: running this plan with Claude Code (native Windows)

For the human (P). Platform: Windows 11 Home, no WSL2, no Docker. Claude Code docs: https://docs.anthropic.com/en/docs/claude-code/overview

## 1. One-time setup
1. Move the project out of OneDrive to a plain folder, e.g. `C:\dev\prism-code-intel` (keep `data\apps\`, CLAUDE.md, IMPLEMENTATION_PLAN.md, KICKOFF.md, docs\). Start Claude Code from that folder.
2. Create an empty GitHub repo (private is fine while you work; it must be accessible to the judges at submission).
3. Send the organizer email in IMPLEMENTATION_PLAN.md Section 3 if you haven't.
4. When Claude Code prints the firewall command for the sandbox, run it once in an **admin** PowerShell.

## 2. How to drive it
- One phase per prompt; wait for its acceptance report before the next.
- Accept file edits freely. Don't approve any command that runs APPS code until `tests/test_sandbox.py` passes.
- If a session gets long, start a new one and say: "Read CLAUDE.md, IMPLEMENTATION_PLAN.md and docs/LOG.md, then continue from where the log ends."

## 3. Prompts (copy-paste)

**Prompt 1: switch to native Windows, then Phase 0**
> I've decided not to use WSL2 or Docker; we run natively on Windows. I replaced CLAUDE.md, IMPLEMENTATION_PLAN.md, KICKOFF.md and docs/COUNCIL_REVIEW.md with Windows versions. Re-read all four (the plan's Sections 0, 6, 8 and 11.3 changed). The repo is now at [C:\dev\prism-code-intel]. Answers: registration [yes/no]; use the local RTX 5060 for development encodes, no Kaggle unless it fails; organizer email [sent/not sent]; deadline 30 Sep [exact time]; GitHub repo [URL], set it as origin. Then do Phase 0 (0.1 to 0.6): Python 3.11 venv, verify the CUDA build of PyTorch on the RTX 5060, mteb API check logged, encoder, encode script, bg.py launcher, G1, Output A v0. Stop at the acceptance check and report the G1 numbers.

**Prompt 2: Phase 1 and Phase 2a**
> Do Phase 1 (devset, metrics, bootstrap, bake-off on the local GPU, CPU equivalence check, licenses and training data) and Phase 2a. For 2a, build the Windows sandbox exactly as Section 11.3 describes: run scripts/setup_sandbox.ps1, give me the firewall command to run as admin, then make every hostile-program test pass before running any APPS code. Measure sandbox runs/s on 500 runs, then the gold-solution pass rate on validation. Show me 20 gold failures with causes before fixing the parser. Then start the overnight Stage 2 run on the validation top-50 with bg.py and a log file.

**Prompt 3: Phase 2b (Tuesday morning)**
> Read the logs and docs/LOG.md. Compute G2 into results/ceiling.json, run the Stage 2 grid on validation, apply the G3 bootstrap gate, and report recall@k, gold outcome distribution, ceiling, best grid config, and validation NDCG@10/MRR for Stage 1 vs Stage 2 with CIs.

**Prompt 4: Phase 3**
> Run the three Stage 1 ablations through G3 (query cap, drop examples, fusion with the runner-up). Re-run Stage 2 only if Stage 1 changed. Freeze configs/stage1_final.yaml and configs/stage2_final.yaml and show me both before we touch test.

**Prompt 5: Phase 4**
> Encode the test queries for the frozen config on the local GPU. Then produce Output A with mteb.evaluate (saving predictions) and Output B with mteb's two-stage reranking and ExecutionReranker. One run each, logged. Report both JSONs' ndcg_at_10 and MRR next to validation numbers.

**Prompt 6: Phase 5 (Path B)**
> Build Section 12 (ingest, units, store, search, cli with --rev, --range, --verify) and scripts/simulate_versions.py. Test on a tiny model while long jobs run. When the CPU is free, record the Section 12.5 metrics and CPU latency. Do --range only after --rev works.

**Prompt 7: Phase 6 (deliverables)**
> Write the README per Section 14 (including the sandbox's security model and its limits on Windows), fill results tables from results/ only, draft the PPT outline with real numbers per Section 15, write the demo script, then do a clean-clone test in a new folder following the README. List every Section 14 checklist item with its status.

## 4. Final checks before submitting (Wednesday)
- Both JSONs exist, produced by mteb; the release states which one is the submission.
- Tag `PRISM_GENAI_HACKATHON_Y2026` on the final commit; PPT named per template; video ≤5 min with access set; Google Form submitted once.
