# CLAUDE.md: standing instructions for this repo

Project: Samsung PRISM GenAI Hackathon, Theme 01 (code retrieval on MTEB AppsRetrieval + versioned retrieval).
Deadline: Wed 30 Sep 2026, submit by 18:00 IST.
Platform: native Windows 11 Home. No WSL2, no Docker. Local RTX 5060 (8 GB) for development encodes.
Source of truth: `IMPLEMENTATION_PLAN.md`. Rationale: `docs/COUNCIL_REVIEW.md`.

## Workflow
- Work one phase at a time from IMPLEMENTATION_PLAN.md Section 9. At the end of a phase, run its acceptance check, report the numbers to the user, commit, and stop.
- Log every measured number in `docs/LOG.md` with: command, config file, git commit, hardware (CPU/GPU model).
- Anything marked VERIFY in the plan: read the installed library source first, write what you found in `docs/LOG.md`, then code.
- Ask the user before deciding anything in plan Section 3, before any test-split run not listed in the plan, and before changing a frozen config.

## Hard rules (never break, even to debug)
1. Scoring code uses only query text and document text. Never read qrels, query/doc IDs or `meta_information` as features. IDs are dictionary keys only. `tests/test_no_id_leak.py` must pass.
2. No tuning on the test split. Selection uses the validation split from train (seed 13, 4,000/1,000).
3. Never open test qrels outside the final `mteb.evaluate` run. Do not "peek" at test labels to debug a low score.
4. APPS documents are untrusted code. Run them only through `stage2/sandbox.py` with `tools/sandbox-python/python.exe` (firewall-blocked, Job Object limits, audit-hook guard in `runner.py`). Never run them with the project venv, never exec/eval/import them in the main process, and never weaken or bypass a sandbox layer to make something work.
5. The submitted pipeline defaults to CPU. GPU is for development encodes only.
6. No number goes into README/PPT/demo unless a script produced it into `results/`.

## Long-running jobs
- Encodes, Stage 2 execution runs and MTEB runs take minutes to hours. Start them detached with the launcher, e.g.
  `.venv/Scripts/python scripts/bg.py logs/ceiling.log -- .venv/Scripts/python -m codeintel.eval.ceiling --config configs/stage2_final.yaml`
  then poll the log. Do not hold a foreground command open for long jobs, and do not rely on `nohup`.
- Never measure latency while a long job is using the CPU.
- MTEB caches results under `~/.cache/mteb`; always use the overwrite option so reruns actually run.

## Windows conventions
- Always use the venv interpreter (`.venv/Scripts/python`); the machine also has Python 3.10 and 3.14.
- `PYTHONUTF8=1`; open text files with `encoding="utf-8"`; `pathlib` for paths; `if __name__ == "__main__":` in every script (spawn-based multiprocessing).
- Normalise `\r\n` before comparing program output.
- The one-time firewall rule needs admin rights: print the exact PowerShell command for the user to run; do not try to elevate yourself.

## Code conventions
- Python 3.11, package under `src/codeintel/`, configs in YAML under `configs/`, every preprocessing/scoring option is a config flag.
- Caches: embeddings keyed by sha1(model id | config fingerprint | preprocessed text); executions keyed by sha1(doc) | sha1(sample input) | sandbox version. Caches live in `cache/` (gitignored).
- Tests in `tests/` with pytest. Add a test for every parser/comparator bug found.
- Keep dependencies minimal. No new frameworks (no FastAPI, no vector DB, no LangChain).

## Priorities when time is short
Output A JSON > Output B JSON > P1 `--rev` > README/PPT/video > Bonus `--range`. Ship the last completed phase rather than a half-finished next one.
