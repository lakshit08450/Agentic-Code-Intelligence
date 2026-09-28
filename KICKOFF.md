# Kickoff: running this plan with Claude Code

For the human (P). Follow this top to bottom. Claude Code install and login: https://docs.anthropic.com/en/docs/claude-code/overview

## 1. One-time setup (10 minutes)
```bash
mkdir prism-code-intel && cd prism-code-intel
git init
mkdir -p docs results logs cache
# copy these files into the repo:
#   CLAUDE.md, IMPLEMENTATION_PLAN.md, KICKOFF.md  -> repo root
#   COUNCIL_REVIEW.md                              -> docs/
printf "cache/\nlogs/\n.venv/\n__pycache__/\n*.pyc\n" > .gitignore
git add . && git commit -m "Plan and standing instructions"
# create an empty GitHub repo, then:
git remote add origin https://github.com/<you>/prism-code-intel.git && git push -u origin main
```
Also today:
- Send the organizer email in IMPLEMENTATION_PLAN.md Section 3.
- Check Docker is installed and running (`docker run --rm hello-world`). The sandbox needs it.
- Have a Kaggle (preferred) or Colab account ready for GPU encodes.

Start Claude Code from the repo root (`claude`). It reads CLAUDE.md automatically.

## 2. How to drive it
- Give one phase per prompt. Wait for its acceptance report before the next.
- Accept file edits freely. Review any command that executes APPS code until the sandbox tests pass.
- If a session gets long, start a new one and say: "Read CLAUDE.md, IMPLEMENTATION_PLAN.md and docs/LOG.md, then continue from where the log ends."

## 3. Prompts (copy-paste)

**Prompt 0: orientation**
> Read CLAUDE.md, IMPLEMENTATION_PLAN.md and docs/COUNCIL_REVIEW.md in full. Then ask me the Section 3 questions (OS, cores, RAM, Docker, Kaggle/Colab) and summarise the Phase 0 steps you will take. Don't write code yet.

**Prompt 1: Phase 0**
> Do Phase 0 (0.1 to 0.6). Start with the Section 8 API verification against the installed mteb source and log findings. Build scripts/encode.py and scripts/kaggle_encode.ipynb so I can run the GPU encodes on Kaggle. Stop when G1 is logged and Output A v0 exists, and tell me exactly what to run on Kaggle and which files to bring back.

**Prompt 2: Phase 1 and Phase 2a together**
> Do Phase 1 (devset, metrics, bootstrap, bake-off using the GPU caches I bring back, CPU equivalence check, licenses and training data) and, in parallel, Phase 2a (sampleio, compare, sandbox with hostile-program tests, gold-solution pass rate on validation). Show me 20 gold failures with their causes before fixing the parser. Then start the overnight Stage 2 run on the validation top-50 in the background with a log file.

**Prompt 3: Phase 2b (Tuesday morning)**
> Read logs and docs/LOG.md. Compute G2 into results/ceiling.json, run the Stage 2 grid on validation, apply the G3 bootstrap gate, and report: recall@k, gold outcome distribution, ceiling, best grid config, and validation NDCG@10/MRR for Stage 1 vs Stage 2 with CIs.

**Prompt 4: Phase 3**
> Run the three Stage 1 ablations through G3 (query cap, drop examples, fusion with the runner-up). Re-run Stage 2 only if Stage 1 changed. Freeze configs/stage1_final.yaml and configs/stage2_final.yaml and show me both before we touch test.

**Prompt 5: Phase 4**
> Tell me what to encode on Kaggle for the test queries. Then produce Output A with mteb.evaluate (saving predictions) and Output B with mteb's two-stage reranking and ExecutionReranker. One run each, logged. Report both JSONs' ndcg_at_10 and MRR next to validation numbers.

**Prompt 6: Phase 5 (Path B)**
> Build Section 12 (ingest, units, store, search, cli with --rev, --range, --verify) and scripts/simulate_versions.py. Test on a tiny model while long jobs run. When the CPU is free, record Section 12.5 metrics and CPU latency. Do --range only after --rev works.

**Prompt 7: Phase 6 (deliverables)**
> Write the README per Section 14, fill results tables from results/ only, draft the PPT outline with real numbers per Section 15, write the demo script, then do a clean-clone test in a new folder following the README. List every checklist item in Section 14 with its status.

## 4. Kaggle GPU encode (what the notebook should do)
```python
!git clone https://github.com/<you>/prism-code-intel.git    # private repo: use a token or make it public
%cd prism-code-intel
!pip install -q -e .
!python scripts/encode.py --config configs/bakeoff/gte_modernbert.yaml --device cuda --targets corpus,queries
!python scripts/encode.py --config configs/bakeoff/coderankembed.yaml --device cuda --targets corpus,queries
!python scripts/encode.py --config configs/bakeoff/qwen3_0p6b.yaml   --device cuda --targets corpus,queries
!tar czf /kaggle/working/embeddings.tgz cache/embeddings
```
Enable the GPU accelerator in notebook settings, download `embeddings.tgz` from the Output tab, and extract it into the local repo's `cache/`. Pin the same package versions as local (Claude Code writes them into the notebook).

## 5. Final checks before submitting (Wednesday)
- Both JSONs exist, produced by mteb; the release states which one is the submission.
- Tag `PRISM_GENAI_HACKATHON_Y2026` on the final commit; PPT named per template; video ≤5 min with access set; Google Form submitted once.
