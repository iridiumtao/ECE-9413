# Phase 5: GPU Benchmarking with Modal - Context

**Gathered:** 2026-05-03
**Status:** Ready for planning

<domain>
## Phase Boundary

Create `modal_run.py` in the project root that uploads the current working tree to a Modal GPU instance and runs the SumCheck benchmark suite there. Then re-run all three experiments from `experiment.md` (Baseline, Experiment 01, Experiment 02) on GPU by checking out each tagged git hash locally and invoking `modal run`. Record GPU results in `experiment_gpu.md` with the same section/table structure as `experiment.md`, plus CPU-vs-GPU delta tables.

NOT in scope: modifying `student.py`, changing test data, adding new optimizations (Barrett, etc.), 64/128-bit tracks.

</domain>

<decisions>
## Implementation Decisions

### Modal image configuration

- **D-01:** Use `modal.Image.debian_slim(python_version="3.12")` as base — matches NTT_Jax precedent.
- **D-02:** Install `jax[cuda12]`, `numpy`, `rich`, `pytest`, `sympy` via `pip_install` — same package list as `pyproject.toml` dependencies, replacing `jax[cpu]` with `jax[cuda12]`.
- **D-03:** Bundle the project with `add_local_dir(".", remote_path="/root/assignment2", ignore=["*.egg-info", "__pycache__", ".git", "*.pyc", "code.zip", "*.pdf", "*.pptx"])` — NTT_Jax pattern; current local directory is what gets uploaded, which is how per-hash benchmarking is handled (checkout locally, then run modal).
- **D-04:** `sys.path.insert(0, "/root/assignment2")` inside the remote function — same as NTT_Jax.

### GPU tier

- **D-05:** Default GPU is T4 — cheapest Modal tier, sufficient for a vars20 SumCheck (arrays are at most ~4 MB), same as NTT_Jax default.
- **D-06:** Provide A100 and H100 variants via separate `@app.function(gpu=...)` callables (same three-tier pattern as NTT_Jax) so any tier can be selected via `--gpu` flag.

### Benchmark invocation

- **D-07:** The remote function runs `python -m tests.benchmark --bench --bits 32 --num-vars {N} --runs 8 --warmup 3` via `subprocess.run` (or direct module import + `sys.argv` injection). All three num-vars tiers (4, 16, 20) are run sequentially by the local entrypoint in a single modal call.
- **D-08:** Also support `--tests-only` mode that runs `pytest --bits 32 --num-vars 4` to verify correctness before benchmarking.
- **D-09:** The local entrypoint signature: `main(gpu="T4", tests=True, bench=True, bits=32)` — simple, mirrors NTT_Jax's CLI surface.

### Experiment reproduction approach

- **D-10:** Per-hash workflow: `git checkout <hash> -- student.py` locally → `modal run modal_run.py` (bundles local dir) → record output → `git checkout HEAD -- student.py`. This is the simplest approach and keeps `modal_run.py` on the current branch without needing to stash/restore the whole tree.
- **D-11:** The three hashes to reproduce in order:
  - Baseline: `632526c70eb2cdf398d75ba30afdfd40ff49933d`
  - Experiment 01 (t=0/t=1 shortcut): `86e0b5f94092bb3e7a8344ee7de04d0057f0d336`
  - Experiment 02 (evens/odds reuse): `2c31b3ed2afefa129b5ce9d6bb5ad50d1aa786e2`

### experiment_gpu.md format

- **D-12:** Mirror `experiment.md` exactly — same section headings (Baseline, Experiment 01, Experiment 02), same result tables (num-vars 4/16/20 with Compile/Median/p90/Mpts/s columns).
- **D-13:** File header: `Device: GPU (Modal T4)` instead of `Device: CPU (Apple Silicon, no GPU)`.
- **D-14:** Append a **CPU vs GPU delta table** at the end of each experiment section showing speedup/regression for N=20 median (the most meaningful tier for GPU).

### Claude's Discretion

- How to capture subprocess output from Modal remote function — use `subprocess.run(..., capture_output=True, text=True)` and `print(result.stdout)`, same as how NTT_Jax prints results.
- Whether to run all three num-vars tiers in a single modal invocation or one invocation per tier — single invocation (compile once, warm up once, then run all three in sequence).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Modal reference implementation

- `/Users/oud/Projects/NTT_Jax/assignment1/modal_run.py` — Full Modal script to model; pay attention to: `modal.Image` construction, `add_local_dir` call, `_run_impl` pattern, `@app.function(gpu=...)` for T4/A100/H100, `@app.local_entrypoint()` with `--gpu` flag, `sys.path.insert` inside remote function.

### Current project structure

- `student.py` — the file being benchmarked (only this file changes between hash checkouts)
- `tests/benchmark.py` — benchmark module entry point; invoked as `python -m tests.benchmark --bench --bits 32 --num-vars {N} --runs 8 --warmup 3`; outputs a Rich table to stdout
- `pyproject.toml` — lists all dependencies to replicate in Modal image
- `experiment.md` — CPU baseline and Experiment 01/02 tables; GPU results mirror this format
- `provided.py`, `sumcheck_utils.py` — read-only harness files that must also be bundled

### Git hashes for reproduction

- Baseline hash: `632526c70eb2cdf398d75ba30afdfd40ff49933d`
- Experiment 01 hash: `86e0b5f94092bb3e7a8344ee7de04d0057f0d336`
- Experiment 02 hash: `2c31b3ed2afefa129b5ce9d6bb5ad50d1aa786e2`

### Planning and requirements

- `.planning/ROADMAP.md` §Phase 5
- `.planning/phases/04-incorporate-peer-optimizations-and-benchmark-experimentally/04-CONTEXT.md` — D-05 (keep @jax.jit), D-07 (correctness gate: 51/51 vars4 tests before benchmarking)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets

- `tests/benchmark.py` — already CLI-ready; `python -m tests.benchmark --bench --bits 32 --num-vars 20 --runs 8 --warmup 3` produces the full Rich table. No modification needed; just invoke it inside the Modal remote function.
- `tests/` directory — all test data (`tests/data/`) must be bundled in the Modal image (included via `add_local_dir(".")`).

### Established Patterns

- NTT_Jax's `modal_run.py` is the direct template — same app name pattern (`ece9413-*`), same image construction, same three-GPU-tier pattern, same `sys.path.insert` inside remote function.
- All arithmetic uses `jax[cuda12]` on Modal (replaces `jax[cpu]` from `pyproject.toml`). JAX auto-detects the GPU device; no `JAX_PLATFORMS` env var needed inside Modal.

### Integration Points

- `modal_run.py` lives in the project root alongside `student.py` and `pyproject.toml`.
- The benchmark harness imports `student` (not `student_module` like NTT_Jax) — no module injection needed; `sys.path.insert(0, "/root/assignment2")` + `import tests.benchmark` is sufficient.
- Modal app name: `"ece9413-sumcheck"`.

</code_context>

<specifics>
## Specific Ideas

- User confirmed: "you may create a new branch first and merge the experiments step-by-step" — create `feat/modal-gpu` branch, add `modal_run.py` in Wave 1, merge; then do Wave 2 experiments on main.
- User wants reproducibility preserved: "ensure you don't break the git hash recorded. However, you can edit the git hash if that is easier for you to do." — this means: the git hashes in `experiment.md` must remain valid/reachable; for `experiment_gpu.md`, the same hashes are used (no need for new ones).
- "Keep in mind we want to ensure the reproducibility of all experiments on both local CPU and remote GPU." — `modal_run.py` must document the exact `modal run` command to reproduce each result, just as `experiment.md` documents the `uv run python -m tests.benchmark` command.

</specifics>

<deferred>
## Deferred Ideas

- **Barrett reduction for GPU** — mentioned in Phase 4 deferred ideas; adds complexity and breaks like-for-like comparison; revisit only if CPU-vs-GPU delta is smaller than expected and a new experiment is warranted.
- **A100/H100 experiments** — `modal_run.py` will support them via `--gpu A100/H100`; collecting `experiment_gpu.md` data on T4 first is sufficient for the assignment.
- **64-bit track on GPU** — deferred from earlier phases; still deferred.

</deferred>

---

*Phase: 5-GPU Benchmarking with Modal*
*Context gathered: 2026-05-03*
