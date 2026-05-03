# Phase 5: GPU Benchmarking with Modal - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-03
**Phase:** 5-GPU Benchmarking with Modal
**Mode:** --auto (all areas auto-resolved)
**Areas discussed:** Modal image config, GPU tier, Benchmark invocation, Experiment reproduction, experiment_gpu.md format, Branch strategy

---

## Modal image configuration

| Option | Description | Selected |
|--------|-------------|----------|
| `jax[cuda12]` on debian_slim + add_local_dir | Mirrors NTT_Jax modal_run.py pattern exactly | ✓ |
| pip install from pyproject.toml via uv | Requires uv in image, more complex | |

**Auto-selected:** `jax[cuda12]` + debian_slim (recommended default, matches NTT_Jax precedent)
**Notes:** Package list is identical to pyproject.toml except `jax[cpu]` → `jax[cuda12]`.

---

## GPU tier

| Option | Description | Selected |
|--------|-------------|----------|
| T4 | Cheapest Modal tier, same as NTT_Jax default, sufficient for vars20 | ✓ |
| A100 | More memory, faster; overkill for this workload | |
| H100 | Fastest, most expensive | |

**Auto-selected:** T4 default; A100/H100 available via `--gpu` flag
**Notes:** vars20 table is ~4 MB; T4 has ample memory.

---

## Benchmark invocation

| Option | Description | Selected |
|--------|-------------|----------|
| subprocess.run `python -m tests.benchmark` | Reuses existing CLI, captures Rich output cleanly | ✓ |
| Import benchmark module directly | More coupling, harder to capture Rich output | |

**Auto-selected:** subprocess.run (recommended, keeps benchmark logic unchanged)
**Notes:** All three num-vars tiers (4, 16, 20) run sequentially within one modal function call.

---

## Experiment reproduction approach

| Option | Description | Selected |
|--------|-------------|----------|
| checkout hash locally → modal run (bundles current dir) → restore | Simple, no Modal logic for git, human-readable workflow | ✓ |
| Pass git hash to Modal, clone repo inside remote function | Complex, requires git in image | |

**Auto-selected:** Local checkout + modal run (recommended, simplest approach)
**Notes:** `git checkout <hash> -- student.py` then `modal run modal_run.py`, then `git checkout HEAD -- student.py`.

---

## experiment_gpu.md format

| Option | Description | Selected |
|--------|-------------|----------|
| Mirror experiment.md exactly + CPU/GPU delta table | Same structure, easy comparison | ✓ |
| Summary table only | Loses per-expression detail | |

**Auto-selected:** Mirror experiment.md + delta table (recommended)
**Notes:** Device header changed to "GPU (Modal T4)"; CPU vs GPU delta appended per experiment.

---

## Claude's Discretion

- Subprocess vs direct import for benchmark invocation — subprocess chosen for clean output capture.
- Whether all three num-vars tiers run in one invocation or three — single invocation.

## Deferred Ideas

- Barrett reduction for GPU (was already in Phase 4 deferred; still deferred here)
- A100/H100 experiment data collection
- 64-bit track on GPU
