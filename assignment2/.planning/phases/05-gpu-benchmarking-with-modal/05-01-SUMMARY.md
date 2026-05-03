---
phase: 05-gpu-benchmarking-with-modal
plan: "01"
subsystem: infra
tags: [modal, jax, cuda, gpu, benchmarking, subprocess]

# Dependency graph
requires:
  - phase: 04-incorporate-peer-optimizations-and-benchmark-experimentally
    provides: student.py with t=0/t=1 shortcut + evens/odds reuse optimizations; experiment.md with CPU baseline tables
provides:
  - modal_run.py Modal entrypoint with T4/A100/H100 GPU tier dispatch
  - Reproducible per-hash GPU benchmark workflow via add_local_dir + git checkout pattern
  - Secret-safe image build (ignores .env*, *.key, *.pem, *.secret, *.crt)
affects:
  - 05-02-experiment-gpu (reads modal_run.py to reproduce all three experiments on GPU)

# Tech tracking
tech-stack:
  added: [modal, jax[cuda12]]
  patterns:
    - Three @app.function GPU-tier pattern (T4/A100/H100) dispatched via _GPU_FUNCTIONS dict
    - Shared _run_sumcheck_impl with tests + bench phases controlled by boolean flags
    - add_local_dir with credential-exclusion ignore list for secret safety

key-files:
  created:
    - modal_run.py
  modified: []

key-decisions:
  - "app name 'ece9413-sumcheck' (not ece9413-ntt — project-specific)"
  - "Delegate entirely to tests.benchmark via subprocess.run (no NTT-style _bench_multi_run)"
  - "Single modal invocation runs all three num-vars tiers [4, 16, 20] sequentially"
  - "add_local_dir ignore list includes .env, .env*, *.env.*, *.key, *.pem, *.secret, *.crt to prevent credential upload"
  - "tests=not bench_only, bench=not tests_only logic in local_entrypoint"

patterns-established:
  - "Per-hash GPU benchmark workflow: git checkout <hash> -- student.py → modal run modal_run.py --bench-only → git checkout HEAD -- student.py"
  - "_GPU_FUNCTIONS dispatch dict avoids if/elif chain for GPU tier selection"

requirements-completed: [GPU-01, PERF-02]

# Metrics
duration: 1min
completed: 2026-05-03
---

# Phase 5 Plan 01: Modal GPU Entrypoint Summary

**Modal entrypoint creating T4/A100/H100 GPU benchmark runner for SumCheck via tests.benchmark subprocess delegation, with credential-safe add_local_dir and per-hash reproducibility workflow**

## Performance

- **Duration:** ~1 min
- **Started:** 2026-05-03T21:45:11Z
- **Completed:** 2026-05-03T21:45:55Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments

- Created `modal_run.py` at project root with app name `ece9413-sumcheck`
- JAX CUDA image: `debian_slim(python_version="3.12")` + `jax[cuda12]`, numpy, rich, pytest, sympy
- Three GPU-tier functions (T4/A100/H100) with shared `_run_sumcheck_impl` delegating to `tests.benchmark` via subprocess
- Local entrypoint with `--gpu`, `--tests-only`, `--bench-only`, `--bits` CLI flags
- Secret-exclusion ignore list: `.env`, `.env*`, `*.env.*`, `*.key`, `*.pem`, `*.secret`, `*.crt`

## Task Commits

1. **Task 1: Create modal_run.py** - `ccf7dc5` (feat)

**Plan metadata:** (docs commit pending)

## Files Created/Modified

- `/Users/oud/Projects/ECE-9413/assignment2/modal_run.py` - Modal entrypoint; app `ece9413-sumcheck`; T4/A100/H100 GPU tiers; pytest + benchmark subprocess invocation over num-vars [4, 16, 20]; secret-safe add_local_dir

## Decisions Made

- Delegated entirely to `tests.benchmark` via subprocess (no custom timing loop) — simplest approach, reuses the existing Rich table output
- Single modal invocation runs all three num-vars tiers sequentially — compile once, bench all tiers
- `add_local_dir(".", ...)` uploads the current local working tree — enables per-hash benchmarking via `git checkout <hash> -- student.py` workflow documented in CONTEXT.md D-10

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

To run GPU benchmarks, Modal authentication is required:
```bash
modal setup   # or: pip install modal && modal token new
```
Then:
```bash
modal run modal_run.py                    # tests + bench all tiers on T4
modal run modal_run.py --bench-only       # skip tests, bench all tiers on T4
modal run modal_run.py --tests-only       # pytest only (vars4, 32-bit)
modal run modal_run.py --gpu A100         # run on A100
```

Per-hash experiment reproduction (Wave 2 workflow):
```bash
git checkout 632526c70eb2cdf398d75ba30afdfd40ff49933d -- student.py
modal run modal_run.py --bench-only
git checkout HEAD -- student.py
```

## Next Phase Readiness

- `modal_run.py` is complete and ready for Wave 2 (05-02: GPU experiment runs)
- Modal auth is the only prerequisite for actual GPU invocation (handled as checkpoint in Wave 2 plan)
- All three experiment git hashes from CONTEXT.md D-11 are ready to reproduce on GPU

## Self-Check

- `modal_run.py` exists at `/Users/oud/Projects/ECE-9413/assignment2/modal_run.py` ✓
- Commit `ccf7dc5` exists ✓
- Syntax check passes (`python -c "import ast; ast.parse(...)"`) ✓

## Self-Check: PASSED

---
*Phase: 05-gpu-benchmarking-with-modal*
*Completed: 2026-05-03*
