---
phase: 03-jax-optimization-submission
plan: "01"
subsystem: jax-optimization
tags: [jax, jit, functools, sumcheck, benchmark, finite-fields]

# Dependency graph
requires:
  - phase: 02-sumcheck-prover
    provides: sumcheck_32 implementation passing all vars4/vars16/vars20 tests
provides:
  - JIT-decorated sumcheck_32 with static_argnames for expression and num_rounds
  - Benchmark timing data for vars4 (N=16), vars16 (N=65536), vars20 (N=1048576)
affects: [03-02-submission]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Convert list[list[str]] expression to tuple[tuple[str]] before passing as JIT static arg (hashability requirement)"
    - "functools.partial(jax.jit, static_argnames=...) for specializing on Python-level constants"

key-files:
  created: []
  modified:
    - student.py

key-decisions:
  - "D-05: expression converted to tuple-of-tuples in sumcheck() dispatcher so JIT static hashing succeeds — list[list[str]] is unhashable"
  - "D-06: JIT decorator placed on sumcheck_32; outer benchmark harness JIT inlines it (nested JIT, no double-compilation)"

patterns-established:
  - "JIT static arg pattern: convert mutable Python containers to immutable equivalents before the JIT boundary"

requirements-completed: [BENCH-01, BENCH-02, BENCH-03]

# Metrics
duration: 5min
completed: "2026-05-03"
---

# Phase 3 Plan 01: JIT Optimization + Benchmarks Summary

**`sumcheck_32` decorated with `jax.jit(static_argnames=("expression","num_rounds"))` and benchmarked at vars4/vars16/vars20 — all 91 tests pass, timing captured for report.pdf**

## Performance

- **Duration:** 5min
- **Started:** 2026-05-03T19:46:31Z
- **Completed:** 2026-05-03T19:51:44Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments

- Added `import functools` and `@functools.partial(jax.jit, static_argnames=("expression", "num_rounds"))` decorator to `sumcheck_32`
- Fixed hashability blocker by converting `expression` from `list[list[str]]` to `tuple[tuple[str]]` in the `sumcheck()` dispatcher
- Ran all three required benchmarks (vars4, vars16, vars20 — 8 runs, 3 warmup each) — all complete without error
- 91 total tests pass (60 sumcheck case tests + 31 primitive/JIT edge-case tests)

## Benchmark Results

**vars4 (N=16, 32-bit):**
- Compile: 95–579 ms | Median: 0.014–0.105 ms | p90: 0.015–0.232 ms

**vars16 (N=65536, 32-bit):**
- Compile: 263–1197 ms | Median: 0.183–1.001 ms | p90: 0.218–1.518 ms
- Throughput: 65–357 Mpts/s depending on expression

**vars20 (N=1048576, 32-bit):**
- Compile: 324–1860 ms | Median: 1.09–9.34 ms | p90: 1.11–19.2 ms
- Throughput: 112–959 Mpts/s depending on expression

## Task Commits

Each task was committed atomically:

1. **Task 1: Add JIT decorator to sumcheck_32 and run full test suite** - `463f10f` (feat)

Task 2 required no file changes — benchmark output is captured in session log above.

**Plan metadata:** (docs commit — see final commit below)

## Files Created/Modified

- `student.py` - Added `import functools`, JIT decorator on `sumcheck_32`, tuple conversion in `sumcheck()` dispatcher

## Decisions Made

- D-05: `expression` must be converted to `tuple[tuple[str]]` before it reaches the JIT boundary. JAX requires static args to be hashable; `list[list[str]]` raises `TypeError: unhashable type: 'list'`. The conversion is placed in the dispatcher (`sumcheck()`) so the public API is unchanged.
- D-06: JIT is applied to `sumcheck_32` directly (inner JIT). The benchmark harness applies an outer JIT over the lambda wrapping `student.sumcheck`. JAX inlines nested JITs — no double-compilation occurs; the inner JIT specializes on expression structure at first trace.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed unhashable expression type blocking JIT static arg**
- **Found during:** Task 1 (Add JIT decorator and run tests)
- **Issue:** `expression` is `list[list[str]]` which raises `TypeError: unhashable type: 'list'` when JAX tries to hash it as a static argument.
- **Fix:** Added `expression = tuple(tuple(term) for term in expression)` in the `sumcheck()` dispatcher, before the `sumcheck_32` call. The body of `sumcheck_32` iterates over expression with `for term in expression:` — tuples support iteration identically to lists, so no body changes were needed.
- **Files modified:** `student.py` (sumcheck dispatcher, 1 line added)
- **Verification:** Full test suite re-run — 91 tests pass, 0 failed.
- **Committed in:** `463f10f` (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 bug fix)
**Impact on plan:** Required fix for correctness — JIT decorator is non-functional without it. No scope creep.

## Issues Encountered

- `list[list[str]]` is not hashable by JAX's static argument mechanism. Plan specified the decorator but did not specify the tuple conversion. Fix was minimal and backward-compatible.

## User Setup Required

None - no external service configuration required.

## Known Stubs

None - all implementation is complete and wired.

## Threat Flags

None — no new network endpoints, auth paths, file access patterns, or schema changes introduced.

## Self-Check

Checking created/modified files and commits exist:

- `student.py`: Present and contains `import functools`, JIT decorator, and tuple conversion.
- Commit `463f10f`: Task 1 commit (feat: add JIT decorator).

## Self-Check: PASSED

## Next Phase Readiness

- `student.py` is JIT-optimized and all required tests pass
- Benchmark timing data is captured in this session (see Benchmark Results above)
- Ready for Plan 02: make_submission.sh and final submission packaging

---
*Phase: 03-jax-optimization-submission*
*Completed: 2026-05-03*
