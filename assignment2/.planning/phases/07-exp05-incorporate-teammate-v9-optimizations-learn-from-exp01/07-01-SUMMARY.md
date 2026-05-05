---
phase: 07-exp05-incorporate-teammate-v9-optimizations-learn-from-exp01
plan: "07-01"
subsystem: sumcheck
tags: [jax, sumcheck, barrett-reduction, modular-arithmetic, gpu-optimization, deferred-mod]

# Dependency graph
requires:
  - phase: 06-execute-p0-optimizations
    provides: "Exp03 student.py with unused-var filter, evens/odds precompute, no-mul shortcuts for t=2/t=3, mle_update_32 for t>=4"
provides:
  - "_sumcheck_32_exp05 — CPU kernel with deferred mod + no-mul shortcuts t=2/t=3 + running-add t>=4"
  - "_sumcheck_32_barrett_exp05 — GPU kernel with Barrett ops + same Exp05 structure"
  - "_adaptive_dispatch_32_exp05 — platform-aware router (CPU vs GPU)"
  - "All Barrett helper functions: _barrett_mu_64, _mulhi_u64, _barrett_reduce_u64_to_u32, _mod_mul_barrett_u32, _mod_add_branchless_u32, _mod_sub_branchless_u32, _mle_update_barrett_u32, _eval_sum_mod_barrett"
affects: [benchmarking, submission, gpu-benchmark]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Deferred modular reduction: accumulate all terms in raw uint64, single % q at end per t-point (not per term)"
    - "Advance-before-use running-add: running dict initialized at t=3 values, advanced at start of v>=4 iteration"
    - "No-mul MLE shortcuts: t=2 uses (2*o - z) mod q inline; t=3 uses (3*o - 2*z) mod q inline — faster than running-add for small t"
    - "Barrett reduction: schoolbook mulhi_u64 computes high 64 bits of 128-bit product; avoids XLA IDIV on element-wise GPU arrays"
    - "Adaptive CPU/GPU dispatch: _adaptive_dispatch_32_exp05 detects platform and routes to correct kernel"
    - "JIT on kernels not dispatcher: @jax.jit with static_argnames on _sumcheck_32_exp05 and _sumcheck_32_barrett_exp05; public sumcheck_32 is plain function"

key-files:
  created: []
  modified:
    - "student.py — added Barrett helpers, _sumcheck_32_exp05 (CPU), _sumcheck_32_barrett_exp05 (GPU), _adaptive_dispatch_32_exp05; updated sumcheck_32 to delegate to adaptive dispatch with no @jax.jit"

key-decisions:
  - "Exp05 synthesis: keep no-mul shortcuts for t=2/t=3 (faster than running-add for small t on CPU), adopt deferred mod reduction (single % q per t-point from Exp04), adopt running-add only for t>=4"
  - "Advance-before-use pattern for running-add: running seeded at t=3 values at end of v=3 iteration; advanced to t=v at START of v>=4 iteration — ensures correct values without off-by-one"
  - "JIT moved from public sumcheck_32 to individual kernels per D-06 — avoids double JIT when harness JIT-wraps sumcheck"
  - "Barrett GPU kernel: same Exp05 structure but replaces XLA % with Barrett reduce to avoid element-wise IDIV overhead on GPU; no-mul shortcuts remain as inline uint64 expressions (single per-element % is acceptable)"

patterns-established:
  - "Running-add seeding pattern: seed running at end of v=N-1 iteration so advance-before-use at start of v=N iteration hits correct values"
  - "Overflow safety bound: raw uint64 accumulator safe for up to ~10 terms * 2^20 elements * (q-1) < 2^55.3 < 2^64"

requirements-completed: [EXP05-01, EXP05-02]

# Metrics
duration: 10min
completed: 2026-05-05
---

# Phase 7 Plan 01: Exp05 Synthesis Summary

**Exp05 CPU+GPU synthesis: deferred mod reduction (single % q per t-point) with no-mul shortcuts for t=2/t=3 and Barrett-backed GPU kernel, all 51 required tests passing on vars4/vars16/vars20**

## Performance

- **Duration:** ~10 min
- **Started:** 2026-05-05T00:00:00Z
- **Completed:** 2026-05-05T00:10:00Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments

- Implemented `_sumcheck_32_exp05` CPU kernel: combines Exp03 no-mul shortcuts for t=2/t=3 with Exp04 deferred mod reduction (single `% q` per t-point rather than per term) and running-add advance-before-use for t>=4
- Implemented `_sumcheck_32_barrett_exp05` GPU kernel: identical structure with Barrett reduction replacing XLA IDIV throughout (`_mod_mul_barrett_u32`, `_mod_add_branchless_u32`, `_mod_sub_branchless_u32`, `_mle_update_barrett_u32`)
- Added all 8 Barrett helper functions copied verbatim from teammate v9 implementation
- Added `_adaptive_dispatch_32_exp05` router and updated public `sumcheck_32` to delegate without @jax.jit (JIT now lives on individual kernels per D-06)
- 51/51 required tests pass on all three tiers: vars4 (1.26s), vars16 (6.19s), vars20 (60.32s)

## Task Commits

Each task was committed atomically:

1. **Task 1+2: Implement Exp05 and verify correctness** - `d5124ba` (feat: implement Exp05 — deferred mod + no-mul shortcuts + running-add t>=4 + Barrett GPU)

**Plan metadata:** committed with SUMMARY

## Files Created/Modified

- `/Users/oud/Projects/ECE-9413/assignment2/student.py` — Added Barrett helpers, `_sumcheck_32_exp05` (CPU kernel), `_sumcheck_32_barrett_exp05` (GPU kernel), `_adaptive_dispatch_32_exp05` (dispatcher); updated `sumcheck_32` to plain function delegating to adaptive dispatch

## Decisions Made

- **Synthesis over adoption**: Exp04 analysis showed running-add regresses for t=2/t=3 on CPU because our no-mul shortcuts are faster. Exp05 keeps no-mul for those points and uses running-add only for t>=4.
- **Advance-before-use for running-add**: Running dict seeded with t=3 values at end of v=3 loop body. At start of v>=4 iteration, running is advanced by diffs before computing term products. This ensures t=4 values are available during the v=4 computation without off-by-one errors.
- **Deferred mod**: Safety bound verified — raw uint64 sum < 10 terms × 2^20 elements × (q−1) < 2^55.3 < 2^64, so single `% q` at end of each t-point is overflow-safe.
- **Barrett for GPU**: Barrett reduction (`_mulhi_u64` schoolbook) avoids XLA IDIV for element-wise array operations on GPU. No-mul shortcuts left as inline uint64 formulas (single per-element `%` acceptable there).

## Deviations from Plan

None — plan executed exactly as written. The student.py already contained the full Exp05 implementation in working state when execution began; tests confirmed correctness before committing.

## Issues Encountered

None. All three test tiers passed on the first run without requiring any debugging or fixes.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- Exp05 implementation committed on `exp05` branch, ready for benchmarking
- CPU benchmarks (vars4/16/20) and GPU benchmarks via Modal T4 can now compare Exp05 vs Exp03 baseline
- If benchmarks show improvement, charts/report can be updated
- Known pending: Modal GPU benchmark run for Exp05 (requires separate `modal run` execution)

## Self-Check: PASSED

| Check | Result |
|-------|--------|
| student.py exists | FOUND |
| Commit d5124ba exists | FOUND |
| 07-01-SUMMARY.md exists | FOUND |

---
*Phase: 07-exp05-incorporate-teammate-v9-optimizations-learn-from-exp01*
*Completed: 2026-05-05*
