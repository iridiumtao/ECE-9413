---
phase: 01-primitives-mle-update
plan: 01
subsystem: core-arithmetic
tags: [jax, jnp, modular-arithmetic, mle, uint64, sumcheck]

# Dependency graph
requires: []
provides:
  - "mod_add_32: overflow-safe (a+b) mod q via uint64 promotion"
  - "mod_sub_32: underflow-safe (a+q-b) mod q via uint64 promotion"
  - "mod_mul_32: overflow-safe (a*b) mod q via uint64 promotion"
  - "mle_update_32: linear interpolation composing all three primitives"
affects: [02-sumcheck-prover, 03-optimization-submission]

# Tech tracking
tech-stack:
  added: [jax.numpy as jnp]
  patterns:
    - "uint64 intermediate promotion for overflow-safe 32-bit modular arithmetic"
    - "composition pattern: mle_update_32 delegates entirely to three primitives"

key-files:
  created: []
  modified: [student.py]

key-decisions:
  - "D-01: Promote all operands to uint64 before arithmetic; cast result back to uint32"
  - "D-02: mod_sub_32 uses (a + q - b) pattern — NOT (a - b + q) — to avoid uint64 underflow when b > a"
  - "D-03: mle_update_32 formula is zero + target * (one - zero) mod q; composed from mod_sub_32, mod_mul_32, mod_add_32"
  - "No jax.vmap — plain array ops already vectorized by JAX/XLA"

patterns-established:
  - "Pattern 1 (uint64 promotion): Every 32-bit arithmetic function casts all three operands to jnp.uint64 before computing, then casts back to jnp.uint32"
  - "Pattern 2 (safe subtraction): mod_sub_32 uses a64 + q64 - b64 form, never b64 - a64 as first operation"
  - "Pattern 3 (primitive composition): higher-level kernels call mod_*_32 functions directly rather than repeating uint64 casting"

requirements-completed: [PRIM-01, PRIM-02, PRIM-03, MLE-01]

# Metrics
duration: 1min
completed: 2026-04-30
---

# Phase 1 Plan 01: Primitives + MLE Update Summary

**Four 32-bit JAX kernels implemented: mod_add_32, mod_sub_32, mod_mul_32 with uint64 overflow protection, and mle_update_32 composing them for Boolean-variable fold in the SumCheck prover**

## Performance

- **Duration:** 1 min
- **Started:** 2026-04-30T21:22:29Z
- **Completed:** 2026-04-30T21:23:35Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments

- Implemented three overflow-safe 32-bit modular arithmetic primitives using uint64 intermediate promotion (all 30 edge-case parametrized tests pass)
- Implemented `mle_update_32` as a composition of the three primitives; textbook example (F_17, t=8, [0,2,4,6] → [8,10,12,14]) verified
- `import jax.numpy as jnp` added as required; no jax.vmap usage; all optional 64/128-bit stubs remain untouched

## Task Commits

Each task was committed atomically:

1. **Task 1: Add jnp import and implement mod_add_32, mod_sub_32, mod_mul_32** - `2f4629b` (feat)
2. **Task 2: Implement mle_update_32 composing the three primitives** - `2590640` (feat)

**Plan metadata:** (docs commit — see state update)

## Files Created/Modified

- `/Users/oud/Projects/ECE-9413/assignment2/student.py` — Added `import jax.numpy as jnp`; replaced four `raise NotImplementedError` stubs with concrete implementations (mod_add_32 lines 20-25, mod_sub_32 lines 28-33, mod_mul_32 lines 36-41, mle_update_32 lines 122-132)

## Test Results

```
uv run pytest tests/test_sumcheck.py::test_mod_add_32bit_edge_cases \
              tests/test_sumcheck.py::test_mod_sub_32bit_edge_cases \
              tests/test_sumcheck.py::test_mod_mul_32bit_edge_cases \
              --bits 32
30 passed in 0.88s
```

Textbook smoke check:
```
mle_update_32 textbook example: PASS
# F_17 domain, t=8, [0,2,4,6] folds to [8,10,12,14], dtype=uint32
```

## Decisions Made

- **D-01 (uint64 promotion):** All three primitives cast a, b, q to `jnp.uint64` before arithmetic. Required because `uint32` values up to `MAX_PRIME_32 - 1 ≈ 4.29e9`; their sum/product overflows uint32 silently.
- **D-02 (safe subtraction):** `mod_sub_32` uses `(a64 + q64 - b64) % q64` — adding q before subtracting b ensures the intermediate `a64 + q64` is always ≥ b64 within uint64 range, eliminating underflow risk.
- **D-03 (mle formula):** `mle_update_32` computes `zero + target * (one - zero)` by chaining `mod_sub_32 → mod_mul_32 → mod_add_32`. Each call re-enters the uint64 promotion chain, so no intermediate wrapping occurs.

## Deviations from Plan

None — plan executed exactly as written.

The plan's acceptance criterion "grep -c 'TODO(student): implement when enabling' student.py returns 6" reflects a stale count from before `mle_update_64`, `mle_update_128`, `sumcheck_64`, `sumcheck_128` stubs were visible. The actual file has always contained 10 such TODO comments. The 64/128-bit optional stubs are completely untouched — confirmed by verifying that `mod_add_32`, `mod_sub_32`, `mod_mul_32`, `mle_update_32` all have zero `NotImplementedError` lines while all optional stubs retain theirs.

## Issues Encountered

None.

## Hand-off Note for Phase 2

All four 32-bit primitives are ready. `sumcheck_32` may now call `mod_add_32`, `mod_sub_32`, `mod_mul_32`, and `mle_update_32` directly. The Boolean-fold table update kernel (`mle_update_32`) is fully vectorized over table entries and accepts a scalar challenge value as `target_eval`.

## Next Phase Readiness

- Phase 2 (`sumcheck_32`) can begin immediately: all four dependency primitives are implemented and tested
- No blockers
- Run `uv run pytest --bits 32 --num-vars 4` to verify the prover loop once Phase 2 is complete

## Self-Check: PASSED

- student.py: FOUND
- SUMMARY.md: FOUND
- Commit 2f4629b (Task 1): FOUND
- Commit 2590640 (Task 2): FOUND

---
*Phase: 01-primitives-mle-update*
*Completed: 2026-04-30*
