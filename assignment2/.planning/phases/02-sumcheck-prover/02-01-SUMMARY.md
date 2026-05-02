---
plan: 02-01
phase: 02-sumcheck-prover
status: complete
completed: "2026-05-02"
---

# Summary: 02-01 sumcheck_32 Prover Loop

## What Was Built

Implemented `sumcheck_32` in `student.py` — the full 32-bit SumCheck prover loop. Replaces the `raise NotImplementedError` stub with a working implementation that passes all required correctness tests.

## Implementation

**Algorithm:** For each round `i`:
1. Evaluate round polynomial `g_i(v)` at `v ∈ {0, ..., degree}` using MLE extension via `mle_update_32(table[::2], table[1::2], v, q=q)`.
2. Sum elementwise products across additive terms and multiplicative factors using `mod_mul_32` and `jnp.sum(...uint64) % q`.
3. Fold all 6 tables using `challenges[i]` via `mle_update_32` (skipped on last round).

**Returns:** `(claim0, round_evals)` — both JAX arrays; `claim0 = mod_add_32(g_0(0), g_0(1), q)`.

**Key decisions honored:**
- D-02: LSB-first interleaved indexing (`table[::2]` = even = var=0, `table[1::2]` = odd = var=1)
- D-03: `degree = max(len(term) for term in expression)`, eval at `degree+1` points
- D-04: `claim0` from round-0 evals, no separate pre-loop sum
- D-05: Python for-loop over rounds, `jnp.stack` for output
- D-07: Fold all 6 tables, only for `i < len(challenges)`
- D-08: `round_evals` shape `(num_rounds, degree+1)` via `jnp.stack`

## Test Results

No debugging was needed — implementation passed first run.

| Stage | Command | Result |
|-------|---------|--------|
| vars4 | `uv run pytest --bits 32 --num-vars 4 -q` | 51/51 pass |
| vars16 | `uv run pytest --bits 32 --num-vars 16 -q` | 51/51 pass |
| vars20 | `uv run pytest --bits 32 --num-vars 20 -q` | 51/51 pass |

Expressions tested: `a`, `a*b`, `a*b + c`, `a*b*c` across 5 cases per num-vars size.

## Commits

- `4950a64` feat(02-01): implement sumcheck_32 prover loop

## Self-Check: PASSED

All acceptance criteria met:
- `sumcheck_32` contains no `raise NotImplementedError`
- Uses `table[::2]` / `table[1::2]` interleaved slicing
- Uses `jnp.stack` for `round_evals`
- Uses `mod_add_32` for `claim0`
- Has `if i < len(challenges)` guard before fold
- All three required pytest commands exit 0
