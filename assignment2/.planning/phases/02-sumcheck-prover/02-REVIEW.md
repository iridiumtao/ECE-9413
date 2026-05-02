---
phase: 02-sumcheck-prover
reviewed: 2026-05-02T00:00:00Z
depth: standard
files_reviewed: 1
files_reviewed_list:
  - student.py
findings:
  critical: 0
  warning: 3
  info: 4
  total: 7
status: issues_found
---

# Phase 02: Code Review Report

**Reviewed:** 2026-05-02
**Depth:** standard
**Files Reviewed:** 1
**Status:** issues_found

## Summary

`student.py` implements the 32-bit SumCheck prover using JAX. The core protocol logic — MLE fold, round polynomial evaluation, fold-then-challenge sequencing, and claim0 derivation — is mathematically correct. The uint64 promotion strategy for safe intermediate arithmetic is sound, and the fold timing relative to challenge consumption correctly follows the API contract.

Three warnings were found: two unguarded conditions that each crash with a confusing error message rather than a clear protocol violation report, and one intra-function variable that silently carries the wrong dtype identity into downstream code. Four informational items cover a stale comment, unused computation, and the expected `NotImplementedError` stubs.

No BLOCKER-level defects were found in the 32-bit compulsory path.

---

## Warnings

### WR-01: Empty expression crashes with `ValueError: max() arg is an empty sequence`

**File:** `student.py:159`
**Issue:** `degree = max(len(term) for term in expression)` raises a bare `ValueError` from the Python built-in `max()` when `expression == []`. The caller receives no indication of which argument is invalid or why. While an empty expression is outside the stated API contract, the failure mode is an opaque standard-library error rather than a clear precondition violation.
**Fix:**
```python
if not expression:
    raise ValueError("expression must contain at least one additive term")
degree = max(len(term) for term in expression)
```

---

### WR-02: `num_rounds == 0` causes `IndexError` at `claim0` computation

**File:** `student.py:210`
**Issue:** If `num_rounds` is 0 (zero challenges), the `for i in range(num_rounds)` loop never executes, leaving `round_evals_list = []`. The subsequent `round_evals_list[0]` on line 210 raises an `IndexError` with no context. The harness always passes `num_rounds >= 1`, but the function silently depends on that invariant with no guard.
**Fix:**
```python
if num_rounds == 0:
    raise ValueError("num_rounds must be >= 1")
```
Place this guard before the for loop on line 166.

---

### WR-03: `g_i_at_v` accumulates in `uint64` but the intermediate `% q` is computed against a freshly constructed `jnp.asarray(q, ...)` object on every iteration

**File:** `student.py:192-193`
**Issue:** Inside the `for term in expression` loop, the modulus `jnp.asarray(q, dtype=jnp.uint64)` is reconstructed on every iteration of both the outer `v` loop and the inner `term` loop. Under `jax.jit`, JAX will deduplicate these constants during tracing, so there is no correctness defect. In eager mode, however, each call allocates a fresh scalar device array. More importantly, `g_i_at_v` is typed as `jnp.uint64` while `term_sum` is also `jnp.uint64`, but the inline construction `jnp.asarray(0, dtype=jnp.uint64)` on line 171 is repeated each time the inner loop is entered. If a later refactor changes the dtype on one branch but not the other, silent integer promotion may produce wrong results. Hoisting the constant out of the loop removes the allocation overhead and eliminates the fragile dtype coupling.
**Fix:**
```python
q64 = jnp.asarray(q, dtype=jnp.uint64)   # hoist once before the outer for loop

# inside the inner loop, replace:
#   % jnp.asarray(q, dtype=jnp.uint64)
# with:
#   % q64
```

---

## Info

### IN-01: Comment on line 191 overstates the maximum number of table entries

**File:** `student.py:191`
**Issue:** The comment reads `"up to 2^20 entries"`. In round 0, each table has `2^num_vars` entries, but `tables[var][::2]` selects every other entry, giving `2^(num_vars-1)` entries. For `num_vars=20`, the actual maximum is `2^19 = 524288`, not `2^20`. The overflow bound still holds (the sum fits in `uint64`), but the comment is inaccurate and could mislead future readers.
**Fix:** Update the comment to read:
```
# Safe uint64 sum: at most 2^(num_vars-1) entries per round,
# each < q < 2^32, so sum < 2^(num_vars-1) * 2^32 <= 2^51 < 2^64.
```

---

### IN-02: All variables in `eval_tables` are folded even when a variable is absent from the expression

**File:** `student.py:202-205`
**Issue:** The fold step iterates over all keys in `tables`, which mirrors all keys from `eval_tables`. If `eval_tables` contains variables not referenced in `expression` (e.g., `eval_tables` always contains all six canonical variables `a..g` but `expression` only uses `a` and `b`), the code folds every unused variable every round. This is not incorrect — unused variables are never read — but it wastes memory operations proportional to the number of unused variables times the number of rounds.
**Fix (optional):** Pre-filter `tables` to only include variables actually appearing in the expression:
```python
used_vars = {var for term in expression for var in term}
tables = {var: jnp.asarray(table, dtype=jnp.uint32)
          for var, table in eval_tables.items()
          if var in used_vars}
```

---

### IN-03: `TODO` comments in unimplemented optional paths

**File:** `student.py:50, 56, 62, 137, 142` (and all optional `_64`/`_128` stubs)
**Issue:** Each unimplemented function body contains `# TODO(student): implement when enabling ...`. These are expected per the assignment scope ("64-bit and 128-bit kernels are intentionally left unimplemented here") and do not affect the compulsory 32-bit path. Flagged for completeness.
**Fix:** No action required for base-track submission. Remove or implement before enabling the optional tracks.

---

### IN-04: Magic number `0` used as initial accumulator for `g_i_at_v` without named constant

**File:** `student.py:171`
**Issue:** `g_i_at_v = jnp.asarray(0, dtype=jnp.uint64)` uses a literal `0` as the additive identity. This is universally understood and not a real defect, but naming it (or at least co-locating it with the `q64` constant from WR-03's fix) would make the zero-initialization pattern explicit and guard against accidental re-initialization of a non-zero value in future edits.
**Fix:** No code change required. If WR-03 is addressed, this zero literal will remain readable in context.

---

_Reviewed: 2026-05-02_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
