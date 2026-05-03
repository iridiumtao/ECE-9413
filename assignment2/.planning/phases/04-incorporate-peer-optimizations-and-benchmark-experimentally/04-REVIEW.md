---
phase: 04-incorporate-peer-optimizations-and-benchmark-experimentally
reviewed: 2026-05-03T00:00:00Z
depth: standard
files_reviewed: 2
files_reviewed_list:
  - student.py
  - experiment.md
findings:
  critical: 0
  warning: 1
  info: 2
  total: 3
status: issues_found
---

# Phase 4: Code Review Report

**Reviewed:** 2026-05-03
**Depth:** standard
**Files Reviewed:** 2
**Status:** issues_found

## Summary

Phase 4 introduced two optimizations to `sumcheck_32`: a `t=0`/`t=1` shortcut that bypasses `mle_update_32` for the two integer evaluation points, and an evens/odds pre-computation dict built once per round. Both optimizations are mathematically correct. The SumCheck protocol invariants are preserved: the fold step reuses the same evens/odds slices that were used for the g_i evaluation, the challenge indexing matches the `num_rounds - 1` contract, and all modular arithmetic stays within `uint64` bounds.

One warning-level quality defect was found: a constant is materialized inside the inner term loop rather than at the top of its enclosing branch. Two info-level findings cover a comment with off-by-one exponents and a missing conclusion section in the experiment log.

---

## Warnings

### WR-01: `v_scalar` Created Inside Inner Term Loop at Line 191

**File:** `student.py:191`

**Issue:** `v_scalar = jnp.asarray(v, dtype=jnp.uint32)` is placed inside the `for term in expression` loop (the innermost loop), so it is created once per term per `v >= 2` value. Because `v` is a static Python integer (not a JAX traced value), JAX's JIT compiler _may_ deduplicate it via constant folding, but this is not guaranteed across all JAX versions and backends. On CPU the compiler currently coalesces identical integer constants, but the code creates an unnecessary dependency chain during tracing that could prevent deduplication on some accelerators. The correct placement is immediately before the `for term in expression` loop, where it is computed once per `v >= 2` iteration.

**Fix:**
```python
# Before:
else:
    # v >= 2: full linear interpolation required.
    for term in expression:
        v_scalar = jnp.asarray(v, dtype=jnp.uint32)  # created k times
        term_product = mle_update_32(...)

# After:
else:
    # v >= 2: full linear interpolation required.
    v_scalar = jnp.asarray(v, dtype=jnp.uint32)  # created once per v
    for term in expression:
        term_product = mle_update_32(
            evens[term[0]], odds[term[0]], v_scalar, q=q,
        )
        for var in term[1:]:
            factor = mle_update_32(evens[var], odds[var], v_scalar, q=q)
            term_product = mod_mul_32(term_product, factor, q)
```

---

## Info

### IN-01: Inaccurate Overflow-Bound Comment at Line 199

**File:** `student.py:199`

**Issue:** The comment reads `"Safe uint64 sum: up to 2^20 entries, each < q < 2^32, sum < 2^52 < 2^64."` Both numeric claims are off by one power of two. The table is sliced into evens/odds before `jnp.sum` is called, so the largest half-table that can appear is `2^(num_vars-1) = 2^19` entries (at round 0 with `num_vars=20`), not `2^20`. The resulting bound on the sum is `2^19 * (2^32 - 1) < 2^51`, not `2^52`. The overall claim — that the sum fits safely in a `uint64` — is still correct; only the stated bound is wrong.

**Fix:**
```python
# Before:
# Safe uint64 sum: up to 2^20 entries, each < q < 2^32, sum < 2^52 < 2^64.

# After:
# Safe uint64 sum: evens/odds each have at most 2^19 entries (half of 2^20),
# each entry < q < 2^32, so sum < 2^51 << 2^64.
```

### IN-02: Experiment Log Missing Conclusion for Experiment 02

**File:** `experiment.md:99–146`

**Issue:** The Experiment 02 section presents benchmark data but provides no conclusion, adoption statement, or analysis. The data itself shows a measurable regression for multi-term expressions at `num_vars=20`:

| Expression | Exp 01 (ms) | Exp 02 (ms) | Delta |
|---|---|---|---|
| `a*b + c` | 5.241 | 5.538 | +5.7% (slower) |
| `a*b*c` | 6.508 | 6.667 | +2.4% (slower) |

The current `student.py` contains the Experiment 02 code (both shortcuts combined), but the log never records whether this trade-off was accepted, attributed to measurement noise, or whether a follow-up experiment was planned. A reader cannot determine from the log alone why the regression case was adopted. A brief conclusion paragraph noting the decision and reasoning should be added.

**Fix:** Append a conclusion to `experiment.md` Experiment 02 section, for example:
```
### Conclusion

Experiment 02 is adopted as the combined phase-4 implementation. The regression on
`a*b + c` (+5.7%) and `a*b*c` (+2.4%) at N=20 is within measurement noise range
(p90 spread in Exp 01 is already ~5%) and outweighed by the correctness simplification
of slicing each table exactly once per round. All other expressions show improvement or
are neutral.
```

---

_Reviewed: 2026-05-03_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
