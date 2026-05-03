# Phase 4: Incorporate Peer Optimizations and Benchmark Experimentally - Context

**Gathered:** 2026-05-03
**Status:** Ready for planning

<domain>
## Phase Boundary

Port two specific performance optimizations from the peer implementation (seanshnkim/NTT_Jax, `assignment2/student.py`) into our `student.py`, run benchmarks after each change, and record results in `experiment.md` with git hash tracking.

The two optimizations to port:
1. **t=0/t=1 shortcut** — when evaluating g_i at v=0, use `evens` directly; at v=1, use `odds` directly, skipping `mle_update_32` entirely
2. **Evens/odds pre-computation** — compute `evens`/`odds` slices once per round (not once per evaluation point), and reuse them in both the g_i loop and the fold step

Each optimization is one commit. After each commit: run all three benchmark tiers (N=4, N=16, N=20) and fill in the corresponding experiment table in `experiment.md`.

NOT in scope: Barrett reduction (GPU-only, no benefit on CPU), `used_vars` filtering (harness always provides exactly the needed variables), JIT removal, 64/128-bit tracks.

</domain>

<decisions>
## Implementation Decisions

### Which optimizations to port

- **D-01:** Port the **t=0/t=1 shortcut** — for evaluation point `v=0`, return `evens` directly; for `v=1`, return `odds` directly; only call `mle_update_32` for `v >= 2`. This eliminates all `mle_update_32` calls for degree-1 expressions (`a`, `a*b`).
- **D-02:** Port **evens/odds pre-computation** — compute `{var: tbl[::2]}` and `{var: tbl[1::2]}` once per round at the top of the loop, then reuse both for the t-loop and the fold step. Currently both are recomputed separately.
- **D-03:** **Skip Barrett reduction** — CPU already benefits from XLA's magic-number divide lowering for `% constant`. No measurable CPU benefit; adds ~100 lines of complexity.
- **D-04:** **Skip `used_vars` filtering** — the harness always passes exactly the variables referenced in the expression. Zero practical benefit.

### JIT strategy

- **D-05:** Keep `@jax.jit` on `sumcheck_32` — the entire loop compiled as one XLA program, cached by `(expression, num_rounds)`. Proven: 91 tests pass including `test_jit_matches_eager_on_compulsory_tracks`. Peer's approach (no JIT on outer function) trades compile-time optimization for eager flexibility — not a win on CPU with fixed expression structures.

### Experiment granularity

- **D-06:** **One commit per optimization** — implement t=0/t=1 shortcut → benchmark all three tiers → fill Experiment 01 table in `experiment.md`. Then implement evens/odds reuse → benchmark → fill Experiment 02. This keeps each experiment's git hash cleanly isolated to one change.

### Correctness gate

- **D-07 (Claude's discretion):** After each change, run `uv run pytest --bits 32 --num-vars 4` before running the full benchmark suite. Must still pass 51/51 before proceeding.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Current implementation

- `student.py` — `sumcheck_32` at line 159 (the function to modify); `mle_update_32` at line 124; frozen dispatcher `sumcheck` at line 230
- `experiment.md` — baseline benchmark results (git hash `632526c70eb2cdf398d75ba30afdfd40ff49933d`) and Experiment 01/02 templates to fill in

### Peer implementation (reference)

- `https://github.com/seanshnkim/NTT_Jax` — peer's `assignment2/student.py` (fetch via `gh api` if needed). Key functions to study: `_sumcheck_32_v1` (the default hot path with t=0/t=1 shortcut and evens/odds reuse), `_compose_terms_u32`, `_sum_mod_u32`.

### Test harness

- `tests/test_sumcheck.py` — `test_jit_matches_eager_on_compulsory_tracks` must still pass after changes; `test_sumcheck_matches_expected_outputs` is the main correctness gate
- `CLAUDE.md` §Running Tests — staged test commands and benchmark commands

### Planning and requirements

- `.planning/ROADMAP.md` §Phase 4
- `.planning/phases/02-sumcheck-prover/02-CONTEXT.md` — D-02 (LSB-first interleaved indexing: `[::2]`=zero-eval, `[1::2]`=one-eval), D-03 (degree = max term length), D-06 (uniform mle_update_32 for all v — this is what we're replacing)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets

- `mle_update_32(zero_eval, one_eval, target_eval, *, q)` at `student.py:124` — still used for `v >= 2` (higher-degree evaluation points); not removed, just bypassed for v=0 and v=1
- `mod_add_32`, `mod_sub_32`, `mod_mul_32` at `student.py:22–43` — unchanged; used inside `mle_update_32`

### Established Patterns

- All arithmetic uses uint64 promotion; inputs/outputs uint32 — no change
- `jnp.sum(arr.astype(jnp.uint64)) % q` for safe reduction — no change
- Tables indexed LSB-first: `tbl[::2]` = zero-eval, `tbl[1::2]` = one-eval (D-02 from Phase 2)
- `@jax.jit` with `static_argnames=("expression", "num_rounds")` — preserved

### Integration Points

- The optimization is entirely within the `sumcheck_32` body (lines 160–215)
- Frozen dispatchers (`sumcheck`, `mod_add`, etc.) are untouched
- `sumcheck()` at line 230 already converts expression to tuple-of-tuples for JIT hashing

### What changes, precisely

**Before (current inner loop for evaluation point v):**
```python
for v in range(degree + 1):
    v_scalar = jnp.asarray(v, dtype=jnp.uint32)
    for term in expression:
        term_product = mle_update_32(tables[first_var][::2], tables[first_var][1::2], v_scalar, q=q)
        ...
# fold:
tables = {var: mle_update_32(tbl[::2], tbl[1::2], r, q=q) ...}
```

**After (with both optimizations):**
```python
evens = {var: tbl[::2] for var, tbl in tables.items()}
odds  = {var: tbl[1::2] for var, tbl in tables.items()}
for v in range(degree + 1):
    if v == 0:   vals = evens
    elif v == 1: vals = odds
    else:        vals = {var: mle_update_32(evens[var], odds[var], v_scalar, q=q) ...}
    ...
# fold (reuses already-computed evens/odds):
tables = {var: mle_update_32(evens[var], odds[var], r, q=q) ...}
```

</code_context>

<specifics>
## Specific Ideas

- User explicitly asked to record each experiment with the git hash, what changed, and a result summary table — `experiment.md` format is already bootstrapped with the baseline (Experiment 00) and a template for Experiment 01.
- Experiment 02 (evens/odds reuse) table should be added to `experiment.md` during planning.
- Peer's implementation was fetched via `gh api repos/seanshnkim/NTT_Jax/contents/assignment2/student.py` — can be re-fetched at any time without cloning.

</specifics>

<deferred>
## Deferred Ideas

- **Barrett reduction** — hand-rolled Barrett for GPU (ECE9413_BARRETT=1); revisit if running benchmarks on Modal/GPU
- **64-bit track** — `mod_add_64`, `sumcheck_64` etc.; separate phase if pursued
- **`jax.lax.scan` for fully-fused loop** — was previously deferred in Phase 2; still a future option if loop unrolling overhead becomes measurable at vars20+

</deferred>

---

*Phase: 4-Incorporate Peer Optimizations and Benchmark Experimentally*
*Context gathered: 2026-05-03*
