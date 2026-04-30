# Phase 2: SumCheck Prover - Context

**Gathered:** 2026-04-30
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement `sumcheck_32` in `student.py` — the full prover loop — and pass all three required test commands:
```
uv run pytest --bits 32 --num-vars 4
uv run pytest --bits 32 --num-vars 16
uv run pytest --bits 32 --num-vars 20
```
This covers all 4 base expressions (`a`, `a*b`, `a*b+c`, `a*b*c`). Advanced track (`--enable-challenge32`) is out of scope for Phase 2.

</domain>

<decisions>
## Implementation Decisions

### Scope

- **D-01:** Required track only — target expressions `a`, `a*b`, `a*b+c`, `a*b*c` on vars4, vars16, vars20. Advanced expressions (`a*a*b*b*c`, `a*b*c+d*e`, `a*b*c*g+d*e*g`) are left for Phase 3 extra credit, if time permits. The generic algorithm handles all expressions; this decision is about what to debug and validate in Phase 2.

### Table indexing convention (locked from test-data verification)

- **D-02:** Tables use **LSB-first (interleaved) indexing** — NOT the "first half = zero-eval, second half = one-eval" description in ROADMAP.md. The ROADMAP description is wrong.
  - Even indices (0, 2, 4, ...) → current variable = 0
  - Odd indices (1, 3, 5, ...) → current variable = 1
  - Verified from ground-truth round tables: `r1_a[j] = mle_update_32(r0_a[2j], r0_a[2j+1], challenge, q)` ✓
  - Extract slices as: `zero_slice = table[::2]`, `one_slice = table[1::2]`

### Round polynomial degree (locked from test-data verification)

- **D-03:** The round polynomial `g_i(X)` has **degree = max term length** across all additive terms in the expression. Evaluation points are `{0, 1, ..., degree}`.
  - `a` → degree 1 → eval at {0, 1} → `round_evals[i]` has 2 elements
  - `a*b` → degree 2 → eval at {0, 1, 2} → 3 elements
  - `a*b+c` → degree 2 (max term `a*b`) → 3 elements
  - `a*b*c` → degree 3 → eval at {0, 1, 2, 3} → 4 elements
  - Evaluation at integer `v` via: `table_at_v[j] = mle_update_32(table[2j], table[2j+1], v, q)`
  - Verified from ground truth: `g_0(2)` for `a*b` = `sum(mle_ext_a_at_2 * mle_ext_b_at_2)` = 3426062706 ✓

### Claim0 derivation

- **D-04:** `claim0 = mod_add_32(round_evals[0][0], round_evals[0][1], q)` — derived from round 0 evals after they are computed. Mathematically: `claim0 = g_0(0) + g_0(1) mod q = sum over all 2^n boolean inputs`. Verified from test data: `(3392080849 + 2564444893) % 3603169181 = 2353356561 = stored claim0` ✓. No extra pre-loop pass needed.

### JIT strategy

- **D-05:** Use a **Python for-loop** over rounds with `jnp.stack` to assemble `round_evals`. JAX traces through Python loops at JIT time (unrolls them at compile time). This satisfies `test_jit_matches_eager_on_compulsory_tracks` without requiring `jax.lax.scan`. Keep it simple — correctness first.

### Round polynomial computation pattern

- **D-06 (Claude's discretion):** For each evaluation point `v ∈ {0, 1, ..., degree}`:
  1. For each additive term in expression: compute elementwise product over variables, accumulate sum
  2. Sum across additive terms
  - Use `mle_update_32(table[::2], table[1::2], v, q)` uniformly for all `v` (works for v=0 and v=1 too, since MLE extension at boolean points is identity)
  - Sum via `jnp.sum(product.astype(jnp.uint64)) % q` — safe for up to 2^20 entries (max sum ≈ 2^52 < 2^64)
  - After round i: fold ALL tables in eval_tables using `challenges[i]` (not just variables in expression)

### Table folding

- **D-07 (locked from test data):** After computing round `i` evals, fold all tables:
  `new_table[var] = mle_update_32(table[var][::2], table[var][1::2], challenges[i], q)`
  Applied for `i < len(challenges)` (i.e., for the first `num_rounds - 1` rounds). The last round does not fold.

### Output format

- **D-08:** Return `(claim0, round_evals)` where:
  - `claim0`: scalar `jnp.uint32`
  - `round_evals`: 2D JAX array of shape `(num_rounds, degree+1)` with dtype `jnp.uint32`
  - Built by `jnp.stack(round_evals_list)` where each element is `jnp.stack(g_i_evals)`

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Assignment harness and API

- `student.py` — `sumcheck_32` stub at line 157; frozen dispatcher `sumcheck` at line 174; primitives from Phase 1 at lines 20–132
- `provided.py` — `expression_round_trace()` debug helper (vars4 only); `VARIABLE_NAMES = ("a","b","c","d","e","g")`; `EXPRESSIONS` list with all 7 expressions
- `tests/test_sumcheck.py` — `test_sumcheck_matches_expected_outputs` (main correctness test); `test_jit_matches_eager_on_compulsory_tracks` (JIT consistency); `_verifier_consistency_check` (validates `g_i(0)+g_i(1)==claim` each round); `_extract_claim0_and_round_evals` (validates return types are JAX arrays)
- `tests/case_utils.py` — `expected_entry()` for understanding the expected output schema; `load_tables()` for table loading

### Planning and requirements

- `.planning/ROADMAP.md` §Phase 2 — task list (NOTE: "split in half" description is incorrect; actual split is interleaved per D-02)
- `.planning/REQUIREMENTS.md` — SC-01 through SC-04, CORR-01 through CORR-03
- `.planning/PROJECT.md` §Context — frozen API constraints, challenge format, expression format
- `.planning/phases/01-primitives-mle-update/01-CONTEXT.md` — Phase 1 decisions (D-01 through D-03, dtype promotion patterns)

### Test data (for debug)

- `tests/data/vars4/v4_case32_0_meta.json` — includes ground-truth `claim0`, `round_evals`, `challenges` for all 7 expressions
- `tests/data/vars4/v4_case32_0_round_tables.npz` — per-round table snapshots; use `provided.expression_round_trace()` to access

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets

- `mle_update_32(zero_eval, one_eval, target_eval, *, q)` at `student.py:122` — use with `table[::2]`, `table[1::2]` for all evaluation points including integers > 1
- `mod_mul_32(a, b, q)` at `student.py:36` — elementwise product for building term polynomials
- `mod_add_32(a, b, q)` at `student.py:20` — for claim0 derivation
- Frozen dispatcher `sumcheck()` at `student.py:174` — already routes to `sumcheck_32`; do not modify

### Established Patterns (from Phase 1)

- All arithmetic uses uint64 promotion internally; inputs/outputs are uint32
- `jnp.sum(arr.astype(jnp.uint64)) % q` for safe reduction of up to 2^20 elements
- No `jax.vmap` — plain array slicing (`[::2]`, `[1::2]`) already vectorized by XLA
- `jax_enable_x64 = True` is set globally; `jnp.uint64` is available

### Integration Points

- `eval_tables: dict[str → jnp.uint32 array]` — ALL 6 variable tables are present; fold all of them even if not in expression
- `challenges: jnp.uint32 array` of length `num_rounds - 1` (final verifier challenge excluded)
- `expression: list[list[str]]` — outer = additive terms, inner = multiplicative factors; variable names index into `eval_tables`

</code_context>

<specifics>
## Specific Ideas

- Use `provided.expression_round_trace(expr_idx, case_id="v4_case32_0")` for debugging: it returns ground-truth per-round table snapshots and expected round_evals for vars4 cases.
- Debug order: start with `--expr-id "a"` (degree 1, simplest), then `"a*b"`, then `"a*b + c"`, then `"a*b*c"`.
- Run `--num-vars 4` before scaling to 16 or 20.

</specifics>

<deferred>
## Deferred Ideas

- Advanced track expressions (`--enable-challenge32`): `a*a*b*b*c`, `a*b*c+d*e`, `a*b*c*g+d*e*g` — algorithm handles them generically but debugging/validation deferred to Phase 3
- `jax.lax.scan` for fully-fused JIT loop — not needed for Phase 2; consider in Phase 3 if benchmark shows loop unrolling overhead for vars20

</deferred>

---

*Phase: 2-SumCheck Prover*
*Context gathered: 2026-04-30*
