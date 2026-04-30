# Phase 1: Primitives + MLE Update - Context

**Gathered:** 2026-04-30
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement four arithmetic kernel stubs in `student.py`: `mod_add_32`, `mod_sub_32`, `mod_mul_32`, and `mle_update_32`. Verify they pass vars4 unit-level smoke tests. No sumcheck prover loop in this phase — that is Phase 2.

</domain>

<decisions>
## Implementation Decisions

### dtype Promotion

- **D-01:** All three modular arithmetic primitives (`mod_add_32`, `mod_sub_32`, `mod_mul_32`) must promote inputs to `jnp.uint64` for intermediate computation. `mod_add_32` can overflow `uint32` because `a + b` can reach ≈ `2 × MAX_PRIME_32 > 2^32`. `mod_sub_32` must handle `a − b` going negative before adding `q`. `mod_mul_32` is already documented in PROJECT.md as needing uint64.
- **D-02:** All three primitives must cast output back to `jnp.uint32`. Uniform output dtype ensures composition is predictable and matches the harness's `jnp.uint32` eval tables.

### mle_update_32 formula

- **D-03:** Use the roadmap-specified form: `result = zero_eval + target_eval * (one_eval − zero_eval)` mod q. One multiply. The subtraction uses `mod_sub_32` to handle underflow safely.

### Claude's Discretion

- **mle_update_32 internal composition:** Claude should choose the cleanest approach. Composing via the three primitives (`mod_sub_32`, `mod_mul_32`, `mod_add_32`) is the preferred direction — it isolates bugs to each function and avoids duplicating mod logic. Inline JAX uint64 arithmetic is acceptable if composition creates awkward dtype flow.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Assignment spec and harness

- `student.py` — frozen dispatcher API; only the four stub functions are edited here
- `provided.py` — expression format, `VARIABLE_NAMES`, `expression_round_trace()` debug helper; read-only
- `sumcheck_utils.py` — expression normalization helpers; read-only
- `tests/test_sumcheck.py` — harness tests; converts tables to `jnp.uint32`, challenges to `jnp.uint32`, calls `sumcheck()` dispatcher
- `tests/conftest.py` — pytest fixtures and case selection flags (`--bits`, `--num-vars`)

### Planning documents

- `.planning/ROADMAP.md` §Phase 1 — exact task list, formulas, and success criteria for each primitive
- `.planning/REQUIREMENTS.md` — PRIM-01, PRIM-02, PRIM-03, MLE-01 definitions
- `.planning/PROJECT.md` §Context — key constraints: `jax_enable_x64 = True`, `MAX_PRIME_32 = 4294967291`, frozen dispatcher, `student.py` is the only editable file

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets

- Frozen dispatchers `mod_add`, `mod_sub`, `mod_mul`, `mle_update` in `student.py:82–136` — already wired to the 32-bit variants; don't rename or change signatures
- `provided.VARIABLE_NAMES` — tuple of allowed variable names; referenced by test harness

### Established Patterns

- `jax.config.update("jax_enable_x64", True)` is set at module top of `student.py` and `test_sumcheck.py` — `jnp.uint64` is available and expected
- Harness converts eval tables to `jnp.uint32` before dispatch (`test_sumcheck.py:21–27`); challenges also `jnp.uint32` — primitives receive uint32 inputs
- Harness normalizes all outputs with `% q` before comparing — off-by-`q` errors won't hide from tests

### Integration Points

- Phase 2 (`sumcheck_32`) will call `mle_update_32` directly per round; the uint32 output contract from D-02 ensures tables stay in uint32 throughout the prover loop

</code_context>

<specifics>
## Specific Ideas

No specific references beyond what is in the roadmap — open to standard JAX uint64 patterns.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 1-Primitives + MLE Update*
*Context gathered: 2026-04-30*
