# Phase 2: SumCheck Prover - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-30
**Phase:** 2-SumCheck Prover
**Areas discussed:** Scope, Claim0 derivation, JIT strategy

---

## Scope

| Option | Description | Selected |
|--------|-------------|----------|
| Required track only | Target `a`, `a*b`, `a*b+c`, `a*b*c` for vars4/16/20. Advanced left for Phase 3. | ✓ |
| Include advanced too | Also target `--enable-challenge32` expressions in Phase 2. | |

**User's choice:** Required track only (recommended)

**Follow-up — completion definition:**

| Option | Description | Selected |
|--------|-------------|----------|
| All 3 required commands | vars4, vars16, vars20 all passing. Phase 3 handles benchmarks. | ✓ |
| vars4 only first | De-risk to smallest case, unlock rest in Phase 3. | |
| vars4 + defer JIT test | Phase 2 correctness; JIT test deferred to Phase 3. | |

**User's choice:** All 3 required test commands passing by end of Phase 2.

---

## Claim0 Derivation

| Option | Description | Selected |
|--------|-------------|----------|
| Derive from round 0 (recommended) | `claim0 = mod_add_32(g_0(0), g_0(1), q)` — no redundant table pass. | ✓ |
| Compute from full tables before loop | Extra pre-loop pass; cleaner to reason about but redundant work. | |

**User's choice:** Derive from round 0 evals.

---

## JIT Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Python loop + jnp.stack (recommended) | Simple; JAX traces through it; satisfies JIT test. | ✓ |
| jax.lax.scan | Fully JIT-fused; complex; requires flat state arrays. | |
| No JIT concern in Phase 2 | Defer to Phase 3 even if JIT test is in Phase 2 test file. | |

**User's choice:** Python loop + jnp.stack.

---

## Claude's Discretion

- Round polynomial evaluation pattern: `mle_update_32(table[::2], table[1::2], v, q)` uniformly for all `v ∈ {0, ..., degree}`. Claude determined this from test-data verification.
- Table folding targets: fold ALL variables in `eval_tables`, not just those in the expression.
- Degree determination: `degree = max(len(term) for term in expression)`, computed once before the loop.

## Deferred Ideas

- Advanced `--enable-challenge32` expressions: noted for Phase 3 extra credit
- `jax.lax.scan` optimization: noted for Phase 3 if benchmark warrants it

## Key Pre-Discussion Findings (from codebase + test-data analysis)

Two correctness-critical facts were discovered from ground-truth data BEFORE the user discussion (not user decisions — verified facts):

1. **Interleaved indexing**: ROADMAP "split in half" is wrong. Actual split: even indices = zero-eval, odd indices = one-eval. Verified: `mle_update_32(r0_a[::2], r0_a[1::2], challenge, q) = r1_a` matches ground truth.

2. **Degree = max term length**: `a*b` needs 3 evaluation points (not 2). Verified: `g_0(2) = sum(mle_ext_a_at_2 * mle_ext_b_at_2) = 3426062706` matches stored expected.
