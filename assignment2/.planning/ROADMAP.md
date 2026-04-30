# Roadmap: ECE-9413 Assignment 2 — SumCheck Prover

**3 phases** | **15 v1 requirements** | All requirements covered ✓

| # | Phase | Goal | Requirements | Success Criteria |
|---|-------|------|--------------|-----------------|
| 1 | Primitives + MLE Update | Correct modular arithmetic and MLE folding kernel | PRIM-01/02/03, MLE-01 | 4 |
| 2 | SumCheck Prover | Full prover loop passes all required correctness tests | SC-01/02/03/04, CORR-01/02/03 | 4 |
| 3 | Optimization + Submission | JIT-compiled, benchmarked, packaged for submission | BENCH-01/02/03, SUB-01 | 3 |

---

## Phase 1: Primitives + MLE Update

**Goal:** Implement correct 32-bit modular arithmetic primitives and the MLE update (linear interpolation) kernel; verify they pass unit-level tests on vars4 before proceeding.

**Requirements:** PRIM-01, PRIM-02, PRIM-03, MLE-01

**Key tasks:**

1. Implement `mod_add_32(a, b, q)` — elementwise `(a + b) % q`; use `jnp.uint64` to avoid overflow before reducing.
2. Implement `mod_sub_32(a, b, q)` — `(a - b + q) % q` to avoid underflow.
3. Implement `mod_mul_32(a, b, q)` — cast inputs to `uint64`, multiply, reduce, cast back to `uint32`.
4. Implement `mle_update_32(zero_eval, one_eval, target_eval, *, q)` — linear interpolation: `zero_eval + target_eval * (one_eval - zero_eval)` mod q, applied elementwise to first half of each table.
5. Run vars4 primitive smoke tests to confirm correctness.

**Success criteria:**

1. `mod_add_32(a, b, q)` matches `(int(a) + int(b)) % int(q)` for edge inputs (0, q-1, overflow values).
2. `mod_sub_32(a, b, q)` never produces negative values; handles `a < b` correctly.
3. `mod_mul_32(a, b, q)` correctly computes `(q-1)*(q-1) % q` without overflow.
4. `mle_update_32` folds a table of length `2^k` to `2^(k-1)` using the given challenge `r`.

**UI hint:** no

---

## Phase 2: SumCheck Prover

**Goal:** Implement `sumcheck_32` — the full prover loop — and pass all required 32-bit correctness tests for base polynomials (`a`, `a*b`, `a*b+c`, `a*b*c`) on vars4, vars16, and vars20.

**Requirements:** SC-01, SC-02, SC-03, SC-04, CORR-01, CORR-02, CORR-03

**Key tasks:**

1. Evaluate the full expression (product and sum of eval tables) to compute `claim0` (initial sum over Boolean hypercube).
2. For each round `i` in `range(num_rounds)`:
   a. Sum each term of the expression: split each table in half (index 0 to half = zero-eval, half to end = one-eval), compute `g_i(0)` and `g_i(1)`.
   b. Collect `[g_i(0), g_i(1)]` as the round evaluation.
   c. Apply `mle_update_32` to fold all tables using `challenges[i]` (if `i < num_rounds - 1`).
3. Return `(claim0, round_evals)` where `round_evals` is a JAX array of shape `(num_rounds, 2)`.
4. Debug using `provided.expression_round_trace()` for vars4 if outputs do not match.
5. Run staged tests: vars4 first, then vars16, then vars20.

**Success criteria:**

1. `sumcheck_32` returns a tuple where both elements are JAX arrays.
2. `claim0` matches the harness's expected `initial_claim` for all 5 vars4 cases on all base expressions.
3. `round_evals` matches `expected_round_evals` for all vars4 cases.
4. All required test commands pass:
   ```
   uv run pytest --bits 32 --num-vars 4
   uv run pytest --bits 32 --num-vars 16
   uv run pytest --bits 32 --num-vars 20
   ```

**UI hint:** no

---

## Phase 3: JAX Optimization + Benchmarks + Submission

**Goal:** Apply `jax.jit` (and `jax.vmap` where applicable) to critical kernels, run the required benchmarks, and produce the final `code.zip` submission.

**Requirements:** BENCH-01, BENCH-02, BENCH-03, SUB-01

**Key tasks:**

1. Wrap `sumcheck_32` (or its inner expression evaluator) with `jax.jit`; verify correctness is preserved after JIT.
2. Apply `jax.vmap` to elementwise operations inside `mle_update_32` or the expression evaluator if not already vectorized.
3. Run required benchmarks:
   ```
   uv run python -m tests.benchmark --bench --bits 32 --num-vars 4  --runs 8 --warmup 3
   uv run python -m tests.benchmark --bench --bits 32 --num-vars 16 --runs 8 --warmup 3
   uv run python -m tests.benchmark --bench --bits 32 --num-vars 20 --runs 8 --warmup 3
   ```
4. Record benchmark numbers for the report (`report.pdf`).
5. Run `bash scripts/make_submission.sh` to produce `code.zip`.
6. (Optional) Attempt advanced 32-bit polynomials (`--enable-challenge32`) for extra credit.

**Success criteria:**

1. `jax.jit`-compiled path produces identical outputs to the non-JIT path.
2. Benchmark completes without error for all three problem sizes.
3. `make_submission.sh` exits 0 and produces `code.zip`.

**UI hint:** no

---

## Requirement Coverage

| Requirement | Phase |
|-------------|-------|
| PRIM-01 | 1 |
| PRIM-02 | 1 |
| PRIM-03 | 1 |
| MLE-01 | 1 |
| SC-01 | 2 |
| SC-02 | 2 |
| SC-03 | 2 |
| SC-04 | 2 |
| CORR-01 | 2 |
| CORR-02 | 2 |
| CORR-03 | 2 |
| BENCH-01 | 3 |
| BENCH-02 | 3 |
| BENCH-03 | 3 |
| SUB-01 | 3 |

All 15 v1 requirements covered ✓

---
*Roadmap created: 2026-04-30*
