# Roadmap: ECE-9413 Assignment 2 — SumCheck Prover

**3 phases** | **15 v1 requirements + 2 extended** | All requirements covered ✓

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

**Plans:** 1 plan

Plans:
- [x] 01-01-PLAN.md — Implement 4 primitives (mod_add_32, mod_sub_32, mod_mul_32, mle_update_32) and verify with vars4 edge-case tests (completed 2026-04-30, commits 2f4629b 2590640)

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

**Plans:** 1 plan

Plans:
- [x] 02-01-PLAN.md — Implement sumcheck_32 (full prover loop) and verify with staged tests (vars4 → vars16 → vars20) (completed 2026-05-02, commits 4950a64 a6c3e50)

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

**Plans:** 2 plans

Plans:
- [x] 03-01-PLAN.md — Add JIT decorator to sumcheck_32, verify tests pass, run all three benchmarks (BENCH-01/02/03) (completed 2026-05-03, commits 463f10f c8d4a05)
- [x] 03-02-PLAN.md — Run make_submission.sh, produce code.zip (SUB-01) (completed 2026-05-03, commits 6b59ac0 234c998)

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

### Phase 4: Incorporate peer optimizations and benchmark experimentally

**Goal:** Port two peer optimizations (t=0/t=1 shortcut and evens/odds pre-computation) into `sumcheck_32`, benchmark each change against the Phase 3 baseline, and record results in `experiment.md` with git hash tracking.
**Requirements:** OPT-01, OPT-02
**Depends on:** Phase 3
**Plans:** 2 plans

Plans:
**Wave 1**
- [x] 04-01-PLAN.md — Apply t=0/t=1 shortcut, run all three benchmark tiers, fill Experiment 01 table (OPT-01) (completed 2026-05-03, commits 86e0b5f 5e77307)

**Wave 2** *(blocked on Wave 1 completion)*
- [x] 04-02-PLAN.md — Pre-compute evens/odds once per round, run all three benchmark tiers, fill Experiment 02 table (OPT-02) (completed 2026-05-03, commits 2c31b3e f2c4ead)

**Cross-cutting constraints:**
- All 51 vars4 tests must pass after each code change before running benchmarks (D-07)

---

### Phase 5: GPU Benchmarking with Modal

**Goal:** Port the benchmark harness to Modal so every experiment in `experiment.md` can be re-run on real GPU hardware; record GPU results in `experiment_gpu.md` with matching git hash tracking for reproducibility.
**Requirements:** GPU-01, GPU-02, PERF-02
**Depends on:** Phase 4
**Plans:** 2 plans

Plans:
**Wave 1**
- [x] 05-01-PLAN.md — Create `modal_run.py` (app=ece9413-sumcheck, jax[cuda12], T4/A100/H100 GPU tiers, all three num-vars tiers per run) (GPU-01, PERF-02) (completed 2026-05-03, commit ccf7dc5)

**Wave 2** *(blocked on Wave 1 completion)*
- [ ] 05-02-PLAN.md — Checkout each git hash, run modal_run.py, record GPU output, write experiment_gpu.md with CPU vs GPU delta tables (GPU-02, PERF-02)

### Phase 6: Execute P0 optimizations — revert Exp01, filter unused tables, no-mul MLE shortcut, ablation benchmarks

**Goal:** Restore the fastest known variant (Exp01), eliminate all wasted unused-variable fold work, implement the README-hinted no-mul MLE shortcut as a single-reduction uint64 expression for v=2 and v=3, run correctness and benchmarks to produce Exp03 data, and generate the ablation chart needed for the report.

**Requirements:** OPT-03 (unused-var filter), OPT-04 (no-mul MLE), REPORT-01 (ablation chart)
**Depends on:** Phase 4 (has all benchmark data); Phase 5 (GPU data)

**Key tasks:**
1. Tag Exp01 (`git tag exp01 86e0b5f`) and revert `student.py` to it (`git checkout exp01 -- student.py`). Verify with `uv run pytest --bits 32 --num-vars 4`.
2. Filter `tables` to variables actually used by `expression` at `student.py:164-167`:
   ```python
   used_vars = set().union(*expression)
   tables = {v: jnp.asarray(eval_tables[v], dtype=jnp.uint32) for v in used_vars}
   ```
3. Implement no-mul MLE specialization for `v=2` and `v=3` in the `v >= 2` branch (`student.py:189-197`) using single-reduction uint64 arithmetic — e.g. for v=2: `((2*o64 + q64 - z64) % q64).astype(uint32)`. Do **not** use chained `mod_add_32`/`mod_sub_32` calls (multiple reductions defeat the saving).
4. Run full correctness suite: `uv run pytest --bits 32 --num-vars 4/16/20`.
5. Run benchmarks on vars4/16/20 and record as Exp03 in `experiment.md` and optionally `experiment_gpu.md`.
6. Run `--enable-challenge32` on advanced polynomials and record numbers (fills §5.1 "repeated constituent polynomials" question).
7. Generate ablation bar chart (baseline / Exp01 / Exp03) and CPU vs GPU comparison table for the report.
8. Re-run `bash scripts/make_submission.sh` to produce a final `code.zip` from the Exp03 state.

**Success criteria:**
1. `student.py` at the Exp03 state folds only the variables referenced by `expression` — confirmed by inspecting `tables` keys in a vars4 debug trace.
2. No-mul specialization matches `mle_update_32` outputs for v=2 and v=3 on all vars4 test cases.
3. All 51 vars4 + vars16 + vars20 correctness tests pass after both changes.
4. Exp03 median latency for `a` at vars20 is below Exp01 (1.145 ms CPU), confirming the unused-var filter wins.
5. Ablation chart exists with three bars per expression.
6. `code.zip` produced from the Exp03 commit.

**Plans:** 3 plans

Plans:
**Wave 1**
- [x] 06-01-PLAN.md — Revert student.py to Exp01, apply unused-var filter (OPT-03) + no-mul MLE shortcut for v=2/v=3 (OPT-04), commit Exp03 with vars4/16/20 tests green (completed 2026-05-04, commits 7090536 41864f2)

**Wave 2** *(blocked on Wave 1 completion)*
- [x] 06-02-PLAN.md — Run Exp03 CPU benchmarks, record in experiment.md, run --enable-challenge32 advanced polys (D-08), regenerate code.zip (D-09) (completed 2026-05-04, commits 7064471 6b9905a 9977831)

**Wave 3** *(blocked on Wave 2 completion)*
- [x] 06-03-PLAN.md — Create scripts/bench_ablation.sh + scripts/plot_ablation.py, emit report/ablation_chart.png (REPORT-01) (completed 2026-05-04, commits 4e75101 11498ba 9ba30b3)

### Phase 7: Exp05: incorporate teammate v9 optimizations — learn from Exp01–Exp03 baseline, merge Exp04 gains, benchmark on local CPU and Modal T4 GPU, update charts if improved

**Goal:** Create Exp05 as a deliberate synthesis of our Exp03 and teammate's v9 — keeping what Exp03 does well and incorporating v9's gains that don't regress. Run correctness tests, benchmark on local CPU and Modal T4 GPU, record results, and update the ablation chart if Exp05 beats Exp03.
**Requirements:** EXP05-01, EXP05-02, EXP05-03, EXP05-04, EXP05-05
**Depends on:** Phase 6
**Plans:** 4 plans

Plans:
**Wave 1**
- [ ] 07-01-PLAN.md — Create exp05 branch, implement _sumcheck_32_exp05 (deferred mod + no-mul t=2/t=3 + running-add t>=4) + _sumcheck_32_barrett_exp05 + adaptive dispatch, run correctness tests (EXP05-01, EXP05-02)

**Wave 2** *(blocked on Wave 1 completion)*
- [ ] 07-02-PLAN.md — Run CPU benchmark suite (vars4/16/20), record Exp05 results in experiment.md (EXP05-03)
- [ ] 07-03-PLAN.md — Run Modal T4 GPU benchmark, record Exp05 GPU results in experiment_gpu.md (EXP05-04)

**Wave 3** *(blocked on Wave 2 + Wave 3 completion)*
- [ ] 07-04-PLAN.md — Update scripts/plot_ablation.py with Exp05 data, regenerate ablation charts (EXP05-05)

---
*Roadmap created: 2026-04-30*
