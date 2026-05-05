---
created: 2026-05-04T00:00:00Z
updated: 2026-05-04T00:00:00Z (revised after Codex adversarial review)
title: Trajectory audit — ranked pivots before extra-credit work
area: planning
files:
  - student.py:159-220
  - tests/case_utils.py:31-61
  - .planning/todos/pending/2026-05-04-experiment-structure-and-extra-credit-plan.md
  - experiment.md
  - experiment_gpu.md
  - sumcheck_intro.md:480-484
  - Report_Guidelines.pdf §5.1, §5.4, §9
---

## 1. Bottom-line trajectory assessment

The current plan spends effort on the wrong things. The grading rubric (Report Guidelines §9 + §5.1) rewards three concrete artifacts: a **fast 32-bit submission**, **measured optimization effort with honest discussion**, and **figures tied to commands**. The roadmap re-prioritizes scaffolding (env vars, ablation runner) over the highest-leverage moves: (a) shipping the fastest variant already produced, (b) eliminating wasted work in unused-variable folds, and (c) executing the README-hinted MLE shortcut. All three are short.

## 2. Audit of current `student.py`

### A. Unused-variable folds — the largest single waste (Codex catch)
`tests/case_utils.py:41` and `:54` populate `eval_tables` with **all six variables** (`a, b, c, d, e, g`) for every test case, regardless of which expression is being evaluated. `student.py:164-167` then materializes all six as `uint32` arrays, `:171-172` slices all six every round, and `:210-213` folds all six every round via `mle_update_32`. For `--expr "a"` at vars20 the prover does **6× the table-fold work it needs to**; for `a*b+c` it does ~2×. This is pure wasted bandwidth and arithmetic on every round of every benchmark. Fix is one line: filter `tables` to `set().union(*expression)` immediately after the asarray pass.

### B. Production code is the GPU-regressed Exp02
`student.py:171-172` pre-computes `evens` / `odds` dicts at the top of each round. Per `experiment_gpu.md`, this regressed every base expression on T4 GPU (+17.3% on `a*b+c`, +11.9% on `a*b*c`, +20.9% on `a`, +15.6% on `a*b`). On CPU the picture is mixed — `a` and `a*b` improve very slightly, but `a*b+c` and `a*b*c` regress. Given that grading benchmarks include GPU and that Exp01 is at worst tied on CPU `a`/`a*b`, restoring Exp01 is the safer submitted baseline. Use `git checkout 86e0b5f -- student.py` via a tag, then verify hashes before benchmarking.

### C. The MLE Update no-mul shortcut is unimplemented
`sumcheck_intro.md:482` is an explicit hint: *"Since MLE Update is a linear extrapolation, is there a smarter way to do this?"* For the base track (max degree 3), only `v ∈ {2, 3}` reaches the `mle_update_32` path. Both are degenerate cases of `mle_update(z, o, t) = (o−z)·t + z`:
- `v=2`: `2o − z mod q`
- `v=3`: `3o − 2z mod q`

**Implementation matters.** A naive chained version using `mod_add_32`/`mod_sub_32` still pays 2–3 reductions per call, partially defeating the saving. The right form is a single uint64 expression with one reduction, e.g. conceptually `((2*o_u64 + q_u64 - z_u64) % q_u64).astype(uint32)`. This eliminates the modular multiplications in the v-extrapolation path on every table element of every round — which is the dominant work for `a*b+c` and `a*b*c`. Directly answers the §5.1 *"full modular multiplication, or cheaper path?"* question.

**Limitation:** the advanced track polynomial `a*a*b*b*c` is degree 5, requiring `v=4` and `v=5`. The shortcut as stated covers only the base track unless extended (e.g. `4o − 3z`, `5o − 4z`). Decide whether to extend based on whether advanced polynomials are pursued.

### D. uint64 promotion stages — likely XLA-fused, do not micro-optimize
`mod_add_32`/`mod_sub_32`/`mod_mul_32` each promote to uint64, do math, reduce, cast back. XLA almost certainly fuses sequences inside a JIT region, so this is mostly cosmetic. **Do not optimize this without HLO evidence** — it's the kind of "plausible but un-JAX-like suggestion" the report's AI-reflection section asks to flag.

### E. Why Exp02 regressed is not just "dict allocation"
The hypothesis in `experiment_gpu.md:190` ("pre-allocating dict-of-slices adds memory allocation overhead") is plausible but unverified. The dict is constructed at trace time, not at runtime — there is no per-call allocation cost. The real cause is more likely XLA scheduling: the explicit dict gives XLA a stronger lifetime hint that defeats a fusion it was doing in Exp01 (slicing inside the `mod_mul_32` argument expression let XLA fuse the slice into the mul kernel). A 30-minute HLO diff between Exp01 and Exp02 would resolve this and is itself a great §5.1 / §7 (AI reflection) anecdote.

### F. `lax.scan` over rounds is mostly infeasible — demoted
The earlier draft of this audit recommended `lax.scan` over the round loop to cut compile time. **Codex review correctly flagged this as the weakest claim.** Round-to-round table sizes halve, and `lax.scan` requires a fixed carry shape; padding to the original size would trade compile-time savings for runtime work over inactive entries. A scan over the small `v` axis (degree ≤ 3 or 5) is technically feasible but the payoff is modest. Demote to P3/skip unless P0–P2 are done with time to spare.

## 3. Critique of the existing todo as a roadmap

| Item | Verdict |
|---|---|
| Step 1: git tags | ✅ Trivial, do it. |
| Step 2: env-var flags in `student.py` | ❌ **Cut this.** Adds untested branches to a 50%-of-grade submission. The graded artifact should have one code path. Use `git checkout <tag>` for ablation runs. |
| Step 3: `scripts/bench_ablation.py` | ⚠️ Useful only as a thin shell loop over tags; do not let it grow features. ~20 lines, not a "tool." |
| Step 4.1 Advanced polys | ✅ **Free win**, do it. |
| Step 4.2 64-bit | ⚠️ Real work (1 day+). High report value because §5.1 has a dedicated 64/128-bit question. **Do this only if** P0/P1 are done. |
| Step 4.3 Barrett/Montgomery | ⚠️ Either a small win or a clean negative result. Both are publishable. Medium effort. |
| Step 4.4 Hashing between rounds | 🟡 Lowest scientific interest. Quick to implement, narrow report value (one paragraph). |

**What the todo is missing entirely**:
- Reverting production to Exp01 (and using a tag, not loose `git checkout`).
- **Filtering `tables` to variables actually used by the expression** — eliminates 2–6× wasted fold work depending on expression.
- The MLE Update arithmetic shortcut for `v ∈ {2, 3}`, written as a single-reduction uint64 expression (not chained `mod_add`/`mod_sub` calls).
- An HLO-based explanation for the Exp02 GPU regression (great data exists, no diagnosis).

## 4. Ranked actionable pivots (Codex-revised)

### P0 — Today, before anything else (~1–2 hours total)
1. **Tag Exp01 and revert production.** `git tag exp01 86e0b5f` then `git checkout exp01 -- student.py`. Verify with `uv run pytest --bits 32 --num-vars 4`.
2. **Drop unused-variable work** in `sumcheck_32`. Insert at the top of the function:
   ```python
   used_vars = set().union(*expression)
   tables = {v: tables[v] for v in used_vars}
   ```
   This eliminates the work flagged in §2.A. **Probably the single biggest runtime win** because it scales with table count rather than with arithmetic.
3. **No-mul MLE specialization** for `v=2` and `v=3` in `sumcheck_32:189-197`, written as a single uint64 expression with one `% q` reduction (not chained `mod_add`/`mod_sub` calls). For example, for `v=2`:
   ```python
   z64 = z.astype(jnp.uint64)
   o64 = o.astype(jnp.uint64)
   q64 = jnp.asarray(q, dtype=jnp.uint64)
   factor = ((2*o64 + q64 - z64) % q64).astype(jnp.uint32)
   ```
4. Re-run correctness on vars4/16/20, then benchmark. Tag this state as `exp03`.

### P1 — High grading leverage (~half day each)
5. **Run `--enable-challenge32`** on the advanced polynomials and record numbers. Fills the §5.1 *repeated constituent polynomials* question with `a*a*b*b*c`. Free CP. (Note: degree-5 path will still hit full `mle_update_32` for `v=4,5` unless the shortcut is extended.)
6. **Generate the ablation chart** from git tags (baseline / Exp01 / Exp03). One bar chart, four expressions, three bars each. This is figure §5.4.1 #6.

### P2 — Solid extra credit, real effort
7. **HLO diff Exp01 vs Exp02** (`jax.jit(...).lower(...).compile().as_text()`). Even staying on Exp01, the diagnosis is a strong §5.4 paragraph and §7 (AI reflection) bullet.
8. **64-bit track** (`sumcheck_64` etc., currently stubbed). Directly answers a §5.1 sub-question. The 64×64→128 multiplication needs split-limb math in JAX (no native u128); this is the interesting story.
9. **Barrett reduction** as an isolated experiment on `mod_mul_32`. Even a wash result is publishable as "we tried, plain `% q` after uint64 widening was not the bottleneck."

### P3 — Skip unless P0–P2 are done with time to spare
10. **`lax.scan` over rounds** — likely infeasible due to halving table sizes; padding would trade compile-time saving for runtime regression. Demoted from P1 after Codex review.
11. Hashing between rounds (low scientific interest, one paragraph payoff).
12. 128-bit primitives (very high effort, marginal narrative gain).
13. TPU run (interesting but Modal-only, infrastructure cost).
14. Extending the no-mul shortcut to `v=4,5` for advanced track — only if advanced track is being benchmarked seriously.

### Do not do
- Env-var flags inside `student.py` (Step 2 of the existing todo). Adds branch/test surface to the file the rubric says is the only one to edit.
- Grow `bench_ablation.py` beyond a shell loop over git tags.
- Optimize the uint64-promotion staging without HLO evidence.

## 5. One-line summary

Restore Exp01, filter unused variables out of `tables`, implement the no-mul MLE shortcut as a single-reduction uint64 expression, run advanced polynomials, build the ablation chart — that ordering captures ~80% of the achievable grade improvement and is achievable in a single working session, while the existing todo's env-var scaffolding adds risk to the graded artifact for zero rubric points.
