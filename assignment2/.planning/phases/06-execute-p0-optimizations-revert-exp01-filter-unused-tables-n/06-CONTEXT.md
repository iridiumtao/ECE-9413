# Phase 6: Execute P0 Optimizations - Context

**Gathered:** 2026-05-04
**Status:** Ready for planning

<domain>
## Phase Boundary

Restore `student.py` to the Exp01 state (fastest known variant), apply two new optimizations — unused-variable table filter and no-mul MLE shortcut — benchmark the combined result as Exp03, record it in `experiment.md` (and optionally `experiment_gpu.md`), run `--enable-challenge32` for advanced polynomials, and regenerate `code.zip` from the Exp03 commit. Produce an ablation bar chart covering baseline/Exp01/Exp03 for the report.

NOT in scope: 64-bit or 128-bit tracks, `lax.scan` over rounds (infeasible due to halving table sizes), env-var feature flags in `student.py`, Barrett/Montgomery reduction (defer to P2/P3), hashing between rounds.

</domain>

<decisions>
## Implementation Decisions

### Production baseline reset

- **D-01:** Revert `student.py` to Exp01 commit `86e0b5f` as the starting point — `git tag exp01 86e0b5f` then `git checkout exp01 -- student.py`. Verify with `uv run pytest --bits 32 --num-vars 4` before proceeding. This removes the Exp02 GPU regression (+15–21%) without losing the Exp01 CPU gains (−35% to −67% vs baseline).

### Unused-variable table filter

- **D-02:** Add `used_vars = set().union(*expression)` filter at the TOP of `sumcheck_32`, immediately after the `tables` dict is materialized (line ~164), to drop all variables not referenced in the expression. Place in `sumcheck_32` body (not in the `sumcheck` dispatcher) so the JIT-compiled graph sees the smaller variable set. Example:
  ```python
  used_vars = set().union(*expression)
  tables = {v: jnp.asarray(eval_tables[v], dtype=jnp.uint32) for v in used_vars}
  ```
- **D-03 (override of Phase 4 D-04):** Phase 4 CONTEXT.md D-04 stated "the harness always passes exactly the variables referenced in the expression — zero practical benefit." This is factually incorrect. `tests/case_utils.py:41` iterates `provided.VARIABLE_NAMES = ("a","b","c","d","e","g")` and loads all 6 variables for every test case, regardless of which expression is being tested. Concretely: running `--expr "a"` at vars20 currently materializes, slices, and folds 6 tables of 2^20 = 1M elements each — 5M elements wasted per round × 20 rounds. D-04 is overridden by this evidence.

### No-mul MLE shortcut

- **D-04:** Implement constant-v MLE specialization for `v=2` and `v=3` in the `v >= 2` branch of `sumcheck_32` (currently `student.py:189-197`). Replace the call to `mle_update_32(evens[var], odds[var], v_scalar, q=q)` with inline single-reduction uint64 expressions:
  - `v=2`: `((2*o64 + q64 - z64) % q64).astype(jnp.uint32)` — no modular multiplication, one reduction
  - `v=3`: `((3*o64 + 2*q64 - 2*z64) % q64).astype(jnp.uint32)` — no mul, one reduction
  Do **not** chain `mod_add_32`/`mod_sub_32` helper calls — each call incurs its own `% q` reduction, defeating the saving.
- **D-05:** Cover only `v=2` and `v=3` for now — these cover all base-track expressions (max degree 3 for `a*b*c`). Extension to `v=4,5` is required for `a*a*b*b*c` (degree 5 advanced track) but deferred until base Exp03 is confirmed correct and benchmarked.

### Ablation chart method

- **D-06:** Use git tags for ablation, not env-var flags. Tag each state:
  - `git tag baseline 632526c`
  - `git tag exp01 86e0b5f`
  - `git tag exp03 <hash-of-exp03-commit>`
  Per-checkout workflow: `git checkout <tag> -- student.py` → run benchmarks → record → restore. A short shell loop (~20 lines) in `scripts/bench_ablation.sh` collects all data; no branching logic added to `student.py`.
- **D-07:** The ablation bar chart should show median latency for all 4 base expressions at vars20, with three grouped bars: baseline / Exp01 / Exp03. Target §5.4.1 of the report.

### Advanced polynomials run

- **D-08:** After Exp03 is benchmarked and correctness is confirmed, run `uv run pytest --bits 32 --num-vars 20 --enable-challenge32` to verify advanced polys pass, then `uv run python -m tests.benchmark --bench --bits 32 --num-vars 20 --enable-challenge32 --runs 8 --warmup 3` to collect numbers. Fills the §5.1 "repeated constituent polynomials" gap (`a*a*b*b*c`).

### New production submission

- **D-09:** Exp03 commit becomes the new production baseline. Re-run `bash scripts/make_submission.sh` from the Exp03 commit to produce a new `code.zip`. This overwrites the Phase 3 submission artifact.

### Correctness gate

- **D-10 (Claude's discretion):** Run `uv run pytest --bits 32 --num-vars 4` after each code change (filter, shortcut) before running the full benchmark suite. Must pass all vars4 tests before proceeding to vars16/vars20.

### Folded Todos

- **Folded:** `2026-05-04-experiment-structure-and-extra-credit-plan.md` — "Set up experiment reproducibility structure and execute extra credit plan." The git-tags part (Step 1) is adopted; the env-var flags (Step 2) and `bench_ablation.py` as a full tool (Step 3) are replaced by the simpler shell loop in D-06. Advanced polys (Step 4.1) folded as D-08. 64-bit and Barrett (Steps 4.2, 4.3) remain deferred.
- **Folded:** `2026-05-04-trajectory-audit-and-ranked-pivots.md` — All P0 decisions are codified above (D-01 through D-09). The Codex-revised priority order is the basis for this phase's task sequence.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Files being modified

- `student.py:159-220` — `sumcheck_32` function; specifically lines 164-167 (tables dict), 169 (round loop), 171-172 (evens/odds), 189-197 (v≥2 branch), 210-213 (fold step)
- `experiment.md` — experiment log; add Exp03 section in the same format as Exp01/Exp02

### Harness behavior (critical — overrides Phase 4 D-04)

- `tests/case_utils.py:31-61` — `load_tables()` always loads all 6 variables via `provided.VARIABLE_NAMES` regardless of expression; confirms D-03 above
- `provided.py:7` — `VARIABLE_NAMES = ("a", "b", "c", "d", "e", "g")`

### Prior experiment data

- `experiment.md` — CPU baseline/Exp01/Exp02 tables; Exp03 goes here
- `experiment_gpu.md` — GPU baseline/Exp01/Exp02; optionally add Exp03 GPU run

### Planning artifacts

- `.planning/ROADMAP.md` §Phase 6 — goal, key tasks, success criteria
- `.planning/todos/pending/2026-05-04-trajectory-audit-and-ranked-pivots.md` — Codex-revised ranked pivots (authoritative over the earlier experiment plan)
- `.planning/phases/04-incorporate-peer-optimizations-and-benchmark-experimentally/04-CONTEXT.md` — D-01/D-02 (t=0/t=1 shortcut, evens/odds); D-04 overridden here; D-05 (keep @jax.jit); D-07 (vars4 correctness gate)

### Test and benchmark commands

- `uv run pytest --bits 32 --num-vars 4` — vars4 correctness gate (must pass between changes)
- `uv run pytest --bits 32 --num-vars 4/16/20` — full required correctness suite
- `uv run python -m tests.benchmark --bench --bits 32 --num-vars 4/16/20 --runs 8 --warmup 3` — required benchmark commands
- `uv run pytest --bits 32 --num-vars 20 --enable-challenge32` — advanced polynomials correctness
- `uv run python -m tests.benchmark --bench --bits 32 --num-vars 20 --enable-challenge32 --runs 8 --warmup 3` — advanced polynomials benchmark
- `bash scripts/make_submission.sh` — produces `code.zip` from current state

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets

- `mod_add_32`, `mod_sub_32` at `student.py:22-35` — available but should NOT be chained for the no-mul shortcut (each call re-reduces); use raw uint64 arithmetic instead
- `mle_update_32` at `student.py:124-134` — still used for challenges (fold step) and for any `v >= 4` (advanced track); untouched by D-04/D-05

### Established Patterns

- All arithmetic uses uint64 promotion; inputs/outputs uint32 — the no-mul shortcut must follow this: promote to uint64, compute, reduce with `% q64`, cast back to uint32
- Tables indexed LSB-first: `tbl[::2]` = zero-eval, `tbl[1::2]` = one-eval — unchanged
- `@jax.jit` on `sumcheck_32` with `static_argnames=("expression", "num_rounds")` — preserved; `used_vars = set().union(*expression)` is safe inside JIT (derived from static `expression`)
- `jnp.sum(term_product.astype(jnp.uint64)) % q_u64` accumulation pattern — unchanged

### Integration Points

- `sumcheck()` dispatcher at `student.py:235` passes `eval_tables` dict unmodified; filter happens inside `sumcheck_32`, not at the dispatcher boundary
- Frozen API (`mod_add`, `mod_sub`, `mod_mul`, `mle_update`, `sumcheck`) is untouched — only the body of `sumcheck_32` changes

### Current state vs target

**Exp02 (current HEAD — to be reverted):**
```python
# Line 164-167: loads all 6 tables
tables = {var: jnp.asarray(table, dtype=jnp.uint32) for var, table in eval_tables.items()}
# Line 171-172: evens/odds dict precompute (GPU regression)
evens = {var: tbl[::2] for var, tbl in tables.items()}
odds  = {var: tbl[1::2] for var, tbl in tables.items()}
# Line 192-197: full mle_update_32 for v >= 2 (includes mod mul)
factor = mle_update_32(evens[term[0]], odds[term[0]], v_scalar, q=q)
```

**Exp03 (target):**
```python
# Filter unused variables (new — D-02)
used_vars = set().union(*expression)
tables = {v: jnp.asarray(eval_tables[v], dtype=jnp.uint32) for v in used_vars}
# NO evens/odds dict precompute (reverted to Exp01 — D-01)
evens = {var: tbl[::2]  for var, tbl in tables.items()}  # inside round loop, NO dict
odds  = {var: tbl[1::2] for var, tbl in tables.items()}  # (Exp01 inline style)
# No-mul specialization for v=2 and v=3 (new — D-04)
elif v == 2:
    z64 = evens[term[0]].astype(jnp.uint64)
    o64 = odds[term[0]].astype(jnp.uint64)
    term_product = ((2*o64 + q64 - z64) % q64).astype(jnp.uint32)
    ...
```

</code_context>

<specifics>
## Specific Ideas

- Codex adversarial review: "The biggest miss is that student.py keeps and folds ALL eval tables, even when the expression uses only `a`." This became D-02/D-03 above.
- No-mul MLE shortcut must be written as a single-remainder uint64 expression, not chained `mod_add_32`/`mod_sub_32` calls — Codex correction that changed the D-04 implementation form.
- Phase 4 D-04 ("harness always passes exactly the variables referenced") is overridden by direct inspection of `tests/case_utils.py:41`. Any downstream agent seeing D-04 in Phase 4 CONTEXT.md should treat it as superseded.

</specifics>

<deferred>
## Deferred Ideas

- **`lax.scan` over rounds** — infeasible without padding (halving table sizes break fixed carry shape); demoted from P1 to P3 after Codex review
- **64-bit track** — `sumcheck_64` stubbed in `student.py:223`; answers §5.1 question; defer to P2 if time allows
- **Barrett/Montgomery reduction** — worth a negative-result experiment for the report; defer to P2
- **No-mul shortcut for v=4 and v=5** — needed only for `a*a*b*b*c` (degree 5) advanced track; implement only if D-08 advanced poly run is attempted
- **Hashing between rounds** — low scientific value; skip unless report still needs bonus material after P0/P1
- **128-bit primitives** — high effort, marginal benefit; skip

</deferred>

---

*Phase: 6-Execute P0 Optimizations*
*Context gathered: 2026-05-04*
