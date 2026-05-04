# Phase 6: Execute P0 Optimizations - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-04
**Phase:** 06-execute-p0-optimizations-revert-exp01-filter-unused-tables-n
**Mode:** --auto (all areas auto-selected, recommended option chosen for each)
**Areas discussed:** production_reset, filter_placement, no_mul_scope, ablation_method, advanced_polys

---

## production_reset

| Option | Description | Selected |
|--------|-------------|----------|
| Revert to Exp01 then build Exp03 on top | Cleanest baseline; removes GPU regression; fastest known state | ✓ |
| Build Exp03 on top of current Exp02 HEAD | Avoids a git checkout; but Exp02 regresses on GPU for all expressions | |

**Auto-selected:** "Revert Exp01 → Exp03"
**Notes:** Exp02 regresses on GPU +11–21% across all expressions per `experiment_gpu.md`. Exp01 is at worst tied on CPU for `a`/`a*b`. Reverting is the correct production reset.

---

## filter_placement

| Option | Description | Selected |
|--------|-------------|----------|
| In `sumcheck_32` body (before materialization) | JIT sees smaller static graph; fewer arrays allocated; recommended | ✓ |
| In `sumcheck` dispatcher | Would filter before conversion but dispatcher is a frozen API boundary | |

**Auto-selected:** "In `sumcheck_32` at the top, before materialization"
**Notes:** `used_vars = set().union(*expression)` is safe inside JIT because `expression` is a static arg. Filtering at the dispatcher would touch the frozen API surface unnecessarily.

**Prior decision conflict resolved:** Phase 4 D-04 claimed "the harness always passes exactly the variables referenced in the expression." Direct inspection of `tests/case_utils.py:41` shows this is false — `load_tables()` always loads all 6 variables from `provided.VARIABLE_NAMES`. D-04 is overridden by D-03 in this phase's CONTEXT.md.

---

## no_mul_scope

| Option | Description | Selected |
|--------|-------------|----------|
| v=2 and v=3 only (base track, degree ≤ 3) | Covers all required base expressions; lower risk; recommended | ✓ |
| v=2 through v=5 (advanced track, degree ≤ 5) | Needed for `a*a*b*b*c`; defer until base Exp03 is confirmed | |

**Auto-selected:** "v=2 and v=3 only"
**Notes:** Base track is 50% of the grade. Extension to v=4,5 deferred — only needed if `--enable-challenge32` advanced polys are actively benchmarked. The no-mul form must use a single uint64 reduction (NOT chained `mod_add_32`/`mod_sub_32`), per Codex adversarial review correction.

---

## ablation_method

| Option | Description | Selected |
|--------|-------------|----------|
| git tags + per-checkout shell script | No code changes to student.py; reproducible; recommended | ✓ |
| Env-var flags in student.py | Adds untested branching to the graded submission file; rejected | |

**Auto-selected:** "git tags + shell script loop"
**Notes:** Env-var flags were proposed in the earlier experiment plan todo; Codex review correctly identified this as adding branch/test surface to the file the rubric says is the only one to edit. The shell script approach is ~20 lines and lives in `scripts/bench_ablation.sh`, not in `student.py`.

---

## advanced_polys

| Option | Description | Selected |
|--------|-------------|----------|
| Run --enable-challenge32 after Exp03 benchmarks | Free CP; fills §5.1 "repeated constituent polynomials" gap; recommended | ✓ |
| Skip for now | Leaves a known report gap; no reason to skip if Exp03 is stable | |

**Auto-selected:** "Run after Exp03 benchmarks"
**Notes:** `a*a*b*b*c` (degree 5) is the key advanced poly. The no-mul shortcut for v=4,5 is not yet implemented, so these will run the full `mle_update_32` path for those evaluation points — still correct, just not maximally optimized.

---

## Claude's Discretion

- Exact uint64 arithmetic form for v=3: `(3*o64 + 2*q64 - 2*z64) % q64` — chosen to keep the intermediate positive without overflow (q < 2^32, so 3*o < 3*2^32 < 2^34; 2*q < 2^33; 2*z < 2^33 — sum fits in uint64)
- Whether to run Exp03 GPU benchmarks via Modal after CPU Exp03 is confirmed — leave to planner based on time available

## Deferred Ideas

- `lax.scan` over rounds — infeasible without padding; demoted to P3/skip
- 64-bit track — P2 if time allows
- Barrett/Montgomery — P2 as a negative-result experiment
- No-mul shortcut for v=4,5 — only if advanced polys are pursued
- Hashing between rounds — skip unless report needs bonus material
- 128-bit primitives — skip
