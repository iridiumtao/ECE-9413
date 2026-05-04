---
phase: 06-execute-p0-optimizations-revert-exp01-filter-unused-tables-n
plan: 01
subsystem: sumcheck
tags: [jax, sumcheck, modular-arithmetic, optimization, mle]

# Dependency graph
requires:
  - phase: 04-incorporate-peer-optimizations-and-benchmark-experimentally
    provides: Exp01 (t=0/t=1 shortcut, 86e0b5f) and Exp02 (evens/odds hoist, 2c31b3e) student.py states
provides:
  - Exp03 student.py — unused-var filter (OPT-03) + no-mul MLE shortcut for v=2/v=3 (OPT-04)
  - git tags: baseline (632526c), exp01 (86e0b5f), exp03 (7090536)
  - Full correctness suite passing: vars4 (51/51), vars16 (51/51), vars20 (51/51)
affects:
  - 06-02 (ablation benchmarks need exp03 tag and student.py)
  - code.zip regeneration (Plan 03)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "OPT-03: used_vars = set().union(*expression) inside @jax.jit — safe because expression is a static arg; eliminates all 5 unused table materializations and folds per round"
    - "OPT-04: inline single-reduction uint64 expression for constant v values — no chained mod_add/sub/mul helpers; one % q64 per element"

key-files:
  created: []
  modified:
    - student.py

key-decisions:
  - "D-01: Revert student.py from Exp02 to Exp01 commit 86e0b5f before applying Exp03 changes"
  - "D-02: Add used_vars filter inside sumcheck_32 (not in dispatcher) so JIT-compiled graph sees the smaller variable set"
  - "D-04: v=2 no-mul shortcut: ((2*o64 + q64 - z64) % q64).astype(jnp.uint32) — single reduction, no helper chains"
  - "D-05: v=3 no-mul shortcut: ((3*o64 + 2*q64 - 2*z64) % q64).astype(jnp.uint32) — same pattern"
  - "Exp03 reintroduces per-round evens/odds dict (from Exp02) so v=2/v=3 branches can reference evens[var]/odds[var] without repeated inline slicing"

patterns-established:
  - "Per-round evens/odds dict: built once at top of round loop, shared by all v branches and the fold step"
  - "JIT-safe filter: Python set operations on static_argnames are compile-time constants — no traced overhead"

requirements-completed: [OPT-03, OPT-04]

# Metrics
duration: 15min
completed: 2026-05-04
---

# Phase 6 Plan 01: Exp03 — Unused-Var Filter + No-Mul MLE Shortcut

**Exp03 student.py: Exp01 base reverted from Exp02, then augmented with OPT-03 unused-variable table filter and OPT-04 single-reduction uint64 no-mul specialization for v=2/v=3, all 51 vars4/vars16/vars20 tests passing**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-05-04T19:13:00Z
- **Completed:** 2026-05-04T19:28:17Z
- **Tasks:** 3 (Tasks 1+2 not individually committed per plan; single Exp03 commit covers all changes)
- **Files modified:** 1 (student.py)

## Accomplishments

- Tagged `baseline` → `632526c` (pre-optimization) and `exp01` → `86e0b5f` (fastest known before Exp03)
- Reverted student.py to Exp01 state (removed Exp02 hoisted evens/odds dict pre-compute)
- Applied OPT-03 (`used_vars = set().union(*expression)` filter) — eliminates folds of up to 5 unused 2^N tables per round
- Applied OPT-04 (v=2 and v=3 inline uint64 specializations) — single `% q64` per element instead of 3-call chain via `mle_update_32`
- Tagged `exp03` → `7090536` (HEAD after Exp03 commit)
- Full required correctness suite: vars4 51/51, vars16 51/51, vars20 51/51

## Task Commits

Per plan, Tasks 1 and 2 explicitly deferred commit until Task 3 stacked all changes:

1. **Task 1: Tag commits + revert to Exp01** — no separate commit (stacked)
2. **Task 2: Unused-variable table filter** — no separate commit (stacked)
3. **Task 3: No-mul MLE shortcut + Exp03 commit** — `7090536` (feat(06))

**Final commit:** `7090536` — `feat(06): apply Exp03 — unused-var filter + no-mul MLE shortcut`

## Files Created/Modified

- `/Users/oud/Projects/ECE-9413/assignment2/student.py` — Exp03 state: used_vars filter + per-round evens/odds dict + v=2/v=3 no-mul specializations + v>=4 mle_update_32 fallback

## Decisions Made

- **Reintroduced per-round evens/odds dict in Exp03:** Exp01 used literal `tables[term[0]][::2]` inline. To enable the v=2/v=3 branches to reference `evens[var]`/`odds[var]`, the per-round dict was reintroduced at the top of the round loop body (consistent with Exp02 structure but now only over `used_vars`). This is explicitly covered in Task 3 harmonization note.
- **used_vars filter placed inside sumcheck_32, not in dispatcher:** Ensures JIT compile-time set construction — the Python `set()` call executes at trace time because `expression` is in `static_argnames`.
- **v>=4 fallback preserved:** `mle_update_32` call retained in the `else` branch for `--enable-challenge32` advanced polynomials.

## Deviations from Plan

### Auto-harmonized Structure

**1. [Plan Note — Harmonize] Added per-round evens/odds dict alongside Exp01 revert**
- **Found during:** Task 3 (no-mul MLE shortcut implementation)
- **Issue:** Exp01 (reverted) uses `tables[term[0]][::2]` inline — no `evens`/`odds` dicts. The v=2/v=3 branches require `evens[var]`/`odds[var]` keys. Task 3 explicitly notes to "harmonize by introducing the per-round evens/odds dict" in this case.
- **Fix:** Added `evens = {var: tbl[::2] for var, tbl in tables.items()}` and `odds = {var: tbl[1::2] for var, tbl in tables.items()}` at the top of the round loop; updated v=0/v=1 branches to use these dicts; updated fold step to use `evens[var]`/`odds[var]` (not inline slices)
- **Files modified:** student.py
- **Verification:** vars4 51/51, vars16 51/51, vars20 51/51 pass
- **Committed in:** 7090536

---

**Total deviations:** 1 auto-harmonized (plan-anticipated, covered by Task 3 note)
**Impact on plan:** Expected harmonization — Exp03 target in CONTEXT.md showed the per-round dict as part of the design. No scope creep.

## Issues Encountered

None — all three test tiers passed on the first run after each edit.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Exp03 commit hash `7090536` and tag `exp03` are ready for Plan 02 (ablation benchmarks)
- All three git tags (baseline/exp01/exp03) exist for `scripts/bench_ablation.sh`
- `student.py` passes all required tests; ready for `bash scripts/make_submission.sh` (Plan 03)
- Advanced polynomials (`--enable-challenge32`) not yet tested in this plan — deferred to Plan 02 per D-08

## Test Results Summary

| Suite | Cases | Result |
|-------|-------|--------|
| vars4 (N=16) | 51/51 | PASS |
| vars16 (N=65536) | 51/51 | PASS |
| vars20 (N=1048576) | 51/51 | PASS |

## Git Tags

| Tag | Commit | Description |
|-----|--------|-------------|
| `baseline` | `632526c` | Pre-optimization (Phase 3 final) |
| `exp01` | `86e0b5f` | t=0/t=1 shortcut (Phase 4 Plan 01) |
| `exp03` | `7090536` | This plan — unused-var filter + no-mul shortcut |

## Known Stubs

None — all expressions are fully wired; no placeholder data.

## Self-Check: PASSED

- student.py exists and contains all required patterns
- exp03 tag points to 7090536
- baseline tag points to 632526c (starts with 632526c)
- exp01 tag points to 86e0b5f
- All three test suites confirmed passing

---
*Phase: 06-execute-p0-optimizations-revert-exp01-filter-unused-tables-n*
*Completed: 2026-05-04*
