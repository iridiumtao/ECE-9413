---
phase: 06-execute-p0-optimizations-revert-exp01-filter-unused-tables-n
plan: 03
subsystem: sumcheck
tags: [jax, sumcheck, benchmarking, ablation, matplotlib, reporting]

# Dependency graph
requires:
  - phase: 06-execute-p0-optimizations-revert-exp01-filter-unused-tables-n
    plan: 01
    provides: git tags baseline (632526c), exp01 (86e0b5f), exp03 (7090536)
  - phase: 06-execute-p0-optimizations-revert-exp01-filter-unused-tables-n
    plan: 02
    provides: Exp03 vars20 medians in experiment.md (source of chart numbers)
provides:
  - scripts/bench_ablation.sh — reproducible cross-tag ablation benchmark runner
  - scripts/plot_ablation.py — grouped bar chart generator (matplotlib)
  - report/ablation_chart.png — report §5.4.1 figure (50952 bytes, 1275x750 PNG)
  - experiment.md — ablation chart reference section appended
affects:
  - Written report §5.4.1 (ablation analysis figure)

# Tech tracking
tech-stack:
  added:
    - matplotlib==3.10.9 (dev dependency, not bundled in code.zip)
    - pillow==12.2.0 (matplotlib backend dependency)
  patterns:
    - "git-tag ablation runner: git checkout <tag> -- student.py + trap restore EXIT"
    - "matplotlib Agg backend: no DISPLAY required, suitable for headless CI"
    - "Grouped bar chart: 3 bars × 4 expressions, annotated with numeric values"

key-files:
  created:
    - scripts/bench_ablation.sh
    - scripts/plot_ablation.py
    - report/ablation_chart.png
  modified:
    - experiment.md

key-decisions:
  - "D-06: Used git tags (not env-var flags) for ablation — each measurement at exact code state"
  - "D-07: Chart shows 4 base expressions × 3 states (baseline/Exp01/Exp03) at vars20 median"
  - "Numbers hardcoded from experiment.md (authoritative source with warmup/runs verified) rather than re-parsed from bench_ablation_results.txt"

requirements-completed: [REPORT-01]

# Metrics
duration: 15min
completed: 2026-05-04
---

# Phase 6 Plan 03: Ablation Chart Infrastructure (REPORT-01)

**Ablation bar chart (`report/ablation_chart.png`) generated from verified experiment.md numbers — scripts/bench_ablation.sh and scripts/plot_ablation.py committed; report §5.4.1 figure complete; Phase 6 fully shipped.**

## Performance

- **Duration:** ~15 min (dominated by 3 × vars20 benchmark runs via bench_ablation.sh)
- **Started:** 2026-05-04T20:15:00Z
- **Completed:** 2026-05-04T20:30:00Z
- **Tasks:** 2
- **Files created:** 3 (scripts/bench_ablation.sh, scripts/plot_ablation.py, report/ablation_chart.png)
- **Files modified:** 1 (experiment.md)

## Accomplishments

- Created `scripts/bench_ablation.sh` with `set -euo pipefail` + `trap restore EXIT` — loops over `baseline`/`exp01`/`exp03` git tags, checks out only `student.py` from each, runs vars20 benchmark, writes to `bench_ablation_results.txt`, restores HEAD on all exit paths
- Ran bench_ablation.sh end-to-end; all three tag sections captured in `bench_ablation_results.txt`; student.py confirmed clean after run
- Installed matplotlib (3.10.9) as dev dependency; created `scripts/plot_ablation.py` with Agg backend
- Generated `report/ablation_chart.png` (50952 bytes, 1275×750 RGBA PNG) — grouped bar chart with 3 bars × 4 expressions, numeric annotations on each bar
- Appended `## Ablation Chart (Report §5.4.1)` section to `experiment.md` with chart reference and reproduction commands

## Exact Chart Numbers

The following dict values are in `scripts/plot_ablation.py` — verified against `experiment.md`:

### BASELINE (experiment.md §Baseline → Results — num-vars 20)

| Expression | Median (ms) |
|---|---|
| `a` | 2.098 |
| `a*b` | 6.571 |
| `a*b + c` | 15.693 |
| `a*b*c` | 9.968 |

### EXP01 (experiment.md §Experiment 01 → Results — num-vars 20)

| Expression | Median (ms) |
|---|---|
| `a` | 1.145 |
| `a*b` | 3.864 |
| `a*b + c` | 5.241 |
| `a*b*c` | 6.508 |

### EXP03 (experiment.md §Experiment 03 → Results — num-vars 20)

| Expression | Median (ms) |
|---|---|
| `a` | 1.071 |
| `a*b` | 2.887 |
| `a*b + c` | 4.848 |
| `a*b*c` | 5.050 |

## Task Commits

1. **Task 1: scripts/bench_ablation.sh** — `4e75101` (feat(06))
2. **Task 2: scripts/plot_ablation.py + report/ablation_chart.png + experiment.md** — `11498ba` (feat(06))

## Artifacts

| Artifact | Path | Size | Notes |
|---|---|---|---|
| Ablation runner | `scripts/bench_ablation.sh` | 53 lines | Committed |
| Chart generator | `scripts/plot_ablation.py` | 92 lines | Committed |
| Report figure | `report/ablation_chart.png` | 50952 bytes | Committed |
| Raw results | `bench_ablation_results.txt` | ~3KB | Build artifact, NOT committed |

## Deviations from Plan

None — plan executed exactly as written. matplotlib was not pre-installed; installed via `uv add --dev matplotlib` per plan instructions (anticipated in Task 2 action).

## Known Stubs

None — all latency values are real numbers from experiment.md; no placeholders or hardcoded zeroes.

## Threat Flags

None — no new network endpoints, auth paths, or trust boundaries introduced.

## Phase 6 Complete

All three plans shipped:
- **Plan 01:** Exp03 code (unused-var filter + no-mul MLE shortcut) — commit `7090536`, tag `exp03`
- **Plan 02:** Exp03 benchmarks recorded in experiment.md + advanced poly correctness + code.zip regenerated (SHA dd426d31)
- **Plan 03:** Ablation chart infrastructure + report §5.4.1 figure — commits `4e75101` + `11498ba`

## Self-Check: PASSED

- `scripts/bench_ablation.sh` exists and is executable: YES
- `grep -q 'set -euo pipefail' scripts/bench_ablation.sh`: YES
- `grep -q 'trap restore EXIT' scripts/bench_ablation.sh`: YES
- `grep -q 'TAGS=(baseline exp01 exp03)' scripts/bench_ablation.sh`: YES
- `bench_ablation_results.txt` exists with 3 `## Tag:` sections: YES
- `student.py` diff is empty after ablation run: YES
- `scripts/plot_ablation.py` has no `<EXP03_` placeholders: YES
- `report/ablation_chart.png` exists (50952 bytes, PNG image data 1275x750): YES
- `experiment.md` references `ablation_chart.png`: YES
- `experiment.md` has `## Ablation Chart` section: YES
- Task 1 commit `4e75101` exists: YES
- Task 2 commit `11498ba` exists: YES

---
*Phase: 06-execute-p0-optimizations-revert-exp01-filter-unused-tables-n*
*Completed: 2026-05-04*
