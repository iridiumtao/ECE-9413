---
phase: 06-execute-p0-optimizations-revert-exp01-filter-unused-tables-n
reviewed: 2026-05-04T00:00:00Z
depth: standard
files_reviewed: 4
files_reviewed_list:
  - student.py
  - experiment.md
  - scripts/bench_ablation.sh
  - scripts/plot_ablation.py
findings:
  critical: 0
  warning: 3
  info: 2
  total: 5
status: issues_found
---

# Phase 06: Code Review Report

**Reviewed:** 2026-05-04
**Depth:** standard
**Files Reviewed:** 4
**Status:** issues_found

## Summary

Reviewed `student.py` (OPT-03 unused-var filter, OPT-04 no-mul MLE shortcut), `experiment.md` (Exp03 benchmark data), `scripts/bench_ablation.sh` (git-tag-based ablation runner), and `scripts/plot_ablation.py` (ablation bar chart generator).

No blockers. The OPT-03 filter is JIT-safe — `expression` is a static arg, so `used_vars` is compile-time constant and the dict comprehension does not introduce recompilation hazards. The OPT-04 v=2 and v=3 uint64 arithmetic is overflow-safe and underflow-safe: max input for v=2 is `3*(q-1) < 1.3e10` and for v=3 is `5*(q-1) < 2.2e10`, both far below `uint64` max (`1.84e19`); the minimum non-negative floor is 1 and 2 respectively, so neither formula can wrap. The v≥4 `mle_update_32` fallback is preserved correctly in the `else` branch. The `bench_ablation.sh` trap/restore pattern is correct, and all benchmark delta percentages in `experiment.md` are arithmetically consistent — with one data-entry inconsistency flagged below.

---

## Warnings

### WR-01: experiment.md Mpts/s inconsistency — Exp03 vars16 `a*b + c`

**File:** `experiment.md:175`
**Issue:** The Exp03 vars16 results table shows `a*b + c` with median 0.577 ms and 109 Mpts/s. The implied Mpts/s from the latency is `2^16 / 0.577 ms × 1000 / 1e6 = 113.6`, a 4.2% discrepancy from the recorded 109. Every other row in the file is internally consistent within 0.5%; this row is an outlier. One of the two values is a transcription error from the benchmark run.
**Fix:** Re-run `uv run python -m tests.benchmark --bench --bits 32 --num-vars 16 --runs 8 --warmup 3` and record both the median latency and Mpts/s from the same run output. The correct Mpts/s for 0.577 ms is approximately 113–114; if the true median was ~0.601 ms then Mpts/s of 109 is correct. Update whichever value is wrong, and propagate the correction if this number appears in the report.

---

### WR-02: `scripts/plot_ablation.py` output path is cwd-relative, not script-relative

**File:** `scripts/plot_ablation.py:89`
**Issue:** `out_path = Path("report") / "ablation_chart.png"` resolves relative to the current working directory at the time the script runs. If someone runs `python /abs/path/scripts/plot_ablation.py` from any directory other than the repo root, the `report/` directory is created in the wrong location and the chart is written there. The script's own docstring says "Output: report/ablation_chart.png" (repo-root-relative), but the code does not enforce this.
**Fix:**
```python
# Replace line 89 with:
out_path = Path(__file__).parent.parent / "report" / "ablation_chart.png"
```
This makes the output location invariant to the caller's working directory.

---

### WR-03: `scripts/bench_ablation.sh` does not remove partial output file on failure

**File:** `scripts/bench_ablation.sh:31-35`
**Issue:** The `restore()` trap function restores `student.py` on any exit, but does not clean up `OUTFILE` if the script exits mid-run (e.g., a benchmark crashes or the user presses Ctrl-C). The resulting partial file looks like a complete run — it starts with a valid header and may contain results from some tags but not others — and can silently corrupt a subsequent chart generation if `plot_ablation.py` were ever modified to parse this file.
**Fix:**
```bash
restore() {
    git checkout "${CURRENT_COMMIT}" -- student.py
    # Remove partial output to avoid silently stale data.
    if [[ -f "${OUTFILE}" && "${SCRIPT_COMPLETE:-0}" != "1" ]]; then
        rm -f "${OUTFILE}"
        echo "Removed partial ${OUTFILE}."
    fi
    echo ""
    echo "Restore complete — student.py is back to ${CURRENT_COMMIT}."
}
# ... at end of script body, before exit:
SCRIPT_COMPLETE=1
```

---

## Info

### IN-01: `student.py` — `q64` recomputed on every v-iteration inside the OPT-04 branches

**File:** `student.py:198, 211`
**Issue:** `q64 = jnp.asarray(q, dtype=jnp.uint64)` appears at the top of both the `v == 2` and `v == 3` branches and is re-executed for every term in the inner `for term in expression` loop. Under JIT this is folded to a constant, so there is no runtime cost. However, it clutters the hot-path code and could confuse a reader into thinking it has per-term overhead.
**Fix:** Hoist `q64` to the top of the `sumcheck_32` function body (alongside the `tables` dict construction), where it is computed once and available to all branches:
```python
q64 = jnp.asarray(q, dtype=jnp.uint64)   # used by OPT-04 v=2/v=3 shortcuts
```

---

### IN-02: `student.py` — `TODO` stubs for optional 64/128-bit tracks are intentional but verbose

**File:** `student.py:52-87, 139-146, 257-263`
**Issue:** Ten `# TODO(student): implement when enabling ...` comments mark unimplemented optional functions. These are intentional placeholders per the assignment specification and do not indicate a quality defect. Listed for visibility only.
**Fix:** No action required unless the optional tracks are being implemented.

---

_Reviewed: 2026-05-04_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
