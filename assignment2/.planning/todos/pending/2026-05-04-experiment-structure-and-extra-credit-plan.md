---
created: 2026-05-04T05:03:43Z
title: Set up experiment reproducibility structure and execute extra credit plan
area: tooling
files:
  - student.py:159-220
  - experiment.md
  - experiment_gpu.md
  - scripts/bench_ablation.py (to create)
---

## Problem

From session discussion: we have three measured experiment states (baseline/exp01/exp02) tracked only by git hashes in experiment.md. Working on extra credit optimizations risks losing reproducibility for the report's ablation chart. Also, Report Guidelines §5.1 has a gap on "repeated constituent polynomials" that requires advanced polynomials data.

Best current implementation is **Exp01** (git `86e0b5f`) — t=0/t=1 shortcut. Exp02 (evens/odds reuse) regresses on GPU and should NOT be the production impl.

## Solution

### Step 1 — Git tags (one-time, 3 commands)
```bash
git tag baseline 632526c
git tag exp01    86e0b5f
git tag exp02    2c31b3ed
```

### Step 2 — Env var flags in student.py
Add two boolean env vars to student.py (does not break submission API):
- `SUMCHECK_OPT_SHORTCUT=0/1` — enables t=0/t=1 shortcut (Exp01)
- `SUMCHECK_OPT_EVENS=0/1`    — enables evens/odds pre-computation (Exp02)

Maps directly to ablation chart axes for the report. Extra credits are independent and do NOT need to cross with these flags.

### Step 3 — scripts/bench_ablation.py
Runner that sweeps all 4 flag combinations across vars4/16/20 and outputs CSV.
Used once to generate the report ablation chart — not a production tool.

### Step 4 — Extra credit (linear, not matrix)
Priority order (CP value high → low):
1. **Advanced polynomials** (free) — just run `--enable-challenge32`, fills §5.1 gap on repeated polynomials (`a*a*b*b*c`)
2. **64-bit track** — implement `sumcheck_64`, already stubbed
3. **Barrett/Montgomery** — replace `mod_mul_32` internals, compare vs Exp01; negative result is also report-worthy
4. **Hashing between rounds** — `sumcheck_32_with_hash` variant, quantify hash cost

Each extra credit compares against Exp01 (current best) only — no cross-product needed.

## Report §5.1 Coverage Status

| Question | Status |
|---|---|
| General function vs special | ✅ covered |
| Barrett/Montgomery | ✅ can answer "no, plain widening" |
| Degree → eval points | ✅ covered |
| Performance vs degree | ✅ data in experiment.md |
| Performance vs num-vars | ✅ data in experiment.md |
| Performance vs num-terms | ✅ a*b vs a*b+c data |
| Repeated constituent polynomials | ❌ needs advanced polys run |
| MLE Update impl | ✅ full mod mul |
| MLE Updates after round | ✅ strictly after, compliant |
| vars4 vs vars16/20 choices | ⚠️ partial (Exp02 GPU regression) |
| Bonus tracks | ❌ not started |
