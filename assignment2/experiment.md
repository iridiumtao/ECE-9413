# Experiment Log — SumCheck Prover Optimization

Each experiment records what changed, the git hash at the time of measurement, and a result summary table.
All benchmarks run with: `uv run python -m tests.benchmark --bench --bits 32 --runs 8 --warmup 3`
Device: CPU (Apple Silicon, no GPU).

---

## Baseline — Current implementation (pre-optimization)

**Git hash:** `632526c70eb2cdf398d75ba30afdfd40ff49933d`

**What this is:**
- `@jax.jit` on `sumcheck_32`
- No `t=0`/`t=1` shortcut — calls `mle_update_32` for every evaluation point including `v=0` and `v=1`
- Recomputes `tbl[::2]` / `tbl[1::2]` separately in fold step (no evens/odds reuse)
- Always naive `% q` reduction
- `claim0` derived from `round_evals[0][0] + round_evals[0][1]`

### Results — num-vars 4 (N=16)

| Expression | Compile (ms) | Median (ms) | p90 (ms) | Mpts/s |
|---|---|---|---|---|
| `a` | ~184 | 0.014 | 0.017 | 1.08 |
| `a*b` | ~247 | 0.018 | 0.021 | 0.90 |
| `a*b + c` | ~349 | 0.021 | 0.025 | 0.75 |
| `a*b*c` | ~430 | 0.039 | 0.085 | 0.41 |

### Results — num-vars 16 (N=65,536)

| Expression | Compile (ms) | Median (ms) | p90 (ms) | Mpts/s |
|---|---|---|---|---|
| `a` | ~464 | 0.221 | 0.256 | 296.66 |
| `a*b` | ~767 | 0.898 | 1.346 | 72.95 |
| `a*b + c` | ~984 | 0.733 | 0.800 | 89.44 |
| `a*b*c` | ~1105 | 0.917 | 0.931 | 71.49 |

### Results — num-vars 20 (N=1,048,576)

| Expression | Compile (ms) | Median (ms) | p90 (ms) | Mpts/s |
|---|---|---|---|---|
| `a` | ~569 | 2.098 | 2.561 | 499.87 |
| `a*b` | ~1181 | 6.571 | 8.029 | 159.58 |
| `a*b + c` | ~3004 | 15.693 | 23.228 | 66.82 |
| `a*b*c` | ~2710 | 9.968 | 10.664 | 105.19 |

---

## Experiment 01 — t=0/t=1 shortcut

**Git hash:** `86e0b5f94092bb3e7a8344ee7de04d0057f0d336`

**What changes:**
- For `v=0`: use `tables[var][::2]` directly — no `mle_update_32` call
- For `v=1`: use `tables[var][1::2]` directly — no `mle_update_32` call
- For `v >= 2`: still calls `mle_update_32` with `tbl[::2]` and `tbl[1::2]` inline
- Fold step unchanged (evens/odds dict reuse is Experiment 02)

**Hypothesis:** For degree-1 expressions (`a`, `a*b`), every `mle_update_32` call in the g_i loop is eliminated. For degree-2 (`a*b + c`) and degree-3 (`a*b*c`), the v=0 and v=1 rows are cheaper but v=2/v=3 still pay full interpolation cost.

### Results — num-vars 4 (N=16)

| Expression | Compile (ms) | Median (ms) | p90 (ms) | Mpts/s |
|---|---|---|---|---|
| `a` | ~85 | 0.014 | 0.016 | 1.12 |
| `a*b` | ~167 | 0.019 | 0.021 | 0.85 |
| `a*b + c` | ~220 | 0.023 | 0.025 | 0.71 |
| `a*b*c` | ~272 | 0.021 | 0.071 | 0.75 |

### Results — num-vars 16 (N=65,536)

| Expression | Compile (ms) | Median (ms) | p90 (ms) | Mpts/s |
|---|---|---|---|---|
| `a` | ~215 | 0.185 | 0.205 | 358 |
| `a*b` | ~440 | 0.387 | 0.465 | 169 |
| `a*b + c` | ~624 | 0.614 | 0.681 | 107 |
| `a*b*c` | ~665 | 0.743 | 1.103 | 88 |

### Results — num-vars 20 (N=1,048,576)

| Expression | Compile (ms) | Median (ms) | p90 (ms) | Mpts/s |
|---|---|---|---|---|
| `a` | ~275 | 1.145 | 1.370 | 916 |
| `a*b` | ~574 | 3.864 | 4.086 | 271 |
| `a*b + c` | ~737 | 5.241 | 5.520 | 200 |
| `a*b*c` | ~877 | 6.508 | 7.555 | 160 |

### Delta vs Baseline (N=20 median)

| Expression | Baseline (ms) | Experiment 01 (ms) | Delta |
|---|---|---|---|
| `a` | 2.098 | 1.145 | −0.953 ms (−45.4%) |
| `a*b` | 6.571 | 3.864 | −2.707 ms (−41.2%) |
| `a*b + c` | 15.693 | 5.241 | −10.452 ms (−66.6%) |
| `a*b*c` | 9.968 | 6.508 | −3.460 ms (−34.7%) |

---

## Experiment 02 — evens/odds pre-computation reuse

**Git hash:** TBD

**What changes:**
- Pre-compute `evens = {var: tbl[::2] ...}` and `odds = {var: tbl[1::2] ...}` once per round
- Use `evens[var]` / `odds[var]` in the v=0 and v=1 shortcut branches
- Use `evens[var]` / `odds[var]` as arguments to `mle_update_32` for v >= 2
- Use `evens[var]` / `odds[var]` in the fold step (replaces inline `tbl[::2]` and `tbl[1::2]`)

**Hypothesis:** Eliminates repeated array slicing — each table is sliced exactly once per round regardless of degree or number of terms. Should show a small but consistent improvement for all expressions, most visible at high degree and large N.

### Results — num-vars 4 (N=16)

| Expression | Compile (ms) | Median (ms) | p90 (ms) | Mpts/s |
|---|---|---|---|---|
| `a` | TBD | TBD | TBD | TBD |
| `a*b` | TBD | TBD | TBD | TBD |
| `a*b + c` | TBD | TBD | TBD | TBD |
| `a*b*c` | TBD | TBD | TBD | TBD |

### Results — num-vars 16 (N=65,536)

| Expression | Compile (ms) | Median (ms) | p90 (ms) | Mpts/s |
|---|---|---|---|---|
| `a` | TBD | TBD | TBD | TBD |
| `a*b` | TBD | TBD | TBD | TBD |
| `a*b + c` | TBD | TBD | TBD | TBD |
| `a*b*c` | TBD | TBD | TBD | TBD |

### Results — num-vars 20 (N=1,048,576)

| Expression | Compile (ms) | Median (ms) | p90 (ms) | Mpts/s |
|---|---|---|---|---|
| `a` | TBD | TBD | TBD | TBD |
| `a*b` | TBD | TBD | TBD | TBD |
| `a*b + c` | TBD | TBD | TBD | TBD |
| `a*b*c` | TBD | TBD | TBD | TBD |

### Delta vs Experiment 01 (N=20 median)

| Expression | Exp 01 (ms) | Exp 02 (ms) | Delta |
|---|---|---|---|
| `a` | TBD | TBD | TBD |
| `a*b` | TBD | TBD | TBD |
| `a*b + c` | TBD | TBD | TBD |
| `a*b*c` | TBD | TBD | TBD |
