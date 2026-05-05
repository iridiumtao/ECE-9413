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

**Git hash:** `2c31b3ed2afefa129b5ce9d6bb5ad50d1aa786e2`

**What changes:**
- Pre-compute `evens = {var: tbl[::2] ...}` and `odds = {var: tbl[1::2] ...}` once per round
- Use `evens[var]` / `odds[var]` in the v=0 and v=1 shortcut branches
- Use `evens[var]` / `odds[var]` as arguments to `mle_update_32` for v >= 2
- Use `evens[var]` / `odds[var]` in the fold step (replaces inline `tbl[::2]` and `tbl[1::2]`)

**Hypothesis:** Eliminates repeated array slicing — each table is sliced exactly once per round regardless of degree or number of terms. Should show a small but consistent improvement for all expressions, most visible at high degree and large N.

### Results — num-vars 4 (N=16)

| Expression | Compile (ms) | Median (ms) | p90 (ms) | Mpts/s |
|---|---|---|---|---|
| `a` | ~103 | 0.014 | 0.016 | 1.09 |
| `a*b` | ~177 | 0.019 | 0.022 | 0.85 |
| `a*b + c` | ~240 | 0.022 | 0.030 | 0.73 |
| `a*b*c` | ~289 | 0.021 | 0.048 | 0.75 |

### Results — num-vars 16 (N=65,536)

| Expression | Compile (ms) | Median (ms) | p90 (ms) | Mpts/s |
|---|---|---|---|---|
| `a` | ~244 | 0.181 | 0.218 | 362 |
| `a*b` | ~498 | 0.370 | 0.457 | 177 |
| `a*b + c` | ~725 | 0.612 | 0.833 | 107 |
| `a*b*c` | ~753 | 0.736 | 1.050 | 89 |

### Results — num-vars 20 (N=1,048,576)

| Expression | Compile (ms) | Median (ms) | p90 (ms) | Mpts/s |
|---|---|---|---|---|
| `a` | ~283 | 1.091 | 1.445 | 961 |
| `a*b` | ~578 | 3.800 | 4.152 | 275 |
| `a*b + c` | ~800 | 5.538 | 6.598 | 189 |
| `a*b*c` | ~912 | 6.667 | 7.317 | 157 |

### Delta vs Experiment 01 (N=20 median)

| Expression | Exp 01 (ms) | Exp 02 (ms) | Delta |
|---|---|---|---|
| `a` | 1.145 | 1.091 | −0.054 ms (−4.7%) |
| `a*b` | 3.864 | 3.800 | −0.064 ms (−1.7%) |
| `a*b + c` | 5.241 | 5.538 | +0.297 ms (+5.7%) |
| `a*b*c` | 6.508 | 6.667 | +0.159 ms (+2.4%) |

---

## Experiment 03 — unused-variable filter + no-mul MLE shortcut

**Git hash:** `7090536734ee551f1ec2ebbc3bab37bca8a90724`

**What changes:**
- Filter unused variables: `used_vars = set().union(*expression)` at top of `sumcheck_32` body — only materializes tables for variables referenced by the expression (D-02/D-03). The pytest harness loads all 6 `VARIABLE_NAMES` regardless of expression (`tests/case_utils.py:31-61`), so without this filter we slice and fold up to 5 unused 2^N tables every round.
- No-mul MLE shortcut for `v=2` and `v=3`: inline single-reduction uint64 expression (D-04). `v=2` uses `2*o - z`; `v=3` uses `3*o - 2*z`. Avoids the 3 chained `% q` reductions inside `mle_update_32`.
- Reverted Exp02 evens/odds dict pre-compute (which caused a +15–21% GPU regression); base is Exp01.

**Hypothesis:** The filter dominates the speedup at single-variable expressions (`a`) where 5 of 6 variables are wasted work; no-mul shortcut helps degree-2 and degree-3 cases where v=2 and v=3 rows previously paid full `mle_update_32` cost.

### Results — num-vars 4 (N=16)

| Expression | Compile (ms) | Median (ms) | p90 (ms) | Mpts/s |
|---|---|---|---|---|
| `a` | ~83 | 0.014 | 0.016 | 1.16 |
| `a*b` | ~145 | 0.016 | 0.017 | 1.03 |
| `a*b + c` | ~194 | 0.022 | 0.024 | 0.72 |
| `a*b*c` | ~249 | 0.021 | 0.023 | 0.77 |

### Results — num-vars 16 (N=65,536)

| Expression | Compile (ms) | Median (ms) | p90 (ms) | Mpts/s |
|---|---|---|---|---|
| `a` | ~210 | 0.179 | 0.197 | 367 |
| `a*b` | ~340 | 0.341 | 0.368 | 192 |
| `a*b + c` | ~500 | 0.577 | 0.645 | 109 |
| `a*b*c` | ~520 | 0.622 | 0.670 | 105 |

### Results — num-vars 20 (N=1,048,576)

| Expression | Compile (ms) | Median (ms) | p90 (ms) | Mpts/s |
|---|---|---|---|---|
| `a` | ~300 | 1.071 | 1.248 | 978 |
| `a*b` | ~450 | 2.887 | 3.236 | 364 |
| `a*b + c` | ~650 | 4.848 | 5.339 | 216 |
| `a*b*c` | ~660 | 5.050 | 5.406 | 208 |

### Delta vs Experiment 01 (N=20 median)

| Expression | Exp 01 (ms) | Exp 03 (ms) | Delta |
|---|---|---|---|
| `a` | 1.145 | 1.071 | −0.074 ms (−6.5%) |
| `a*b` | 3.864 | 2.887 | −0.977 ms (−25.3%) |
| `a*b + c` | 5.241 | 4.848 | −0.393 ms (−7.5%) |
| `a*b*c` | 6.508 | 5.050 | −1.458 ms (−22.4%) |

### Advanced polynomials (`--enable-challenge32`) — num-vars 20

Per D-08, exercise the v>=4 `mle_update_32` fallback for high-degree polys. Fills the report's §5.1 "repeated constituent polynomials" entry.

**Correctness:** PASS — 35/35 cases passed (`uv run pytest --bits 32 --num-vars 20 --enable-challenge32`)

| Expression | Compile (ms) | Median (ms) | p90 (ms) | Mpts/s |
|---|---|---|---|---|
| `a*a*b*b*c` | ~1570 | 9.735 | 10.307 | 107.71 |
| `a*b*c + d*e` | ~1200 | 7.852 | 8.866 | 133.55 |
| `a*b*c*g + d*e*g` | ~2120 | 13.854 | 14.694 | 75.69 |

---

## Experiment 04 — diff-share running-add + term-by-term sum + no redundant claim0 scan (teammate v9, CPU)

**Git hash:** `8b31f02839e236c95fc06d4835d55db52247b11c` (teammate's repo, `_sumcheck_32_v9` variant)

**What changes:**
- No redundant full-table `claim0` scan: derives `claim0 = g_0(0) + g_0(1)` from the first round's evals (which are computed anyway) — eliminates one full 2^20-element pass at vars=20
- Term-by-term scalar reduction: replaces `_compose_terms_u32` (zeros-init + per-element mod_add per term + full sum) with per-term `jnp.sum` accumulated in uint64 — no intermediate composed-polynomial array materialized
- Running-add for t≥2 (diff-share): precomputes `diffs = odds − evens` once per round; `vals[t] = vals[t−1] + diffs` for t=2,3,…,d — replaces `mle_update_32` (mul + sub + add) with a single `mod_add_32` per element
- No Barrett on CPU — XLA lowers `% q` for a compile-time constant to magic multiply + shift, so `% q` is already near-free on x86/Apple Silicon

**Hypothesis:** Claim0 scan elimination saves ~33% of element-passes at vars=20 for all expressions. Term-by-term reduction avoids a temporary composed-polynomial array, benefiting multi-term expressions (`a*b + c`) most. Running-add for t≥2 should reduce work for degree-2 and degree-3 cases. However, Exp03's no-mul MLE shortcut (`2*o − z`, `3*o − 2*z`) is more efficient than running-add for degree ≤ 3 on CPU: it evaluates the extended point in one inline uint64 op, while running-add requires a separate `diffs` precomputation plus one `mod_add` per variable per t-point. The tradeoff favors term-by-term for multi-term expressions but disfavors running-add for high-degree single-term expressions.

### Results — num-vars 20 (N=1,048,576)

| Expression | Compile (ms) | Median (ms) | p90 (ms) | Mpts/s |
|---|---|---|---|---|
| `a` | ~279 | 1.021 | 1.257 | 1027 |
| `a*b` | ~523 | 3.476 | 3.653 | 302 |
| `a*b + c` | ~702 | 4.630 | 5.333 | 226 |
| `a*b*c` | ~776 | 6.766 | 7.437 | 155 |

Medians are median-of-medians across 5 cases (v20_case32_0 through v20_case32_4). `a*b` case 1 had an elevated compile time (~1422 ms vs ~510 ms for the others) — cold-cache artefact; runtime was unaffected.

### Delta vs Experiment 03 (N=20 median)

| Expression | Exp 03 (ms) | Exp 04 (ms) | Delta |
|---|---|---|---|
| `a` | 1.071 | 1.021 | −0.050 ms (−4.7%) |
| `a*b` | 2.887 | 3.476 | +0.589 ms (+20.4%) |
| `a*b + c` | 4.848 | 4.630 | −0.218 ms (−4.5%) |
| `a*b*c` | 5.050 | 6.766 | +1.716 ms (+34.0%) |

---

## Ablation Chart (Report §5.4.1)

See `report/ablation_chart.png` — grouped bar chart showing vars20 median latency for `a`, `a*b`, `a*b + c`, `a*b*c` across the three optimization states (baseline / Exp01 / Exp03).

Generated by `scripts/plot_ablation.py`. Source numbers from the baseline, Experiment 01, and Experiment 03 sections above. Reproduce with:

```bash
bash scripts/bench_ablation.sh    # captures fresh numbers per tag
uv run python scripts/plot_ablation.py
```
