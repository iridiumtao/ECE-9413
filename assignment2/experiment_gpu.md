# Experiment Log — SumCheck Prover Optimization (GPU)

Each experiment records what changed, the git hash at the time of measurement, and a result summary table.
All benchmarks run with: `uv run modal run modal_run.py --bench-only`
Device: GPU (Modal T4).

Aggregation: each expression has 5 test cases per num-vars tier; the table shows the **median** across those 5 per-case medians. Compile time shows the median compile latency similarly. Raw per-case data in `~/gpu_results_raw.md`.

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
| `a` | ~774 | 0.187 | 0.228 | 0.09 |
| `a*b` | ~1990 | 0.199 | 0.248 | 0.08 |
| `a*b + c` | ~2193 | 0.216 | 0.297 | 0.07 |
| `a*b*c` | ~2836 | 0.241 | 0.291 | 0.07 |

### Results — num-vars 16 (N=65,536)

| Expression | Compile (ms) | Median (ms) | p90 (ms) | Mpts/s |
|---|---|---|---|---|
| `a` | ~2000 | 0.366 | 0.421 | 179.00 |
| `a*b` | ~3718 | 0.438 | 0.532 | 149.80 |
| `a*b + c` | ~5601 | 0.653 | 0.722 | 100.29 |
| `a*b*c` | ~6607 | 0.669 | 0.773 | 98.03 |

### Results — num-vars 20 (N=1,048,576)

| Expression | Compile (ms) | Median (ms) | p90 (ms) | Mpts/s |
|---|---|---|---|---|
| `a` | ~2635 | 0.464 | 0.524 | 2261.99 |
| `a*b` | ~4891 | 0.822 | 0.899 | 1276.20 |
| `a*b + c` | ~7137 | 1.326 | 1.455 | 791.02 |
| `a*b*c` | ~8539 | 1.344 | 1.442 | 780.41 |

### CPU vs GPU Delta (N=20 median)

| Expression | CPU (ms) | GPU (ms) | Delta |
|---|---|---|---|
| `a` | 2.098 | 0.464 | −1.634 ms (−77.9%) |
| `a*b` | 6.571 | 0.822 | −5.749 ms (−87.5%) |
| `a*b + c` | 15.693 | 1.326 | −14.367 ms (−91.5%) |
| `a*b*c` | 9.968 | 1.344 | −8.624 ms (−86.5%) |

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
| `a` | ~634 | 0.172 | 0.214 | 0.09 |
| `a*b` | ~1935 | 0.186 | 0.233 | 0.09 |
| `a*b + c` | ~2174 | 0.231 | 0.295 | 0.07 |
| `a*b*c` | ~2889 | 0.253 | 0.284 | 0.06 |

### Results — num-vars 16 (N=65,536)

| Expression | Compile (ms) | Median (ms) | p90 (ms) | Mpts/s |
|---|---|---|---|---|
| `a` | ~2058 | 0.335 | 0.406 | 195.34 |
| `a*b` | ~3227 | 0.504 | 0.541 | 129.99 |
| `a*b + c` | ~5391 | 0.749 | 0.833 | 87.46 |
| `a*b*c` | ~5785 | 0.639 | 0.710 | 102.61 |

### Results — num-vars 20 (N=1,048,576)

| Expression | Compile (ms) | Median (ms) | p90 (ms) | Mpts/s |
|---|---|---|---|---|
| `a` | ~2521 | 0.464 | 0.531 | 2259.17 |
| `a*b` | ~4378 | 0.712 | 0.749 | 1472.56 |
| `a*b + c` | ~6947 | 1.176 | 1.315 | 891.52 |
| `a*b*c` | ~7408 | 1.258 | 1.361 | 833.69 |

### Delta vs GPU Baseline (N=20 median)

| Expression | Baseline GPU (ms) | Experiment 01 GPU (ms) | Delta |
|---|---|---|---|
| `a` | 0.464 | 0.464 | 0.000 ms (0.0%) |
| `a*b` | 0.822 | 0.712 | −0.110 ms (−13.4%) |
| `a*b + c` | 1.326 | 1.176 | −0.150 ms (−11.3%) |
| `a*b*c` | 1.344 | 1.258 | −0.086 ms (−6.4%) |

### CPU vs GPU Delta (N=20 median)

| Expression | CPU (ms) | GPU (ms) | Delta |
|---|---|---|---|
| `a` | 1.145 | 0.464 | −0.681 ms (−59.5%) |
| `a*b` | 3.864 | 0.712 | −3.152 ms (−81.6%) |
| `a*b + c` | 5.241 | 1.176 | −4.065 ms (−77.6%) |
| `a*b*c` | 6.508 | 1.258 | −5.250 ms (−80.7%) |

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
| `a` | ~718 | 0.222 | 0.313 | 0.07 |
| `a*b` | ~2507 | 0.297 | 0.369 | 0.05 |
| `a*b + c` | ~2993 | 0.314 | 0.377 | 0.05 |
| `a*b*c` | ~3732 | 0.333 | 0.362 | 0.05 |

### Results — num-vars 16 (N=65,536)

| Expression | Compile (ms) | Median (ms) | p90 (ms) | Mpts/s |
|---|---|---|---|---|
| `a` | ~2516 | 0.391 | 0.473 | 167.43 |
| `a*b` | ~3882 | 0.581 | 0.660 | 112.81 |
| `a*b + c` | ~6676 | 0.834 | 0.932 | 78.56 |
| `a*b*c` | ~7098 | 0.774 | 0.803 | 84.62 |

### Results — num-vars 20 (N=1,048,576)

| Expression | Compile (ms) | Median (ms) | p90 (ms) | Mpts/s |
|---|---|---|---|---|
| `a` | ~3287 | 0.561 | 0.600 | 1869.29 |
| `a*b` | ~5447 | 0.823 | 0.899 | 1273.54 |
| `a*b + c` | ~8304 | 1.380 | 1.509 | 760.07 |
| `a*b*c` | ~8832 | 1.408 | 1.535 | 744.69 |

### Delta vs Experiment 01 GPU (N=20 median)

| Expression | Exp 01 GPU (ms) | Exp 02 GPU (ms) | Delta |
|---|---|---|---|
| `a` | 0.464 | 0.561 | +0.097 ms (+20.9%) |
| `a*b` | 0.712 | 0.823 | +0.111 ms (+15.6%) |
| `a*b + c` | 1.176 | 1.380 | +0.204 ms (+17.3%) |
| `a*b*c` | 1.258 | 1.408 | +0.150 ms (+11.9%) |

### CPU vs GPU Delta (N=20 median)

| Expression | CPU (ms) | GPU (ms) | Delta |
|---|---|---|---|
| `a` | 1.091 | 0.561 | −0.530 ms (−48.6%) |
| `a*b` | 3.800 | 0.823 | −2.977 ms (−78.3%) |
| `a*b + c` | 5.538 | 1.380 | −4.158 ms (−75.1%) |
| `a*b*c` | 6.667 | 1.408 | −5.259 ms (−78.9%) |

---

## Experiment 03 — unused-variable filter + no-mul MLE shortcut

**Git hash:** `7090536734ee551f1ec2ebbc3bab37bca8a90724`

**What changes:**
- Filter unused variables: only materializes tables for variables referenced by the expression — eliminates up to 5 unused 2^N table slices and folds per round
- No-mul MLE shortcut for `v=2` and `v=3`: inline single-reduction uint64 expression; `v=2` uses `2*o - z`, `v=3` uses `3*o - 2*z` — avoids 3 chained `% q` reductions inside `mle_update_32`
- Reverted Exp02 evens/odds dict pre-compute (which caused +15–21% GPU regression); base is Exp01

**Hypothesis:** The unused-variable filter dominates on CPU (5 of 6 variables are wasted work for `a`). On GPU, XLA already schedules unused branches cheaply, so the filter saves less. The no-mul shortcut may help `a*b*c` slightly by reducing arithmetic in v=2/v=3 rows.

### Results — num-vars 4 (N=16)

| Expression | Compile (ms) | Median (ms) | p90 (ms) | Mpts/s |
|---|---|---|---|---|
| `a` | ~633 | 0.195 | 0.275 | 0.08 |
| `a*b` | ~1573 | 0.196 | 0.273 | 0.08 |
| `a*b + c` | ~2041 | 0.232 | 0.342 | 0.07 |
| `a*b*c` | ~2769 | 0.282 | 0.366 | 0.06 |

### Results — num-vars 16 (N=65,536)

| Expression | Compile (ms) | Median (ms) | p90 (ms) | Mpts/s |
|---|---|---|---|---|
| `a` | ~2048 | 0.367 | 0.449 | 178 |
| `a*b` | ~2961 | 0.498 | 0.551 | 132 |
| `a*b + c` | ~5057 | 0.756 | 0.811 | 87 |
| `a*b*c` | ~4983 | 0.625 | 0.691 | 105 |

### Results — num-vars 20 (N=1,048,576)

| Expression | Compile (ms) | Median (ms) | p90 (ms) | Mpts/s |
|---|---|---|---|---|
| `a` | ~2568 | 0.465 | 0.525 | 2254 |
| `a*b` | ~3915 | 0.684 | 0.788 | 1533 |
| `a*b + c` | ~6676 | 1.187 | 1.316 | 883 |
| `a*b*c` | ~6422 | 1.136 | 1.231 | 923 |

### Delta vs GPU Experiment 01 (N=20 median)

| Expression | Exp 01 GPU (ms) | Exp 03 GPU (ms) | Delta |
|---|---|---|---|
| `a` | 0.464 | 0.465 | +0.001 ms (+0.2%) |
| `a*b` | 0.712 | 0.684 | −0.028 ms (−3.9%) |
| `a*b + c` | 1.176 | 1.187 | +0.011 ms (+0.9%) |
| `a*b*c` | 1.258 | 1.136 | −0.122 ms (−9.7%) |

### CPU vs GPU Delta (N=20 median)

| Expression | CPU (ms) | GPU (ms) | Delta |
|---|---|---|---|
| `a` | 1.071 | 0.465 | −0.606 ms (−56.6%) |
| `a*b` | 2.887 | 0.684 | −2.203 ms (−76.3%) |
| `a*b + c` | 4.848 | 1.187 | −3.661 ms (−75.5%) |
| `a*b*c` | 5.050 | 1.136 | −3.914 ms (−77.5%) |

---

## Experiment 04 — diff-share running-add + term-by-term sum + Barrett reduction (teammate v9/barrett_v9)

**Git hash:** `8b31f02839e236c95fc06d4835d55db52247b11c`

**What changes:**
- No redundant full-table `claim0` scan: `claim0` is derived from `g_0(0) + g_0(1)` using the first round's evals (computed anyway) — eliminates one full 2^20-element pass (~33% fewer element-passes at vars=20)
- Term-by-term scalar reduction: replaces `_compose_terms_u32` (zeros-init + per-element mod_add per term + full sum) with per-term reduction directly to a scalar accumulated in uint64 — no composed-polynomial array materialized
- Running-add for t≥2 (diff-share): precomputes `diffs = odds − evens` once per round; for t=2,3,…,d, `vals[t] = vals[t−1] + diffs` — replaces `mle_update` (mul + sub + add) with a single mod_add per element
- Barrett reduction (GPU path, `_sumcheck_32_barrett_v9`): precomputes `μ = ⌊2^64/q⌋` at Python/JIT time and replaces `% q` with `x − q * mulhi(x, μ) + corrective subtract` — avoids CUDA hardware `IDIV` (~20–40 cycles/op) that XLA cannot lower away on PTX

**Hypothesis:** On CPU, eliminating the redundant `claim0` scan and avoiding the intermediate composed-polynomial array are the dominant wins. On GPU, Barrett reduction additionally removes the costly IDIV instruction; combined with diff-share, this should improve `a*b*c` most (more multiplies per element → more IDIV savings) and `a` least (single-term, no intermediate products).

### Results — num-vars 20 (N=1,048,576)

| Expression | Compile (ms) | Median (ms) | p90 (ms) | Mpts/s |
|---|---|---|---|---|
| `a` | ~4,905 | 0.450 | — | 2,330 |
| `a*b` | ~11,934 | 0.729 | — | 1,438 |
| `a*b + c` | ~14,313 | 1.335 | — | 786 |
| `a*b*c` | ~21,005 | 1.107 | — | 947 |

Medians are median-of-medians across 5 cases (our T4 run). Compile times from teammate's reference sweep.

### Delta vs GPU Experiment 03 (N=20 median)

| Expression | Exp 03 GPU (ms) | Exp 04 GPU (ms) | Delta |
|---|---|---|---|
| `a` | 0.465 | 0.450 | −0.015 ms (−3.2%) |
| `a*b` | 0.684 | 0.729 | +0.045 ms (+6.6%) |
| `a*b + c` | 1.187 | 1.335 | +0.148 ms (+12.5%) |
| `a*b*c` | 1.136 | 1.107 | −0.029 ms (−2.6%) |

### CPU vs GPU Delta (N=20 median)

CPU reference: teammate's `_sumcheck_32_v9` (same v9 logic without Barrett).

| Expression | CPU v9 (ms) | GPU barrett_v9 (ms) | Delta |
|---|---|---|---|
| `a` | 1.626 | 0.450 | −1.176 ms (−72.3%) |
| `a*b` | 4.247 | 0.729 | −3.518 ms (−82.8%) |
| `a*b + c` | 6.777 | 1.335 | −5.442 ms (−80.3%) |
| `a*b*c` | 8.516 | 1.107 | −7.409 ms (−87.0%) |

### Analysis — why `a` and `a*b*c` improve but `a*b + c` regresses

The four changes in Exp04 interact differently depending on number of terms and degree:

**`claim0` scan elimination** — helps all expressions equally: skips one full 2^20-element sum by deriving `claim0 = g_0(0) + g_0(1)` from the first round.

**Term-by-term scalar reduction** — the decisive factor for `a*b + c`. The baseline composes the full polynomial elementwise (`a*b + c` per element) and then calls one `jnp.sum`. XLA fuses this into a single kernel: one multiply + one add per element, then reduce. `_eval_sum_mod_barrett` instead loops over terms and calls a separate `jnp.sum` per term — for `a*b + c` (2 terms) that is **two independent GPU reductions per t-point** instead of one. Over 3 t-points × 20 rounds, this doubles the reduction kernel launch count (120 vs 60). GPU reduction kernels are not free to launch and cannot be pipelined when they are data-dependent; the fused single-pass is strictly cheaper. For single-term expressions (`a`, `a*b`, `a*b*c`) the loop visits exactly one term, so term-by-term is structurally identical to compose-then-sum — **no extra kernels, no regression**.

**Running-add for t ≥ 2** — degree determines how many rounds benefit. `a*b*c` (degree 3) has t=2 and t=3 extra points, so 2 `mle_update` multiplications per round are replaced by cheap mod_adds. `a*b + c` (degree 2) only has t=2, so only 1 mle_update is saved — a smaller gain that cannot offset the kernel-launch regression above. `a` (degree 1) has no t≥2 points; running-add does not apply.

**Barrett reduction** — helps proportionally to the number of multiplications per element per t-point. `a*b*c` has 2 multiplications per element (`a*b` then `*c`), so 2 CUDA IDIVs are replaced per element. `a*b + c` and `a*b` have 1 multiplication per element. `a` has none — Barrett contributes nothing for `a`.

**Net effect by expression:**

| Expression | Terms | Extra GPU reductions / round | Running-add savings | Barrett IDIVs / element | Net |
|---|---|---|---|---|---|
| `a` | 1 | 0 | none (deg 1) | 0 | claim0 scan is entire gain → −3.2% |
| `a*b` | 1 | 0 | 1 mle_update (deg 2) | 1 | small gains roughly cancel overhead → neutral (+6.6%) |
| `a*b + c` | **2** | **+3 per round** | 1 mle_update (deg 2) | 1 | **double-reduction overhead dominates → +12.5%** |
| `a*b*c` | 1 | 0 | 2 mle_updates (deg 3) | 2 | multiple gains, no overhead → −2.6% |

The regression on `a*b + c` is not a Barrett or diff-share problem — it is a direct consequence of applying a CPU optimization (avoid intermediate array allocation) to GPU, where the baseline's single fused compose+reduce kernel is already optimal. The same asymmetry explains why Exp02 (evens/odds dict) also regressed: both Exp02 and Exp04's term-by-term mode add Python-level structure (dicts, per-term loops) that breaks XLA's ability to fuse what would otherwise be a single kernel.

---

## Observations

### GPU vs CPU speedup
The T4 GPU provides dramatic speedups at N=20: 4–14× faster across the baseline, shrinking to 2–5× after CPU-only optimizations reduce the CPU baseline. The largest relative speedup is on `a*b+c` in the baseline (−91.5%), reflecting how much more effectively GPU parallelizes the multi-term evaluation.

### Optimization effectiveness on GPU

| Optimization | CPU effect (N=20 `a*b+c`) | GPU effect (N=20 `a*b+c`) |
|---|---|---|
| t=0/t=1 shortcut (Exp01) | −66.6% | −11.3% |
| evens/odds pre-computation (Exp02) | +5.7% (regression) | +17.3% (regression) |
| unused-var filter + no-mul shortcut (Exp03) | −7.5% | +0.9% (neutral) |
| diff-share + term-by-term sum + Barrett (Exp04) | — (teammate impl) | +12.5% (regression vs Exp03; +0.7% vs Baseline) |

The t=0/t=1 shortcut provides a smaller but still positive speedup on GPU. The evens/odds pre-computation **regresses on GPU** for all expressions: pre-allocating dict-of-slices adds memory allocation overhead that outweighs the slice-reuse benefit, especially since XLA already schedules element-wise stride operations efficiently without intermediate buffers. The unused-variable filter and no-mul shortcut are **neutral on GPU** for `a` and `a*b + c` (within noise), and provide a small improvement for `a*b*c` (−9.7%). The CPU-dominant savings from filtering unused variables do not transfer to GPU: XLA's kernel already skips inactive branches cheaply, so the filter adds overhead (dict lookup) without eliminating real GPU work.
