---
plan: 07-03
status: complete
---

# 07-03 SUMMARY — Exp05 GPU Benchmark (Modal T4)

## What was done

1. **Pushed exp05 branch** to `origin/exp05` — required for Modal to pick up the latest `student.py`.

2. **Fixed JIT static_argnames bug** — `_sumcheck_32_barrett_exp05` called `int(q)` and `_barrett_mu_64(int(q))` inside a JIT-compiled function without marking `q` as static. On GPU (Modal T4) JAX raises `ConcretizationTypeError` because `q` is an abstract tracer. Fixed by adding `"q"` to `static_argnames`. Since `q` is a fixed prime per benchmark run, making it static is correct. All 51 tests continue to pass.

3. **Ran Modal T4 benchmark** (`modal run modal_run.py --bench-only`) — completed successfully for vars4, vars16, and vars20.

4. **Wrote Exp05 GPU section** to `experiment_gpu.md` — includes raw per-case tables, median-of-medians summary tables, delta vs Exp03 GPU, CPU vs GPU delta (Exp05 CPU path), and analysis.

5. **Committed** both the bug fix (`0740ba7`) and the GPU results doc (`4e73248`).

## Actual Exp05 GPU vars20 Median Numbers (Modal T4)

| Expression | Compile (ms) | Median (ms) | p90 (ms) | Mpts/s |
|---|---|---|---|---|
| `a` | ~4,799 | 0.445 | 0.519 | 2,354 |
| `a*b` | ~10,658 | 0.807 | 0.923 | 1,299 |
| `a*b + c` | ~12,138 | 1.420 | 1.632 | 738 |
| `a*b*c` | ~14,586 | 1.210 | 1.289 | 867 |

## Delta vs Exp03 GPU (N=20 median)

| Expression | Exp 03 GPU (ms) | Exp 05 GPU (ms) | Delta |
|---|---|---|---|
| `a` | 0.465 | 0.445 | −0.020 ms (−4.3%) |
| `a*b` | 0.684 | 0.807 | +0.123 ms (+18.0%) |
| `a*b + c` | 1.187 | 1.420 | +0.233 ms (+19.6%) |
| `a*b*c` | 1.136 | 1.210 | +0.074 ms (+6.5%) |

Exp05's GPU path regresses vs Exp03 for `a*b`, `a*b + c`, and `a*b*c`. Only `a` improves slightly. The deferred scalar reduction (single Barrett reduce per t-point) did not eliminate the multi-term GPU kernel overhead because the bottleneck is the per-element Barrett multiply passes + separate `jnp.sum` reductions per term, not the final scalar Barrett reduce.

## CPU vs GPU Speedup for Exp05 (N=20 median)

| Expression | CPU Exp05 (ms) | GPU Exp05 (ms) | Speedup |
|---|---|---|---|
| `a` | 1.077 | 0.445 | 2.4× |
| `a*b` | 2.605 | 0.807 | 3.2× |
| `a*b + c` | 4.295 | 1.420 | 3.0× |
| `a*b*c` | 4.968 | 1.210 | 4.1× |

GPU speedup range: 2.4–4.1× at N=20. Speedup is lower than the baseline's 4–14× range because the Exp05 CPU path is heavily optimized (deferred mod + no-mul shortcuts cut CPU time 30–50% vs Exp03), while the GPU path regresses. The GPU advantage is real but compressed relative to earlier experiments where only the CPU was optimized.

## Key Finding

The Barrett GPU path in Exp05 does not outperform Exp03. The root cause is that adding Python-level term loops over element-wise JAX arrays prevents XLA from fusing what Exp03 composes as a single kernel (one multiply + one add per element → one `jnp.sum`). This is the same XLA fusion breakage observed in Exp04 and Exp02: any Python-level structure that separates element-wise passes forces multiple GPU reduction kernels, each with launch and synchronization overhead that dominates at N=20 on a T4. The single Barrett reduce at the end of each t-point (deferred scalar reduction) saves almost nothing because the expensive operations are the per-element passes, not the final scalar reduction.

**Recommendation for Exp06:** Restore the Exp03 element-wise composition model on GPU (compose the full polynomial per element as `a*b + c` = multiply then add, then single `jnp.sum`), and apply Barrett reduction only inside the fold step (`_mle_update_barrett_u32`). This would combine Exp03's XLA fusion advantage with Barrett for the fold arithmetic.
