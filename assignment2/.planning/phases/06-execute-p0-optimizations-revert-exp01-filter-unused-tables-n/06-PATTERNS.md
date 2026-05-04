# Phase 6: Execute P0 Optimizations - Pattern Map

**Mapped:** 2026-05-04
**Files analyzed:** 4 (student.py body, experiment.md, experiment_gpu.md, scripts/bench_ablation.sh)
**Analogs found:** 3 / 4

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `student.py` lines 159-220 (`sumcheck_32`) | utility / numeric kernel | batch / transform | `student.py` lines 22-43 (`mod_add_32`, `mod_mul_32`) + existing `sumcheck_32` body | exact (self-modification) |
| `experiment.md` (Exp03 section) | documentation | N/A | `experiment.md` Experiment 01 / Experiment 02 sections | exact |
| `experiment_gpu.md` (optional Exp03 section) | documentation | N/A | `experiment_gpu.md` Experiment 01 / Experiment 02 sections | exact |
| `scripts/bench_ablation.sh` | utility / shell script | batch | `scripts/make_submission.sh` | role-match |

---

## Pattern Assignments

### `student.py` lines 159-220 — `sumcheck_32` (numeric kernel, batch/transform)

**Analog:** the existing `sumcheck_32` function at `student.py:159-220` (self-modification).

Three targeted edits are required: (1) table filter, (2) revert evens/odds dict to inline Exp01 style (already done — current HEAD is Exp02 but will be reverted to Exp01 first), (3) no-mul specialization for `v=2` and `v=3`.

---

**Imports / JIT decoration pattern** (`student.py:9-16, 159`):

```python
from __future__ import annotations
import functools
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

@functools.partial(jax.jit, static_argnames=("expression", "num_rounds"))
def sumcheck_32(eval_tables, *, q, expression, challenges, num_rounds):
```

The `@jax.jit` decorator with `static_argnames` must be preserved. The `used_vars` filter is derived from the static `expression` arg, so it is safe inside JIT — the filter is a pure Python set operation that becomes a compile-time constant.

---

**Uint64 promotion pattern** (used throughout `mod_add_32`, `mod_sub_32`, `mod_mul_32`; `student.py:22-43`):

```python
# Always: promote inputs to uint64, compute, reduce with % q64, cast back to uint32.
a64 = jnp.asarray(a, dtype=jnp.uint64)
b64 = jnp.asarray(b, dtype=jnp.uint64)
q64 = jnp.asarray(q, dtype=jnp.uint64)
return jnp.asarray((a64 + b64) % q64, dtype=jnp.uint32)
```

The no-mul shortcut for `v=2` and `v=3` must follow the same shape: one promotion, one arithmetic expression, one `% q64`, one cast to uint32 — no intermediate helper calls.

---

**Tables dict initialization pattern** (`student.py:164-165`, Exp02 current HEAD — will be replaced):

```python
# CURRENT (Exp02, to be replaced):
tables = {var: jnp.asarray(table, dtype=jnp.uint32)
          for var, table in eval_tables.items()}

# TARGET (Exp03 — adds used_vars filter, D-02):
used_vars = set().union(*expression)
tables = {v: jnp.asarray(eval_tables[v], dtype=jnp.uint32) for v in used_vars}
```

`set().union(*expression)` is safe because `expression` is a static JIT arg (tuple-of-tuples). The Python set comprehension executes at trace time, not at runtime.

---

**Evens/odds pattern** (`student.py:171-172`, Exp01 inline style — the revert target):

The revert restores to Exp01 commit `86e0b5f`. After revert, evens/odds are computed inline per-round without a pre-computed dict:

```python
# Exp01 style (target after revert of Exp02):
for i in range(num_rounds):
    evens = {var: tbl[::2]  for var, tbl in tables.items()}
    odds  = {var: tbl[1::2] for var, tbl in tables.items()}
```

The important constraint: evens/odds dicts are rebuilt each round from the current `tables`, which shrinks by half each round. Do NOT hoist them outside the round loop.

---

**t=0 / t=1 shortcut pattern** (`student.py:179-188`):

```python
if v == 0:
    # Zero-eval side: even-indexed entries directly, no mle_update call.
    term_product = evens[term[0]]
    for var in term[1:]:
        term_product = mod_mul_32(term_product, evens[var], q)
elif v == 1:
    # One-eval side: odd-indexed entries directly, no mle_update call.
    term_product = odds[term[0]]
    for var in term[1:]:
        term_product = mod_mul_32(term_product, odds[var], q)
```

This pattern is unchanged by Exp03 — it is part of Exp01 and must survive the revert.

---

**No-mul MLE shortcut pattern for v=2 and v=3** (`student.py:189-197`, to replace full `mle_update_32` call):

For any challenge point `v`, `mle_update_32(zero, one, v, q)` computes `zero + v*(one - zero) mod q`.

Substituting `v=2`: `zero + 2*(one - zero) = 2*one - zero`
Substituting `v=3`: `zero + 3*(one - zero) = 3*one - 2*zero`

Inline uint64 form (single `% q64`, no helper call chains):

```python
elif v == 2:
    q64 = jnp.asarray(q, dtype=jnp.uint64)
    z64 = evens[term[0]].astype(jnp.uint64)
    o64 = odds[term[0]].astype(jnp.uint64)
    # 2*one - zero, shifted by q to avoid underflow
    term_product = ((2*o64 + q64 - z64) % q64).astype(jnp.uint32)
    for var in term[1:]:
        z64_v = evens[var].astype(jnp.uint64)
        o64_v = odds[var].astype(jnp.uint64)
        factor = ((2*o64_v + q64 - z64_v) % q64).astype(jnp.uint32)
        term_product = mod_mul_32(term_product, factor, q)
elif v == 3:
    q64 = jnp.asarray(q, dtype=jnp.uint64)
    z64 = evens[term[0]].astype(jnp.uint64)
    o64 = odds[term[0]].astype(jnp.uint64)
    # 3*one - 2*zero, shifted by 2q to avoid underflow
    term_product = ((3*o64 + 2*q64 - 2*z64) % q64).astype(jnp.uint32)
    for var in term[1:]:
        z64_v = evens[var].astype(jnp.uint64)
        o64_v = odds[var].astype(jnp.uint64)
        factor = ((3*o64_v + 2*q64 - 2*z64_v) % q64).astype(jnp.uint32)
        term_product = mod_mul_32(term_product, factor, q)
else:
    # v >= 4: fall through to full mle_update_32 (unchanged from Exp01).
    v_scalar = jnp.asarray(v, dtype=jnp.uint32)
    term_product = mle_update_32(evens[term[0]], odds[term[0]], v_scalar, q=q)
    for var in term[1:]:
        factor = mle_update_32(evens[var], odds[var], v_scalar, q=q)
        term_product = mod_mul_32(term_product, factor, q)
```

Critical: do NOT write `mod_sub_32(2*o, z, q)` — that would chain two `% q` reductions. The whole point is one single `% q64` per element.

---

**Accumulation and fold pattern** (`student.py:199-213`, unchanged in Exp03):

```python
# Safe uint64 sum: up to 2^20 entries, each < q < 2^32, sum < 2^52 < 2^64.
term_sum = jnp.sum(term_product.astype(jnp.uint64)) % jnp.asarray(q, dtype=jnp.uint64)
g_i_at_v = (g_i_at_v + term_sum) % jnp.asarray(q, dtype=jnp.uint64)

# Fold step reuses pre-computed evens/odds.
if i < len(challenges):
    r = challenges[i]
    tables = {
        var: mle_update_32(evens[var], odds[var], r, q=q)
        for var in tables
    }
```

After the Exp03 table filter, `tables` only contains `used_vars` keys — the fold step naturally iterates only those variables.

---

### `experiment.md` — Exp03 section (documentation)

**Analog:** `experiment.md` Experiment 01 section (lines 49-96) and Experiment 02 section (lines 99-146).

**Section header and metadata pattern** (lines 49-57):

```markdown
## Experiment 01 — t=0/t=1 shortcut

**Git hash:** `86e0b5f94092bb3e7a8344ee7de04d0057f0d336`

**What changes:**
- For `v=0`: use `tables[var][::2]` directly — no `mle_update_32` call
- ...

**Hypothesis:** ...
```

Follow the same pattern for Exp03:

```markdown
## Experiment 03 — unused-variable filter + no-mul MLE shortcut

**Git hash:** `<hash-after-exp03-commit>`

**What changes:**
- Filter unused variables: `used_vars = set().union(*expression)` at top of `sumcheck_32`
- No-mul specialization for `v=2` and `v=3` — inline uint64 expression, single `% q`
- Reverted Exp02 evens/odds pre-computation (GPU regression); base is Exp01

**Hypothesis:** ...
```

**Results table pattern** (lines 62-68 for each num-vars tier):

```markdown
### Results — num-vars 4 (N=16)

| Expression | Compile (ms) | Median (ms) | p90 (ms) | Mpts/s |
|---|---|---|---|---|
| `a` | ~XXX | X.XXX | X.XXX | X.XX |
| `a*b` | ~XXX | X.XXX | X.XXX | X.XX |
| `a*b + c` | ~XXX | X.XXX | X.XXX | X.XX |
| `a*b*c` | ~XXX | X.XXX | X.XXX | X.XX |
```

Three tables required: num-vars 4, 16, 20. Follow with a delta table comparing Exp03 vs Exp01 (not vs Exp02, since Exp03 starts from Exp01):

```markdown
### Delta vs Experiment 01 (N=20 median)

| Expression | Exp 01 (ms) | Exp 03 (ms) | Delta |
|---|---|---|---|
| `a` | 1.145 | X.XXX | −X.XXX ms (−X.X%) |
```

---

### `experiment_gpu.md` — optional Exp03 section (documentation)

**Analog:** `experiment_gpu.md` Experiment 01 section (lines 60-116).

Follows the same structure as the CPU log but adds a "CPU vs GPU Delta" table at the end of the experiment section (pattern from lines 108-115):

```markdown
### CPU vs GPU Delta (N=20 median)

| Expression | CPU (ms) | GPU (ms) | Delta |
|---|---|---|---|
```

Also add an entry to the Observations section (lines 178-190) summarizing Exp03 GPU effect. Include the Exp03 row in the optimization effectiveness table pattern at lines 184-188:

```markdown
| no-mul MLE shortcut + filter (Exp03) | X% | X% |
```

---

### `scripts/bench_ablation.sh` (shell script, batch)

**Analog:** `scripts/make_submission.sh` (lines 1-20).

**Shell script header pattern** (`make_submission.sh:1-8`):

```bash
#!/usr/bin/env bash
set -euo pipefail

# <purpose comment>

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"
```

**Core pattern for `bench_ablation.sh`** — git-tag checkout loop with per-tag benchmark run (per D-06):

```bash
#!/usr/bin/env bash
set -euo pipefail

# Collect ablation benchmark data across baseline / exp01 / exp03 git tags.
# Usage: bash scripts/bench_ablation.sh
# Output: bench_ablation_results.txt in repo root.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"

TAGS=(baseline exp01 exp03)
OUTFILE="${repo_root}/bench_ablation_results.txt"

# Save current state so we can restore after the loop.
CURRENT_COMMIT=$(git rev-parse HEAD)

{
    echo "# Ablation benchmark — $(date)"
    echo "# Generated by scripts/bench_ablation.sh"
    echo ""

    for tag in "${TAGS[@]}"; do
        echo "## Tag: ${tag}"
        git checkout "${tag}" -- student.py
        uv run python -m tests.benchmark --bench --bits 32 --num-vars 20 --runs 8 --warmup 3
        echo ""
    done
} | tee "${OUTFILE}"

# Restore student.py to the HEAD commit state.
git checkout "${CURRENT_COMMIT}" -- student.py

echo ""
echo "Results saved to ${OUTFILE}"
echo "Restore complete — student.py is back to HEAD."
```

Key constraints from D-06: use `git checkout <tag> -- student.py` (not full branch checkout), loop over all three tags, restore after. The `set -euo pipefail` from `make_submission.sh` applies here too.

---

## Shared Patterns

### Uint64 Promotion
**Source:** `student.py:22-43` (`mod_add_32`, `mod_mul_32`)
**Apply to:** all new arithmetic in `sumcheck_32` (no-mul shortcut `v=2`, `v=3`)

```python
q64 = jnp.asarray(q, dtype=jnp.uint64)
z64 = array.astype(jnp.uint64)    # input promotion
result = (expression % q64).astype(jnp.uint32)  # reduction + downcast
```

### JIT-safe Static Derivation
**Source:** `student.py:159, 238` (`@jax.jit static_argnames`, dispatcher tuple conversion)
**Apply to:** `used_vars` filter line in `sumcheck_32`

```python
# expression is static — Python set ops are safe at JIT trace time.
used_vars = set().union(*expression)
```

### LSB-first Table Indexing
**Source:** `student.py:171-172, 210-213`
**Apply to:** all evens/odds references in the no-mul shortcut branches

```python
evens[var] = tbl[::2]   # zero-eval side (bit=0 positions)
odds[var]  = tbl[1::2]  # one-eval side  (bit=1 positions)
```

---

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| — | — | — | All files have direct analogs in the codebase |

---

## Metadata

**Analog search scope:** `student.py` (full file), `experiment.md` (full file), `experiment_gpu.md` (full file), `scripts/make_submission.sh` (full file), `scripts/custom_cases.py` (header), `tests/case_utils.py` (full file), `provided.py` (full file)
**Files scanned:** 7
**Pattern extraction date:** 2026-05-04
