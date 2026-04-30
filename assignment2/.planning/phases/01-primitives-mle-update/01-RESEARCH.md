# Phase 1: Primitives + MLE Update — Research

**Researched:** 2026-04-30
**Domain:** JAX uint32/uint64 modular arithmetic + multilinear extension (MLE) folding kernel
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** All three modular arithmetic primitives (`mod_add_32`, `mod_sub_32`, `mod_mul_32`) must
  promote inputs to `jnp.uint64` for intermediate computation. `mod_add_32` can overflow `uint32`
  because `a + b` can reach ≈ `2 × MAX_PRIME_32 > 2^32`. `mod_sub_32` must handle `a − b` going
  negative before adding `q`. `mod_mul_32` is already documented in PROJECT.md as needing uint64.
- **D-02:** All three primitives must cast output back to `jnp.uint32`. Uniform output dtype ensures
  composition is predictable and matches the harness's `jnp.uint32` eval tables.
- **D-03:** Use the roadmap-specified form: `result = zero_eval + target_eval * (one_eval − zero_eval)` mod q.
  One multiply. The subtraction uses `mod_sub_32` to handle underflow safely.

### Claude's Discretion

- **mle_update_32 internal composition:** Claude should choose the cleanest approach. Composing via
  the three primitives (`mod_sub_32`, `mod_mul_32`, `mod_add_32`) is the preferred direction — it
  isolates bugs to each function and avoids duplicating mod logic. Inline JAX uint64 arithmetic is
  acceptable if composition creates awkward dtype flow.

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID      | Description                                                                                           | Research Support                                                                        |
|---------|-------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|
| PRIM-01 | `mod_add_32(a, b, q)` returns `(a + b) % q` correctly for any uint32 inputs and 32-bit prime `q`     | uint64 promotion pattern verified; overflow boundary confirmed empirically               |
| PRIM-02 | `mod_sub_32(a, b, q)` returns `(a - b) % q` correctly (no negative underflow)                        | `(a + q - b) % q` in uint64 pattern verified; no int64 sign issues                     |
| PRIM-03 | `mod_mul_32(a, b, q)` returns `(a * b) % q` correctly (intermediate must not overflow uint32)         | `(q-1)^2 ≈ 1.84e19 < 2^64` confirmed; uint64 product safe; result cast to uint32       |
| MLE-01  | `mle_update_32(zero_eval, one_eval, target_eval, *, q)` computes linear interpolation to fold one variable | Formula `z + t*(o-z) mod q` verified against textbook example; composed primitives work |
</phase_requirements>

---

## Summary

Phase 1 implements four arithmetic kernels that form the foundation of the 32-bit SumCheck prover.
The core challenge is that 32-bit prime fields require intermediate values wider than 32 bits: two
`uint32` values can add to just under `2^33`, and their product can reach just under `2^64`. JAX
with `jax_enable_x64 = True` provides `jnp.uint64`, which is exactly the right intermediate type
for all three arithmetic primitives.

The MLE update (`mle_update_32`) computes a linear interpolation of a Boolean variable's evaluation
table. The canonical formula from the assignment spec (`sumcheck_intro.md`) is
`result = (one_eval − zero_eval) * target_eval + zero_eval` mod `q`, equivalently
`zero_eval + target_eval * (one_eval − zero_eval)` mod `q`. This operation is fully elementwise on
arrays, so plain JAX array operations suffice — no `jax.vmap` is needed. The three arithmetic
primitives compose cleanly to implement it.

The harness always passes `q` as a `jnp.uint32` scalar and `eval_tables` as `jnp.uint32` arrays
for the 32-bit track. Primitive implementations must handle both scalar and array inputs uniformly
because the edge-case unit tests call them on scalar `jnp.uint32` values, while `sumcheck_32`
(Phase 2) will call them on arrays. The `jnp.asarray(..., dtype=jnp.uint64)` cast handles both
cases identically.

**Primary recommendation:** Implement each primitive as a one-liner cast → operate → cast back, then
compose them inside `mle_update_32`. Do not use `jax.vmap` for Phase 1 — it adds overhead for a
task that is already vectorized by JAX's native array broadcasting.

---

## Architectural Responsibility Map

| Capability         | Primary Tier  | Secondary Tier | Rationale                                                             |
|--------------------|---------------|----------------|-----------------------------------------------------------------------|
| mod_add_32         | student.py    | —              | Pure arithmetic kernel; no I/O or service boundary                   |
| mod_sub_32         | student.py    | —              | Pure arithmetic kernel; no I/O or service boundary                   |
| mod_mul_32         | student.py    | —              | Pure arithmetic kernel; no I/O or service boundary                   |
| mle_update_32      | student.py    | —              | Composes the three primitives; no external state                     |
| Frozen dispatchers | student.py    | —              | Already wired; never edited                                           |
| Test harness       | tests/        | —              | Read-only; converts tables/challenges to jnp.uint32 before dispatch  |

---

## Standard Stack

### Core

| Library        | Version | Purpose                                  | Why Standard                                       |
|----------------|---------|------------------------------------------|----------------------------------------------------|
| jax            | 0.10.0  | Array computation, JIT, vmap             | Installed via pyproject.toml; `jax_enable_x64`     |
| jax.numpy      | 0.10.0  | NumPy-compatible array ops (`jnp`)       | Primary API for all tensor math                    |
| numpy          | —       | Test data loading (`.npz` files)         | Read-only; test harness uses it for file I/O       |
| pytest         | —       | Unit test runner                         | Defined in pyproject.toml                          |

[VERIFIED: pyproject.toml, `uv run python -c "import jax; print(jax.__version__)"`]

**No additional installation required.** All dependencies are already in the project's virtual environment.

**Test command for this phase:**
```bash
uv run pytest --bits 32 --num-vars 4 -v
```

---

## Architecture Patterns

### System Architecture Diagram

```
Harness (tests/test_sumcheck.py)
    |
    | uint32 arrays + uint32 q scalar
    v
student.mod_add_32 / mod_sub_32 / mod_mul_32
    |
    | cast inputs --> uint64
    | operate in uint64 field
    | cast output --> uint32
    v
student.mle_update_32
    |
    | calls mod_sub_32(one_eval, zero_eval, q)  --> diff (uint32)
    | calls mod_mul_32(target_eval, diff, q)     --> scaled (uint32)
    | calls mod_add_32(zero_eval, scaled, q)     --> result (uint32)
    v
uint32 array of length N/2 (one Boolean variable folded out)
    |
    v
sumcheck_32 in Phase 2 (consumes mle_update_32 per round)
```

### Recommended Structure in student.py

```
student.py
├── mod_add_32(a, b, q)       # cast→uint64, add, reduce, cast→uint32
├── mod_sub_32(a, b, q)       # cast→uint64, add q, subtract, reduce, cast→uint32
├── mod_mul_32(a, b, q)       # cast→uint64, multiply, reduce, cast→uint32
├── mle_update_32(...)         # compose the three primitives above
└── (frozen dispatchers — no edits)
```

### Pattern 1: uint64 Promotion for Modular Arithmetic

**What:** Cast uint32 inputs to uint64 before any arithmetic, reduce with `%`, cast back to uint32.

**When to use:** All three arithmetic primitives. The intermediate result of `a + b` can reach
`2 * (MAX_PRIME_32 - 1) ≈ 8.59e9` which overflows uint32 (max 4294967295).

```python
# Source: verified empirically — see test runs above
def mod_add_32(a, b, q):
    a64 = jnp.asarray(a, dtype=jnp.uint64)
    b64 = jnp.asarray(b, dtype=jnp.uint64)
    q64 = jnp.asarray(q, dtype=jnp.uint64)
    return jnp.asarray((a64 + b64) % q64, dtype=jnp.uint32)
```

**Key properties confirmed:**
- Works identically for scalar `jnp.uint32` and array `jnp.uint32` inputs
- Works when `q` is a Python int, `jnp.uint32`, or `jnp.uint64`
- JIT-compiles cleanly with `jax.jit`

### Pattern 2: Safe Modular Subtraction via uint64 + q

**What:** Compute `(a - b) % q` without going negative by adding `q` before subtracting.

**When to use:** `mod_sub_32`. Using `jnp.int64` (signed) is an alternative but requires an extra
sign-awareness step; uint64 with `+ q` is cleaner and has no sign-extension ambiguity.

```python
# Source: verified empirically — both methods produce correct results
def mod_sub_32(a, b, q):
    a64 = jnp.asarray(a, dtype=jnp.uint64)
    b64 = jnp.asarray(b, dtype=jnp.uint64)
    q64 = jnp.asarray(q, dtype=jnp.uint64)
    return jnp.asarray((a64 + q64 - b64) % q64, dtype=jnp.uint32)
```

**Overflow safety:** `a + q` has maximum value `(q - 1) + q = 2q - 1 ≈ 8.59e9` for MAX_PRIME_32,
which is far below uint64 max (`1.84e19`). [VERIFIED: `(MAX_PRIME_32-1) + MAX_PRIME_32 = 8589934581 < 2^64`]

### Pattern 3: uint64 Multiplication — No Overflow

**What:** Cast to uint64, multiply, reduce. The product of two sub-`q` values fits in uint64.

**When to use:** `mod_mul_32`. The product `(q-1)^2 ≈ 1.84e19` is just at the uint64 boundary:
`(4294967290)^2 = 18446744022169944100 < 2^64 = 18446744073709551616`. [VERIFIED empirically]

```python
# Source: verified empirically — (q-1)*(q-1) % q = 1 confirmed correct
def mod_mul_32(a, b, q):
    a64 = jnp.asarray(a, dtype=jnp.uint64)
    b64 = jnp.asarray(b, dtype=jnp.uint64)
    q64 = jnp.asarray(q, dtype=jnp.uint64)
    return jnp.asarray((a64 * b64) % q64, dtype=jnp.uint32)
```

**Critical proof:** `MAX_PRIME_32 = 4294967291`. `(MAX_PRIME_32 - 1)^2 = 18446744022169944100`.
uint64 max = `18446744073709551615`. Product fits with `4.7e10` to spare. [VERIFIED]

### Pattern 4: MLE Update via Primitive Composition

**What:** Linear interpolation `zero_eval + target_eval * (one_eval - zero_eval)` mod q.
Composed from three primitives so each arithmetic step uses its own overflow-safe implementation.

**Formula source:** `sumcheck_intro.md` §"Intuition behind the MLE Update formula":
`mle_update(z, o, t) = (o - z)*t + z`. Algebraically equivalent to D-03 form. [CITED: sumcheck_intro.md]

```python
# Source: verified against textbook example (F_17, t=8, [0,2,4,6]->[8,10,12,14])
def mle_update_32(zero_eval, one_eval, target_eval, *, q):
    diff   = mod_sub_32(one_eval, zero_eval, q)     # (o - z) mod q
    scaled = mod_mul_32(target_eval, diff, q)        # t * (o - z) mod q
    return mod_add_32(zero_eval, scaled, q)          # z + t*(o-z) mod q
```

**Result shape:** Output has the same shape as `zero_eval` and `one_eval` (both are half-length
arrays that the caller creates by splitting the full table). [VERIFIED: array smoke test]

### Anti-Patterns to Avoid

- **Raw uint32 arithmetic without upcast:** `(a + b) % q` where a, b are `jnp.uint32` silently
  wraps at `2^32` before the `%` is applied. The result is wrong for values near `MAX_PRIME_32`.
  [VERIFIED: `(4294967290 + 4294967290) % q` gives `4294967284` in uint32 instead of `8589934580 % q`]

- **Using jnp.int64 for mod_sub:** Signed int64 works but introduces sign-extension semantics.
  The uint64 + q pattern is cleaner and avoids any signed/unsigned confusion in downstream code.

- **Applying jax.vmap to mle_update_32:** `mle_update_32` already receives full arrays as inputs.
  Adding `vmap` would treat them as batches of scalars, adding XLA function-call overhead with no
  benefit. Plain array ops are the correct approach. [VERIFIED: benchmarked 512K elements in 0.47ms JIT]

- **Separating q casting:** Always cast `q` to uint64 inside the primitive, even if the caller
  already has a uint64 `q`. Double-casting is free; missing the cast causes silent overflow if the
  caller passes a uint32 `q` that gets widened incorrectly during mixed-dtype arithmetic.

---

## Don't Hand-Roll

| Problem                      | Don't Build              | Use Instead                         | Why                                              |
|------------------------------|--------------------------|-------------------------------------|--------------------------------------------------|
| Modular inverse (future)     | Extended Euclidean algo  | `pow(x, q-2, q)` (Python) or harness | Fermat's little theorem; already in test harness |
| Lagrange interpolation       | Custom polynomial eval   | `_lagrange_eval_at` in test harness | Already implemented in `tests/test_sumcheck.py`  |
| Multi-precision arithmetic   | Custom bigint             | 64-bit track is optional; skip      | uint64 is sufficient for 32-bit track            |

---

## Common Pitfalls

### Pitfall 1: uint32 Overflow in mod_add_32

**What goes wrong:** `a + b` silently wraps at `2^32` before `% q` executes, producing a wrong
result when `a + b > 2^32` (which happens when both inputs are near `MAX_PRIME_32`).

**Why it happens:** JAX `jnp.uint32` arithmetic has wrap-around semantics (no exception thrown).

**How to avoid:** Always cast to `jnp.uint64` before any addition of two 32-bit values.

**Warning signs:** Test case `overflow_add_full` (both `a=b=MAX_PRIME_32-1`) fails; result is a small
positive number instead of `(2*(MAX_PRIME_32-1)) % MAX_PRIME_32`.

### Pitfall 2: Negative Underflow in mod_sub_32

**What goes wrong:** Subtracting `uint32` values when `a < b` produces a large wrapping value
(e.g., `0 - 1 = 4294967295` in uint32), not a negative intermediate that `% q` can correct.

**Why it happens:** Unsigned integers have no negative representation; `a - b` where `a < b`
wraps to `2^32 + (a - b)`, which is congruent to `a - b` only mod `2^32`, not mod `q`.

**How to avoid:** Use `(a + q - b) % q` in uint64. This guarantees `a + q >= b` for any valid
inputs (since `a >= 0` and `b <= q-1`).

**Warning signs:** Test case `sub_negative_wrap` (`a=1, b=MAX_PRIME_32-1`) fails or returns
incorrect result.

### Pitfall 3: uint64 Product Near Boundary

**What goes wrong:** Assuming `(q-1)^2` overflows uint64 and implementing a more complex
multi-precision scheme unnecessarily.

**Why it safe:** `(MAX_PRIME_32 - 1)^2 = 18446744022169944100 < 2^64 = 18446744073709551616`.
There is headroom of `~4.7e10`. [VERIFIED]

**How to avoid:** Use plain `a64 * b64` with no overflow check for 32-bit track. The 64-bit track
(`MAX_PRIME_64 ≈ 1.84e19`) would overflow `uint64` and needs a different approach — but that is
optional and out of scope for Phase 1.

### Pitfall 4: q dtype Mismatch in Modular Reduction

**What goes wrong:** If `q` is uint32 and intermediate result is uint64, the `%` operation may
auto-promote correctly in JAX (verified: it promotes to uint64), but the explicit cast to uint64
is safer and more readable.

**How to avoid:** Always explicitly cast `q` to `jnp.uint64` at the start of each primitive.
Confirmed: `jnp.asarray(q_uint32, dtype=jnp.uint64)` preserves value correctly for all valid
32-bit primes.

### Pitfall 5: Confusing `target_eval` Array Shape in mle_update_32

**What goes wrong:** Assuming `target_eval` is always a scalar. In Phase 2, `sumcheck_32` passes
a scalar challenge (one value per round). In the MLE formula, `t * diff` broadcasts correctly
whether `target_eval` is scalar or array, so the composition is safe either way.

**How to avoid:** Let JAX broadcasting handle it. The `mod_mul_32` implementation uses
`jnp.asarray(target_eval, dtype=jnp.uint64)` which handles both.

**Warning signs:** No issue with Phase 1 tests (they only use scalar challenges). Would only appear
if someone mistakenly passes an array challenge.

---

## Code Examples

### Verified: mod_add_32

```python
# Verified against test edge cases: zero_zero, overflow_add_full, add_wrap_exact_q
import jax.numpy as jnp

def mod_add_32(a, b, q):
    a64 = jnp.asarray(a, dtype=jnp.uint64)
    b64 = jnp.asarray(b, dtype=jnp.uint64)
    q64 = jnp.asarray(q, dtype=jnp.uint64)
    return jnp.asarray((a64 + b64) % q64, dtype=jnp.uint32)
```

### Verified: mod_sub_32

```python
# Verified against: sub_negative_wrap (a=1, b=MAX_PRIME_32-1 -> expected 2)
def mod_sub_32(a, b, q):
    a64 = jnp.asarray(a, dtype=jnp.uint64)
    b64 = jnp.asarray(b, dtype=jnp.uint64)
    q64 = jnp.asarray(q, dtype=jnp.uint64)
    return jnp.asarray((a64 + q64 - b64) % q64, dtype=jnp.uint32)
```

### Verified: mod_mul_32

```python
# Verified: (MAX_PRIME_32-1)*(MAX_PRIME_32-1) % MAX_PRIME_32 = 1
def mod_mul_32(a, b, q):
    a64 = jnp.asarray(a, dtype=jnp.uint64)
    b64 = jnp.asarray(b, dtype=jnp.uint64)
    q64 = jnp.asarray(q, dtype=jnp.uint64)
    return jnp.asarray((a64 * b64) % q64, dtype=jnp.uint32)
```

### Verified: mle_update_32

```python
# Verified against sumcheck_intro.md textbook example:
# z=[0,2,4,6], o=[1,3,5,7], t=8, q=17 -> [8,10,12,14]
# Also verified with MAX_PRIME_32 edge values
def mle_update_32(zero_eval, one_eval, target_eval, *, q):
    diff   = mod_sub_32(one_eval, zero_eval, q)
    scaled = mod_mul_32(target_eval, diff, q)
    return mod_add_32(zero_eval, scaled, q)
```

---

## MLE Update: Formula Derivation from Spec

Source: `sumcheck_intro.md` §"Intuition behind the MLE Update formula" [CITED: sumcheck_intro.md]

Given two points at Boolean inputs `(0, f(0))` and `(1, f(1))`, the linear interpolation to
arbitrary field element `t` is:

```
mle_update(z, o, t) = (o - z)*t + z
                    = z + t*(o - z)    [equivalent, D-03 form]
```

Where:
- `z = zero_eval` — table value at current-variable = 0
- `o = one_eval` — table value at current-variable = 1
- `t = target_eval` — the challenge field element (scalar per round)

This formula halves the table length: input arrays `zero_eval` and `one_eval` each have `N/2`
elements (the caller splits the full-length table); the output has `N/2` elements.

**Algebraic properties confirmed:**
- When `t = 0`: result = `z` (correct: evaluates at 0)
- When `t = 1`: result = `o` (correct: evaluates at 1)
- When `t = 8, z = 0, o = 1, q = 17`: result = `8` [VERIFIED]
- When `t = 8, z = 2, o = 3, q = 17`: result = `10` [VERIFIED]

---

## Memory and Performance Notes

[ASSUMED: based on JAX 0.10.0 behavior observed in benchmarks]

**Array allocation per primitive call:**
Each `mod_*_32` call allocates three intermediate uint64 arrays (`a64`, `b64`, `q64`) plus the
final uint32 result. For a 1M-element array, this is `~32MB` of intermediate storage.

**JIT fusion:** With `jax.jit`, the XLA compiler can fuse the cast + operate + cast into a single
kernel, reducing intermediate materialization. The benchmark shows `~0.47ms` for 512K elements
(mle_update JIT on CPU). This is the expected operating range for vars20 (`2^19` per-variable
half-tables per round).

**vmap conclusion:** Plain array operations are correct and efficient. `jax.vmap` would add overhead
for this use case because the inputs are already arrays — `vmap` is for lifting scalar functions
to batch dimensions, not for operations that are already vectorized.

**Memory footprint estimate (vars20):**
- Per variable: `2^20 = 1M` uint32 values = `4MB`
- 6 variables × 4MB = `24MB` for all tables
- uint64 intermediates per round: `~3 × 6 × 4MB = ~72MB` peak (fused by JIT to less)
- Total: well within normal CPU memory constraints

---

## State of the Art

| Old Approach                          | Current Approach                            | Impact                                    |
|---------------------------------------|---------------------------------------------|-------------------------------------------|
| Python loop per table entry           | Vectorized JAX array ops                    | O(1) kernel dispatch vs O(N) Python calls |
| int overflow for mod arithmetic       | Explicit uint64 upcast-reduce-downcast      | Correctness for all inputs near MAX_PRIME |
| vmap for parallelism                  | Plain arrays (already vectorized)           | Simpler code, lower overhead              |
| jax_enable_x64=False (32-bit only)    | jax_enable_x64=True (uint64 available)      | Enables intermediate widening             |

---

## Assumptions Log

| #  | Claim                                                                 | Section               | Risk if Wrong                           |
|----|-----------------------------------------------------------------------|-----------------------|-----------------------------------------|
| A1 | JIT fusion reduces intermediate uint64 buffer materialization         | Memory & Performance  | Higher peak RAM; correctness unaffected |
| A2 | vmap adds overhead for elementwise ops on pre-existing arrays         | Anti-Patterns         | Minor; vmap would still produce correct results |

---

## Open Questions

1. **q value range across test cases**
   - What we know: `v4_case32_0` uses `q = 3603169181`, not `MAX_PRIME_32 = 4294967291`. The q
     varies per case and is always a 32-bit prime.
   - What's clear: Both values fit in uint32. The uint64 promotion pattern handles any 32-bit q.
   - Recommendation: Implement primitives generically (no hardcoded q assumptions).

2. **test_mod_add_32bit_edge_cases scope**
   - What we know: The 10 edge cases in the harness cover zero, wrap-at-q, overflow, negative-wrap.
     All are scalar `jnp.uint32` inputs.
   - What's clear: These are the primitive-level unit tests; they run independent of `--num-vars`.
   - Recommendation: Verify all 10 pass before proceeding to Phase 2.

---

## Environment Availability

| Dependency | Required By      | Available | Version | Fallback |
|------------|-----------------|-----------|---------|----------|
| JAX        | All kernels     | ✓         | 0.10.0  | —        |
| jax_enable_x64 | uint64 dtype | ✓ (configured) | — | —     |
| uv         | Test runner     | ✓         | —       | pip      |
| pytest     | Test framework  | ✓         | —       | —        |
| numpy      | Test data .npz  | ✓         | —       | —        |
| CPU backend| Default compute | ✓         | —       | GPU/TPU optional |

[VERIFIED: `uv run python -c "import jax; print(jax.__version__)"` → 0.10.0]
[VERIFIED: `uv run python -c "jax.config.update('jax_enable_x64', True); import jax.numpy as jnp; print(jnp.uint64)"` → class available]

---

## Validation Architecture

> `workflow.nyquist_validation` is `false` in `.planning/config.json`. This section is omitted.

---

## Security Domain

> Not applicable — this is a pure arithmetic/algorithm implementation with no I/O, no auth,
> no network, and no user-controlled input beyond numeric values within a known prime field.

---

## Sources

### Primary (HIGH confidence)
- `sumcheck_intro.md` (project file) — MLE update formula, protocol walkthrough, pseudocode
- `tests/test_sumcheck.py` (project file) — exact edge cases, input dtypes, expected outputs
- `tests/data/vars4/v4_case32_0_meta.json` — concrete test data structure, q value examples
- Empirical verification via `uv run python` — all code patterns confirmed working at runtime

### Secondary (MEDIUM confidence)
- [JAX Type Promotion Design](https://docs.jax.dev/en/latest/jep/9407-type-promotion.html) — uint32+uint64 promotion behavior
- [JAX Sharp Bits](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html) — integer overflow semantics
- Reference: `reference/Sum-Check Protocol and Multilinear Extensions (MLEs).md` — MLE Lagrange basis background

### Tertiary (LOW confidence)
- WebSearch results on JAX uint32/uint64 modular arithmetic — confirmed by empirical tests above

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — JAX 0.10.0 installed and verified working
- Architecture: HIGH — all patterns verified with running code
- Pitfalls: HIGH — demonstrated empirically (overflow, underflow, wrong dtype)
- MLE formula: HIGH — verified against textbook example from spec

**Research date:** 2026-04-30
**Valid until:** 2026-06-01 (stable JAX API; arithmetic semantics do not change between patch versions)
