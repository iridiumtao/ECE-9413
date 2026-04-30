# Phase 1: Primitives + MLE Update — Pattern Map

**Mapped:** 2026-04-30
**Files analyzed:** 1 (student.py — only editable file; four stub functions)
**Analogs found:** 4 / 4 (all stubs have call-site patterns extracted from provided.py and test_sumcheck.py)

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `student.py` :: `mod_add_32` | utility (arithmetic kernel) | transform | `student.py` :: `mod_add` dispatcher (lines 82–89) | exact call-site |
| `student.py` :: `mod_sub_32` | utility (arithmetic kernel) | transform | `student.py` :: `mod_sub` dispatcher (lines 92–99) | exact call-site |
| `student.py` :: `mod_mul_32` | utility (arithmetic kernel) | transform | `student.py` :: `mod_mul` dispatcher (lines 102–109) | exact call-site |
| `student.py` :: `mle_update_32` | utility (folding kernel) | transform | `student.py` :: `mle_update` dispatcher (lines 129–136) | exact call-site |

Note: there are no other arithmetic or JAX kernel files in this codebase. The only analogs are the
frozen dispatcher wrappers in `student.py` itself, plus the test harness call sites in
`tests/test_sumcheck.py` which show exact signature usage and expected dtype contracts.

---

## Pattern Assignments

### `student.py` :: `mod_add_32` (utility, transform) — PRIM-01

**Analog call site:** `student.py` lines 82–89 (frozen `mod_add` dispatcher)

**Dispatcher pattern** (`student.py` lines 82–89):
```python
def mod_add(a, b, q, *, bit_width=32):
    if int(bit_width) == 32:
        return mod_add_32(a, b, q)
    if int(bit_width) == 64:
        return mod_add_64(a, b, q)
    if int(bit_width) == 128:
        return mod_add_128(a, b, q)
    raise ValueError(f"Unsupported bit_width={bit_width}")
```

**Test harness call pattern** (`tests/test_sumcheck.py` lines 196–201):
```python
@pytest.mark.parametrize("a,b", EDGE_CASES_32)
def test_mod_add_32bit_edge_cases(a, b):
    q = MAX_PRIME_32
    expected = ((int(a) % q) + (int(b) % q)) % q
    got = _eval_mod_op(student.mod_add_32, a, b, q, input_dtype=jnp.uint32)
    assert got == expected
```

**Input dtype contract** (`tests/test_sumcheck.py` lines 137–149): inputs arrive as `jnp.uint32`
scalars (via `_eval_mod_op` with `input_dtype=jnp.uint32`); `q` also `jnp.uint32`.

**Implementation pattern to copy** (from RESEARCH.md Pattern 1):
```python
def mod_add_32(a, b, q):
    a64 = jnp.asarray(a, dtype=jnp.uint64)
    b64 = jnp.asarray(b, dtype=jnp.uint64)
    q64 = jnp.asarray(q, dtype=jnp.uint64)
    return jnp.asarray((a64 + b64) % q64, dtype=jnp.uint32)
```

**Required import** (`student.py` lines 9–12):
```python
import jax
jax.config.update("jax_enable_x64", True)
# jax.numpy is imported as jnp elsewhere in the file — add:
import jax.numpy as jnp
```

**Output contract:** `jnp.uint32` scalar (for scalar inputs) or `jnp.uint32` array (for array inputs).

---

### `student.py` :: `mod_sub_32` (utility, transform) — PRIM-02

**Analog call site:** `student.py` lines 92–99 (frozen `mod_sub` dispatcher)

**Dispatcher pattern** (`student.py` lines 92–99):
```python
def mod_sub(a, b, q, *, bit_width=32):
    if int(bit_width) == 32:
        return mod_sub_32(a, b, q)
    ...
```

**Test harness call pattern** (`tests/test_sumcheck.py` lines 204–209):
```python
@pytest.mark.parametrize("a,b", EDGE_CASES_32)
def test_mod_sub_32bit_edge_cases(a, b):
    q = MAX_PRIME_32
    expected = ((int(a) % q) - (int(b) % q)) % q
    got = _eval_mod_op(student.mod_sub_32, a, b, q, input_dtype=jnp.uint32)
    assert got == expected
```

**Critical edge case** (`tests/test_sumcheck.py` line 158):
```python
pytest.param(1, MAX_PRIME_32 - 1, id="sub_negative_wrap"),
# a < b: must not underflow uint32; (1 - (q-1)) must yield 2, not a wrapped value
```

**Implementation pattern to copy** (from RESEARCH.md Pattern 2):
```python
def mod_sub_32(a, b, q):
    a64 = jnp.asarray(a, dtype=jnp.uint64)
    b64 = jnp.asarray(b, dtype=jnp.uint64)
    q64 = jnp.asarray(q, dtype=jnp.uint64)
    return jnp.asarray((a64 + q64 - b64) % q64, dtype=jnp.uint32)
```

**Why `a + q - b` not `a - b`:** unsigned subtraction of `uint32` when `a < b` wraps to `2^32 +
(a - b)`, which is congruent to `a - b` only mod `2^32`, not mod `q`. Adding `q` in uint64 space
before subtracting guarantees `a + q >= b` for all valid field inputs.

**Output contract:** `jnp.uint32`, same shape as inputs.

---

### `student.py` :: `mod_mul_32` (utility, transform) — PRIM-03

**Analog call site:** `student.py` lines 102–109 (frozen `mod_mul` dispatcher)

**Dispatcher pattern** (`student.py` lines 102–109):
```python
def mod_mul(a, b, q, *, bit_width=32):
    if int(bit_width) == 32:
        return mod_mul_32(a, b, q)
    ...
```

**Test harness call pattern** (`tests/test_sumcheck.py` lines 212–217):
```python
@pytest.mark.parametrize("a,b", EDGE_CASES_32)
def test_mod_mul_32bit_edge_cases(a, b):
    q = MAX_PRIME_32
    expected = ((int(a) % q) * (int(b) % q)) % q
    got = _eval_mod_op(student.mod_mul_32, a, b, q, input_dtype=jnp.uint32)
    assert got == expected
```

**Critical edge case** (`tests/test_sumcheck.py` line 159):
```python
pytest.param(MAX_PRIME_32 - 1, MAX_PRIME_32 - 1, id="overflow_add_full"),
# (q-1)*(q-1) must equal 1 mod q; product ~1.84e19 fits in uint64
```

**Implementation pattern to copy** (from RESEARCH.md Pattern 3):
```python
def mod_mul_32(a, b, q):
    a64 = jnp.asarray(a, dtype=jnp.uint64)
    b64 = jnp.asarray(b, dtype=jnp.uint64)
    q64 = jnp.asarray(q, dtype=jnp.uint64)
    return jnp.asarray((a64 * b64) % q64, dtype=jnp.uint32)
```

**Overflow safety proof:** `(MAX_PRIME_32 - 1)^2 = 18446744022169944100 < uint64_max =
18446744073709551615`. No multi-precision arithmetic needed for the 32-bit track.

**Output contract:** `jnp.uint32`, same shape as inputs.

---

### `student.py` :: `mle_update_32` (utility, transform) — MLE-01

**Analog call site:** `student.py` lines 129–136 (frozen `mle_update` dispatcher)

**Dispatcher pattern** (`student.py` lines 129–136):
```python
def mle_update(zero_eval, one_eval, target_eval, *, q, bit_width=32):
    if int(bit_width) == 32:
        return mle_update_32(zero_eval, one_eval, target_eval, q=q)
    ...
```

**Stub signature in student.py** (`student.py` lines 112–114):
```python
def mle_update_32(zero_eval, one_eval, target_eval, *, q):
    """Compulsory 32-bit MLE update."""
    raise NotImplementedError
```

**Note:** `q` is a keyword-only argument (after `*`). The dispatcher passes it as `q=q`
(line 131). This signature is frozen — do not change it.

**How Phase 2 will call this function** (from CONTEXT.md §Integration Points and ROADMAP.md §Phase 2
task 2c): `sumcheck_32` will call `mle_update_32(zero_half, one_half, challenge, q=q)` where
`zero_half` and `one_half` are `uint32` arrays of length `N/2` (split from the full table) and
`challenge` is a `uint32` scalar from the `challenges` array.

**Implementation pattern to copy** (from RESEARCH.md Pattern 4):
```python
def mle_update_32(zero_eval, one_eval, target_eval, *, q):
    diff   = mod_sub_32(one_eval, zero_eval, q)   # (one - zero) mod q
    scaled = mod_mul_32(target_eval, diff, q)      # target * (one - zero) mod q
    return mod_add_32(zero_eval, scaled, q)        # zero + target*(one-zero) mod q
```

**Formula source:** `sumcheck_intro.md` §"Intuition behind the MLE Update formula":
`mle_update(z, o, t) = z + t*(o - z)` (D-03 form, equivalent to `(o-z)*t + z`).

**Output contract:** `jnp.uint32` array of the same shape as `zero_eval` and `one_eval`.
When `target_eval` is a scalar challenge and `zero_eval`/`one_eval` are arrays, broadcasting
applies correctly — no `jax.vmap` needed.

**Verified correctness:** textbook example from spec (F_17, z=[0,2,4,6], o=[1,3,5,7], t=8)
yields [8,10,12,14]. The primitive-composition approach is sufficient.

---

## Shared Patterns

### jax_enable_x64 Configuration
**Source:** `student.py` lines 9–12; `tests/test_sumcheck.py` lines 7–9
**Apply to:** Top of `student.py` (already set — do not add again)
```python
import jax
jax.config.update("jax_enable_x64", True)
```
This makes `jnp.uint64` available. It must be called before any JAX array operations. It is
already present in `student.py` — no action required.

### jnp Import
**Source:** `tests/test_sumcheck.py` line 9; `student.py` (missing — must be added)
**Apply to:** `student.py` top-level imports
```python
import jax.numpy as jnp
```
The current `student.py` does not import `jax.numpy`. All four implementations require it.

### uint64 Promotion Pattern (cast → operate → cast back)
**Source:** RESEARCH.md Patterns 1–3; verified empirically
**Apply to:** `mod_add_32`, `mod_sub_32`, `mod_mul_32`
```python
x64 = jnp.asarray(x, dtype=jnp.uint64)   # cast input
# ... operate in uint64 ...
return jnp.asarray(result64, dtype=jnp.uint32)  # cast output
```
Always cast `q` to `jnp.uint64` explicitly inside each primitive, even if the caller already
holds a uint64 `q`. Double-casting is free; omitting the cast causes silent overflow on
mixed-dtype paths.

### Scalar and Array Uniformity
**Source:** RESEARCH.md §"The harness always passes q as a jnp.uint32 scalar"
**Apply to:** All four functions
`jnp.asarray(x, dtype=jnp.uint64)` works identically for scalar `jnp.uint32` and array
`jnp.uint32` inputs. No branching on input shape is needed.

### Test Harness _eval_mod_op Wrapper
**Source:** `tests/test_sumcheck.py` lines 137–149
**Apply to:** Understanding how the harness invokes the three arithmetic primitives
```python
def _eval_mod_op(fn, a, b, q, *, input_dtype=None):
    q_int = int(q)
    a_int = int(a) % q_int
    b_int = int(b) % q_int
    if input_dtype is None:
        q_arg = q_int
        a_arg = a_int
        b_arg = b_int
    else:
        q_arg = jnp.asarray(q_int, dtype=input_dtype)
        a_arg = jnp.asarray(a_int, dtype=input_dtype)
        b_arg = jnp.asarray(b_int, dtype=input_dtype)
    return _scalar_to_int(fn(a_arg, b_arg, q_arg)) % q_int
```
All three primitives receive `jnp.uint32` scalars when tested by the edge-case parametrize
suite. The return value is extracted with `_scalar_to_int` — so the function must return a
JAX scalar (0-D array), not a Python int.

---

## No Analog Found

No files in this phase lack an analog. All four function stubs have their call sites, dispatcher
wrappers, and expected call patterns fully documented in the frozen `student.py` and test harness.

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| — | — | — | All stubs covered |

---

## Metadata

**Analog search scope:** `student.py` (dispatcher section lines 82–136), `tests/test_sumcheck.py`
(edge-case tests lines 152–265 and harness helper lines 137–149)
**Files scanned:** 5 (`01-CONTEXT.md`, `01-RESEARCH.md`, `student.py`, `provided.py`,
`tests/test_sumcheck.py`) plus `.planning/ROADMAP.md` and `.planning/REQUIREMENTS.md`
**Pattern extraction date:** 2026-04-30
