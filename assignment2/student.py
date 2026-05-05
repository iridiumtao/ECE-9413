"""
Assignment 2 student implementation reference skeleton.

This file documents the frozen student-facing API.
Only 32-bit kernels are compulsory in the base track.
64-bit and 128-bit kernels are intentionally left unimplemented here.
"""

from __future__ import annotations

import functools

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


# -----------------------------------------------------------------------------
# 32-bit primitives (compulsory)
# -----------------------------------------------------------------------------

def mod_add_32(a, b, q):
    """Return (a + b) mod q for the 32-bit track."""
    a64 = jnp.asarray(a, dtype=jnp.uint64)
    b64 = jnp.asarray(b, dtype=jnp.uint64)
    q64 = jnp.asarray(q, dtype=jnp.uint64)
    return jnp.asarray((a64 + b64) % q64, dtype=jnp.uint32)


def mod_sub_32(a, b, q):
    """Return (a - b) mod q for the 32-bit track."""
    a64 = jnp.asarray(a, dtype=jnp.uint64)
    b64 = jnp.asarray(b, dtype=jnp.uint64)
    q64 = jnp.asarray(q, dtype=jnp.uint64)
    return jnp.asarray((a64 + q64 - b64) % q64, dtype=jnp.uint32)


def mod_mul_32(a, b, q):
    """Return (a * b) mod q for the 32-bit track."""
    a64 = jnp.asarray(a, dtype=jnp.uint64)
    b64 = jnp.asarray(b, dtype=jnp.uint64)
    q64 = jnp.asarray(q, dtype=jnp.uint64)
    return jnp.asarray((a64 * b64) % q64, dtype=jnp.uint32)


# -----------------------------------------------------------------------------
# 64-bit primitives (optional, left for future implementation)
# -----------------------------------------------------------------------------

def mod_add_64(a, b, q):
    """Optional 64-bit modular add kernel."""
    # TODO(student): implement when enabling 64-bit track.
    raise NotImplementedError


def mod_sub_64(a, b, q):
    """Optional 64-bit modular subtract kernel."""
    # TODO(student): implement when enabling 64-bit track.
    raise NotImplementedError


def mod_mul_64(a, b, q):
    """Optional 64-bit modular multiply kernel."""
    # TODO(student): implement when enabling 64-bit track.
    raise NotImplementedError


# -----------------------------------------------------------------------------
# 128-bit primitives (optional, left for future implementation)
# -----------------------------------------------------------------------------

def mod_add_128(a, b, q):
    """Optional 128-bit modular add kernel."""
    # TODO(student): implement when enabling 128-bit track.
    raise NotImplementedError


def mod_sub_128(a, b, q):
    """Optional 128-bit modular subtract kernel."""
    # TODO(student): implement when enabling 128-bit track.
    raise NotImplementedError


def mod_mul_128(a, b, q):
    """Optional 128-bit modular multiply kernel."""
    # TODO(student): implement when enabling 128-bit track.
    raise NotImplementedError


# -----------------------------------------------------------------------------
# Frozen dispatch API
# -----------------------------------------------------------------------------

def mod_add(a, b, q, *, bit_width=32):
    if int(bit_width) == 32:
        return mod_add_32(a, b, q)
    if int(bit_width) == 64:
        return mod_add_64(a, b, q)
    if int(bit_width) == 128:
        return mod_add_128(a, b, q)
    raise ValueError(f"Unsupported bit_width={bit_width}")


def mod_sub(a, b, q, *, bit_width=32):
    if int(bit_width) == 32:
        return mod_sub_32(a, b, q)
    if int(bit_width) == 64:
        return mod_sub_64(a, b, q)
    if int(bit_width) == 128:
        return mod_sub_128(a, b, q)
    raise ValueError(f"Unsupported bit_width={bit_width}")


def mod_mul(a, b, q, *, bit_width=32):
    if int(bit_width) == 32:
        return mod_mul_32(a, b, q)
    if int(bit_width) == 64:
        return mod_mul_64(a, b, q)
    if int(bit_width) == 128:
        return mod_mul_128(a, b, q)
    raise ValueError(f"Unsupported bit_width={bit_width}")


def mle_update_32(zero_eval, one_eval, target_eval, *, q):
    """Compulsory 32-bit MLE update.

    Linear interpolation: result = zero_eval + target_eval * (one_eval - zero_eval) mod q.
    Composed from mod_sub_32, mod_mul_32, mod_add_32 — each handles its own overflow
    and underflow safety via uint64 promotion. Plain array operations broadcast across
    array zero_eval / one_eval inputs and a scalar target_eval challenge.
    """
    diff = mod_sub_32(one_eval, zero_eval, q)        # (one - zero) mod q
    scaled = mod_mul_32(target_eval, diff, q)         # target * (one - zero) mod q
    return mod_add_32(zero_eval, scaled, q)           # zero + target * (one - zero) mod q


def mle_update_64(zero_eval, one_eval, target_eval, *, q):
    """Optional 64-bit MLE update."""
    # TODO(student): implement when enabling 64-bit track.
    raise NotImplementedError


def mle_update_128(zero_eval, one_eval, target_eval, *, q):
    """Optional 128-bit MLE update."""
    # TODO(student): implement when enabling 128-bit track.
    raise NotImplementedError


def mle_update(zero_eval, one_eval, target_eval, *, q, bit_width=32):
    if int(bit_width) == 32:
        return mle_update_32(zero_eval, one_eval, target_eval, q=q)
    if int(bit_width) == 64:
        return mle_update_64(zero_eval, one_eval, target_eval, q=q)
    if int(bit_width) == 128:
        return mle_update_128(zero_eval, one_eval, target_eval, q=q)
    raise ValueError(f"Unsupported bit_width={bit_width}")


# -----------------------------------------------------------------------------
# Barrett reduction helpers (GPU-safe, avoids XLA IDIV on element-wise arrays).
# Copied from teammate v9 implementation.
# -----------------------------------------------------------------------------

def _barrett_mu_64(q):
    """Return mu = floor(2^64 / q) as a Python int."""
    return (1 << 64) // int(q)


def _mulhi_u64(x, y):
    """Top 64 bits of the 128-bit product x*y, both interpreted as uint64.

    Schoolbook split:
        x = xh*2^32 + xl,  y = yh*2^32 + yl
        x*y = xh*yh * 2^64 + (xh*yl + xl*yh) * 2^32 + xl*yl
    The high 64 bits are xh*yh plus the carries out of the middle and low
    columns. We accumulate the low/mid carries in a way that never needs
    more than uint64 to stay exact: low-32 contributions sum to < 3*2^32
    which is safely uint64.
    """
    mask32 = jnp.uint64(0xFFFFFFFF)
    xh = x >> jnp.uint64(32)
    xl = x & mask32
    yh = y >> jnp.uint64(32)
    yl = y & mask32

    ll = xl * yl                        # < 2^64 (each factor < 2^32)
    lh = xl * yh                        # < 2^64
    hl = xh * yl                        # < 2^64
    hh = xh * yh                        # < 2^64

    # Sum of low halves of (lh, hl) and high half of ll: each < 2^32, total < 3*2^32 < 2^34.
    mid = (ll >> jnp.uint64(32)) + (lh & mask32) + (hl & mask32)
    return hh + (lh >> jnp.uint64(32)) + (hl >> jnp.uint64(32)) + (mid >> jnp.uint64(32))


def _barrett_reduce_u64_to_u32(x_u64, q_u64, mu_u64):
    """x mod q for x in [0, q^2), q < 2^32. Returns uint32."""
    q_hat = _mulhi_u64(x_u64, mu_u64)
    r = x_u64 - q_hat * q_u64           # uint64 wrap is fine: target value is exact mod 2^64
    r = jnp.where(r >= q_u64, r - q_u64, r)
    r = jnp.where(r >= q_u64, r - q_u64, r)
    return r.astype(jnp.uint32)


def _mod_mul_barrett_u32(a_u32, b_u32, q_u64, mu_u64):
    """(a*b) mod q via Barrett. a,b < q < 2^32."""
    prod = a_u32.astype(jnp.uint64) * b_u32.astype(jnp.uint64)
    return _barrett_reduce_u64_to_u32(prod, q_u64, mu_u64)


def _mod_add_branchless_u32(a_u32, b_u32, q_u64):
    """(a+b) mod q via single conditional subtract. a,b < q."""
    s = a_u32.astype(jnp.uint64) + b_u32.astype(jnp.uint64)   # < 2^33
    s = jnp.where(s >= q_u64, s - q_u64, s)
    return s.astype(jnp.uint32)


def _mod_sub_branchless_u32(a_u32, b_u32, q_u64):
    """(a-b) mod q via add-q-then-conditional-subtract. a,b < q."""
    t = a_u32.astype(jnp.uint64) + q_u64 - b_u32.astype(jnp.uint64)   # in [1, 2q)
    t = jnp.where(t >= q_u64, t - q_u64, t)
    return t.astype(jnp.uint32)


def _mle_update_barrett_u32(z, o, t, q_u64, mu_u64):
    """(o - z)*t + z  mod q via Barrett."""
    diff = _mod_sub_branchless_u32(o, z, q_u64)
    scaled = _mod_mul_barrett_u32(diff, t, q_u64, mu_u64)
    return _mod_add_branchless_u32(scaled, z, q_u64)


def _eval_sum_mod_barrett(expr, vals_by_name, q_u64, mu_u64):
    """Sum of poly(x) over all x, one term at a time, Barrett-backed.

    Reduces each term to a scalar before accumulating; avoids materializing
    the full composed-polynomial array. Accumulates in raw uint64; safe since
    sum < num_terms * N/2 * q < 2^54 < q^2, so Barrett final reduction is valid.
    """
    acc = jnp.uint64(0)
    for term in expr:
        prod = vals_by_name[term[0]]
        for v in term[1:]:
            prod = _mod_mul_barrett_u32(prod, vals_by_name[v], q_u64, mu_u64)
        acc = acc + jnp.sum(prod.astype(jnp.uint64))
    return _barrett_reduce_u64_to_u32(acc, q_u64, mu_u64)


# -----------------------------------------------------------------------------
# Exp05 CPU kernel: Exp03 no-mul shortcuts for t=2/t=3 + deferred mod reduction
# (single % q per t-point, not per term) + running-add for t>=4.
# -----------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnames=("expression", "num_rounds"))
def _sumcheck_32_exp05(eval_tables, *, q, expression, challenges, num_rounds):
    """Exp05 CPU kernel: Exp03 no-mul shortcuts + v9 deferred mod + running-add for t>=4."""
    degree = max(len(term) for term in expression)

    # OPT-03 (D-02): only materialize tables for variables actually used by the expression.
    used_vars = set().union(*expression)
    tables = {v: jnp.asarray(eval_tables[v], dtype=jnp.uint32) for v in used_vars}

    q32 = jnp.asarray(q, dtype=jnp.uint32)
    q64 = jnp.asarray(q, dtype=jnp.uint64)

    round_evals_list = []

    for i in range(num_rounds):
        # Pre-compute evens/odds once per round; reused in the v-loop and fold step.
        evens = {var: tbl[::2]  for var, tbl in tables.items()}
        odds  = {var: tbl[1::2] for var, tbl in tables.items()}

        # Deferred running-add: compute diffs once; advance running at start of each v>=4 iter.
        if degree >= 4:
            diffs = {var: mod_sub_32(odds[var], evens[var], q32) for var in tables}
        running = {}  # seeded at end of v=3 iteration; used from v=4 onward

        g_i_evals = []

        for v in range(degree + 1):
            # ADVANCE-BEFORE-USE: advance running BEFORE computing term_product for v>=4.
            # running starts as the t=3 per-variable values (seeded at end of v=3 iter).
            # For v=4: running becomes t=3+diffs = t=4 values. For v=5: t=5. Etc.
            if v >= 4 and degree >= 4:
                running = {var: mod_add_32(running[var], diffs[var], q32) for var in tables}

            # Accumulate all terms in raw uint64; apply single % q at end (deferred mod).
            # Safety: each term_product element < q < 2^32; up to 2^20 elements and ~10 terms
            # gives raw sum < 10 * 2^20 * 2^32 = 2^55.3 < 2^64. No overflow.
            g_i_raw = jnp.uint64(0)

            for term in expression:
                if v == 0:
                    # t=0 shortcut: even-indexed entries are the zero-eval side directly.
                    term_product = evens[term[0]]
                    for var in term[1:]:
                        term_product = mod_mul_32(term_product, evens[var], q)
                elif v == 1:
                    # t=1 shortcut: odd-indexed entries are the one-eval side directly.
                    term_product = odds[term[0]]
                    for var in term[1:]:
                        term_product = mod_mul_32(term_product, odds[var], q)
                elif v == 2:
                    # OPT-04 (D-04): no-mul MLE shortcut for v=2.
                    # mle_update_32(z, o, 2) = 2*o - z (mod q).
                    z64 = evens[term[0]].astype(jnp.uint64)
                    o64 = odds[term[0]].astype(jnp.uint64)
                    term_product = ((2*o64 + q64 - z64) % q64).astype(jnp.uint32)
                    for var in term[1:]:
                        z64_v = evens[var].astype(jnp.uint64)
                        o64_v = odds[var].astype(jnp.uint64)
                        factor = ((2*o64_v + q64 - z64_v) % q64).astype(jnp.uint32)
                        term_product = mod_mul_32(term_product, factor, q)
                elif v == 3:
                    # OPT-04 (D-04): no-mul MLE shortcut for v=3.
                    # mle_update_32(z, o, 3) = 3*o - 2*z (mod q).
                    z64 = evens[term[0]].astype(jnp.uint64)
                    o64 = odds[term[0]].astype(jnp.uint64)
                    term_product = ((3*o64 + 2*q64 - 2*z64) % q64).astype(jnp.uint32)
                    for var in term[1:]:
                        z64_v = evens[var].astype(jnp.uint64)
                        o64_v = odds[var].astype(jnp.uint64)
                        factor = ((3*o64_v + 2*q64 - 2*z64_v) % q64).astype(jnp.uint32)
                        term_product = mod_mul_32(term_product, factor, q)
                else:
                    # v >= 4: running already advanced to the t=v value at the top of this iter.
                    term_product = running[term[0]]
                    for var in term[1:]:
                        term_product = mod_mul_32(term_product, running[var], q)

                g_i_raw = g_i_raw + jnp.sum(term_product.astype(jnp.uint64))

            g_i_evals.append((g_i_raw % q64).astype(jnp.uint32))

            # SEED running at end of v=3 so it is ready for the advance at v=4.
            if v == 3 and degree >= 4:
                running = {}
                for var in tables:
                    z64_v = evens[var].astype(jnp.uint64)
                    o64_v = odds[var].astype(jnp.uint64)
                    running[var] = ((3*o64_v + 2*q64 - 2*z64_v) % q64).astype(jnp.uint32)

        round_evals_list.append(jnp.stack(g_i_evals))

        # Fold all tables using challenges[i]; reuses pre-computed evens/odds.
        if i < len(challenges):
            r = challenges[i]
            tables = {
                var: mle_update_32(evens[var], odds[var], r, q=q)
                for var in tables
            }

    round_evals = jnp.stack(round_evals_list)

    # claim0 = g_0(0) + g_0(1) mod q.
    claim0 = mod_add_32(round_evals_list[0][0], round_evals_list[0][1], q)

    return claim0, round_evals


# -----------------------------------------------------------------------------
# Exp05 GPU kernel: same structure as _sumcheck_32_exp05 but uses Barrett ops
# throughout to avoid XLA IDIV on element-wise arrays (GPU-friendly).
# -----------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnames=("expression", "num_rounds"))
def _sumcheck_32_barrett_exp05(eval_tables, *, q, expression, challenges, num_rounds):
    """Exp05 GPU kernel: Barrett ops + deferred mod + no-mul shortcuts + running-add for t>=4."""
    degree = max(len(term) for term in expression)

    # OPT-03 (D-02): only materialize tables for variables actually used by the expression.
    used_vars = set().union(*expression)
    tables = {v: jnp.asarray(eval_tables[v], dtype=jnp.uint32) for v in used_vars}

    q_u64 = jnp.uint64(int(q))
    mu_u64 = jnp.uint64(_barrett_mu_64(int(q)))
    q64 = q_u64  # alias for no-mul shortcut formulas

    round_evals_list = []

    for i in range(num_rounds):
        # Pre-compute evens/odds once per round; reused in the v-loop and fold step.
        evens = {var: tbl[::2]  for var, tbl in tables.items()}
        odds  = {var: tbl[1::2] for var, tbl in tables.items()}

        # Deferred running-add: compute diffs once; advance running at start of each v>=4 iter.
        if degree >= 4:
            diffs = {var: _mod_sub_branchless_u32(odds[var], evens[var], q_u64) for var in tables}
        running = {}  # seeded at end of v=3 iteration; used from v=4 onward

        g_i_evals = []

        for v in range(degree + 1):
            # ADVANCE-BEFORE-USE: advance running BEFORE computing term_product for v>=4.
            if v >= 4 and degree >= 4:
                running = {
                    var: _mod_add_branchless_u32(running[var], diffs[var], q_u64)
                    for var in tables
                }

            # Accumulate all terms in raw uint64; apply single Barrett reduction at end.
            g_i_raw = jnp.uint64(0)

            for term in expression:
                if v == 0:
                    # t=0 shortcut: even-indexed entries are the zero-eval side directly.
                    term_product = evens[term[0]]
                    for var in term[1:]:
                        term_product = _mod_mul_barrett_u32(term_product, evens[var], q_u64, mu_u64)
                elif v == 1:
                    # t=1 shortcut: odd-indexed entries are the one-eval side directly.
                    term_product = odds[term[0]]
                    for var in term[1:]:
                        term_product = _mod_mul_barrett_u32(term_product, odds[var], q_u64, mu_u64)
                elif v == 2:
                    # OPT-04 (D-04): no-mul MLE shortcut for v=2.
                    # mle_update(z, o, 2) = 2*o - z (mod q).
                    z64 = evens[term[0]].astype(jnp.uint64)
                    o64 = odds[term[0]].astype(jnp.uint64)
                    term_product = ((2*o64 + q64 - z64) % q64).astype(jnp.uint32)
                    for var in term[1:]:
                        z64_v = evens[var].astype(jnp.uint64)
                        o64_v = odds[var].astype(jnp.uint64)
                        factor = ((2*o64_v + q64 - z64_v) % q64).astype(jnp.uint32)
                        term_product = _mod_mul_barrett_u32(term_product, factor, q_u64, mu_u64)
                elif v == 3:
                    # OPT-04 (D-04): no-mul MLE shortcut for v=3.
                    # mle_update(z, o, 3) = 3*o - 2*z (mod q).
                    z64 = evens[term[0]].astype(jnp.uint64)
                    o64 = odds[term[0]].astype(jnp.uint64)
                    term_product = ((3*o64 + 2*q64 - 2*z64) % q64).astype(jnp.uint32)
                    for var in term[1:]:
                        z64_v = evens[var].astype(jnp.uint64)
                        o64_v = odds[var].astype(jnp.uint64)
                        factor = ((3*o64_v + 2*q64 - 2*z64_v) % q64).astype(jnp.uint32)
                        term_product = _mod_mul_barrett_u32(term_product, factor, q_u64, mu_u64)
                else:
                    # v >= 4: running already advanced to the t=v value at the top of this iter.
                    term_product = running[term[0]]
                    for var in term[1:]:
                        term_product = _mod_mul_barrett_u32(term_product, running[var], q_u64, mu_u64)

                g_i_raw = g_i_raw + jnp.sum(term_product.astype(jnp.uint64))

            g_i_evals.append(_barrett_reduce_u64_to_u32(g_i_raw, q_u64, mu_u64))

            # SEED running at end of v=3 so it is ready for the advance at v=4.
            if v == 3 and degree >= 4:
                running = {}
                for var in tables:
                    z64_v = evens[var].astype(jnp.uint64)
                    o64_v = odds[var].astype(jnp.uint64)
                    running[var] = ((3*o64_v + 2*q64 - 2*z64_v) % q64).astype(jnp.uint32)

        round_evals_list.append(jnp.stack(g_i_evals))

        # Fold all tables using challenges[i]; reuses pre-computed evens/odds.
        if i < len(challenges):
            r = challenges[i]
            tables = {
                var: _mle_update_barrett_u32(evens[var], odds[var], r, q_u64, mu_u64)
                for var in tables
            }

    round_evals = jnp.stack(round_evals_list)

    # claim0 = g_0(0) + g_0(1) mod q.
    claim0 = _mod_add_branchless_u32(round_evals_list[0][0], round_evals_list[0][1], q_u64)

    return claim0, round_evals


def _adaptive_dispatch_32_exp05(eval_tables, *, q, expression, challenges, num_rounds):
    """Route CPU to _sumcheck_32_exp05, GPU to _sumcheck_32_barrett_exp05."""
    try:
        backend = jax.devices()[0].platform
    except Exception:
        backend = "cpu"
    if backend == "gpu":
        return _sumcheck_32_barrett_exp05(
            eval_tables, q=q, expression=expression,
            challenges=challenges, num_rounds=num_rounds,
        )
    return _sumcheck_32_exp05(
        eval_tables, q=q, expression=expression,
        challenges=challenges, num_rounds=num_rounds,
    )


def sumcheck_32(eval_tables, *, q, expression, challenges, num_rounds):
    """Compulsory 32-bit sumcheck path — Exp05 adaptive dispatch."""
    return _adaptive_dispatch_32_exp05(
        eval_tables, q=q, expression=expression,
        challenges=challenges, num_rounds=num_rounds,
    )


def sumcheck_64(eval_tables, *, q, expression, challenges, num_rounds):
    """Optional 64-bit sumcheck path."""
    # TODO(student): implement when enabling 64-bit track.
    raise NotImplementedError


def sumcheck_128(eval_tables, *, q, expression, challenges, num_rounds):
    """Optional 128-bit sumcheck path."""
    # TODO(student): implement when enabling 128-bit track.
    raise NotImplementedError


def sumcheck(eval_tables, *, q, expression, challenges, num_rounds, bit_width=32):
    """Frozen dispatcher entrypoint used by the harness."""
    # Convert expression to tuple-of-tuples so it is hashable as a JIT static arg.
    expression = tuple(tuple(term) for term in expression)
    if int(bit_width) == 32:
        return sumcheck_32(
            eval_tables,
            q=q,
            expression=expression,
            challenges=challenges,
            num_rounds=num_rounds,
        )
    if int(bit_width) == 64:
        return sumcheck_64(
            eval_tables,
            q=q,
            expression=expression,
            challenges=challenges,
            num_rounds=num_rounds,
        )
    if int(bit_width) == 128:
        return sumcheck_128(
            eval_tables,
            q=q,
            expression=expression,
            challenges=challenges,
            num_rounds=num_rounds,
        )
    raise ValueError(f"Unsupported bit_width={bit_width}")
