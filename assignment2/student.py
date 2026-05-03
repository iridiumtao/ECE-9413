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


@functools.partial(jax.jit, static_argnames=("expression", "num_rounds"))
def sumcheck_32(eval_tables, *, q, expression, challenges, num_rounds):
    """Compulsory 32-bit sumcheck path."""
    degree = max(len(term) for term in expression)

    tables = {var: jnp.asarray(table, dtype=jnp.uint32)
              for var, table in eval_tables.items()}

    round_evals_list = []

    for i in range(num_rounds):
        g_i_evals = []

        for v in range(degree + 1):
            v_scalar = jnp.asarray(v, dtype=jnp.uint32)
            g_i_at_v = jnp.asarray(0, dtype=jnp.uint64)

            for term in expression:
                first_var = term[0]
                term_product = mle_update_32(
                    tables[first_var][::2],
                    tables[first_var][1::2],
                    v_scalar,
                    q=q,
                )

                for var in term[1:]:
                    factor = mle_update_32(
                        tables[var][::2],
                        tables[var][1::2],
                        v_scalar,
                        q=q,
                    )
                    term_product = mod_mul_32(term_product, factor, q)

                # Safe uint64 sum: up to 2^20 entries, each < q < 2^32, sum < 2^52 < 2^64.
                term_sum = jnp.sum(term_product.astype(jnp.uint64)) % jnp.asarray(q, dtype=jnp.uint64)
                g_i_at_v = (g_i_at_v + term_sum) % jnp.asarray(q, dtype=jnp.uint64)

            g_i_evals.append(jnp.asarray(g_i_at_v, dtype=jnp.uint32))

        round_evals_list.append(jnp.stack(g_i_evals))

        # Fold all tables using challenges[i]; skip fold on the last round.
        if i < len(challenges):
            r = challenges[i]
            tables = {
                var: mle_update_32(tbl[::2], tbl[1::2], r, q=q)
                for var, tbl in tables.items()
            }

    round_evals = jnp.stack(round_evals_list)

    # claim0 = g_0(0) + g_0(1) mod q (D-04).
    claim0 = mod_add_32(round_evals_list[0][0], round_evals_list[0][1], q)

    return claim0, round_evals


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
