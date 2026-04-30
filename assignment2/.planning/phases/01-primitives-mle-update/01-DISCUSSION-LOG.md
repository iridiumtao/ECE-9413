# Phase 1: Primitives + MLE Update - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-30
**Phase:** 1-Primitives + MLE Update
**Areas discussed:** uint64 promotion scope, mle_update_32 composition, mle_update_32 formula variant

---

## uint64 promotion scope

| Option | Description | Selected |
|--------|-------------|----------|
| All three + always uint32 out | mod_add_32, mod_sub_32, mod_mul_32 all promote to uint64 internally; all cast output back to uint32. Uniform pattern, no caller surprises. | ✓ |
| mul only + uint32 out | Only mod_mul_32 uses uint64. mod_add_32 relies on JAX not wrapping — risky if JAX enforces uint32 overflow semantics. | |
| All three + let output be uint64 | All promote to uint64 but don't cast back. Works but mismatches harness's uint32 tables. | |

**User's choice:** All three primitives use uint64 intermediates; always cast output to uint32.
**Notes:** mod_add_32 overflow risk was the key insight — max(a+b) ≈ 2×MAX_PRIME_32 exceeds uint32 range.

---

## mle_update_32 composition

| Option | Description | Selected |
|--------|-------------|----------|
| Compose via primitives | Call mod_sub_32, mod_mul_32, mod_add_32 inside mle_update_32. Bug isolation is clean. | |
| Inline JAX uint64 | Implement the full formula directly with jnp.uint64 casts. Avoids function call indirection. | |
| You decide | No strong preference — Claude picks the cleanest approach. | ✓ |

**User's choice:** You decide (Claude's discretion).
**Notes:** None — user deferred to Claude.

---

## mle_update_32 formula variant

| Option | Description | Selected |
|--------|-------------|----------|
| zero + r*(one−zero) — roadmap form | Follows roadmap spec. One multiply. Uses mod_sub_32 for subtraction. | ✓ |
| (1−r)*zero + r*one — two-mul form | Avoids intermediate subtraction result but requires computing (1−r) which also risks underflow. Two multiplies. | |
| You decide | No preference — Claude picks whichever is cleanest. | |

**User's choice:** `zero + r*(one−zero)` mod q, roadmap form.
**Notes:** Single multiply is cheaper; underflow handled by mod_sub_32 (already decided to use uint64 there).

---

## Claude's Discretion

- **mle_update_32 internal composition:** Claude will choose between composing via primitives or inline arithmetic. Composing via primitives is the preferred direction.

## Deferred Ideas

None — discussion stayed within phase scope.
