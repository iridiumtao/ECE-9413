# Phase 4: Incorporate Peer Optimizations and Benchmark Experimentally - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-03
**Phase:** 04-incorporate-peer-optimizations-and-benchmark-experimentally
**Areas discussed:** Which opts to port, JIT placement, Experiment granularity

---

## Which Optimizations to Port

### Barrett reduction

| Option | Description | Selected |
|--------|-------------|----------|
| Skip Barrett | CPU already gets XLA's magic-number divide optimization via naive % q. No measurable benefit, adds complexity. | ✓ |
| Port Barrett | Useful for GPU/Modal runs. Adds ~100 lines to student.py. | |

**User's choice:** Skip Barrett
**Notes:** We're benchmarking on CPU only. XLA already lowers `% constant` to a magic-number multiply on CPU.

---

### `used_vars` filtering

| Option | Description | Selected |
|--------|-------------|----------|
| Skip filtering | eval_tables always contains exactly the expression's variables per harness design. Zero practical benefit. | ✓ |
| Add used_vars filter | More defensive, matches peer style, minor code addition. | |

**User's choice:** Skip filtering

---

### t=0/t=1 shortcut

| Option | Description | Selected |
|--------|-------------|----------|
| Yes, port it | Eliminates all mle_update_32 calls for degree-1 expressions. Clean win. | ✓ |
| No, keep uniform path | Simpler mental model, no branching. | |

**User's choice:** Port the shortcut
**Notes:** Trade-off analysis applied (--analyze mode). For `a` (degree 1) at N=20, every round evaluation point is either evens or odds — 100% of mle_update_32 calls eliminated.

---

### Evens/odds pre-computation and reuse

| Option | Description | Selected |
|--------|-------------|----------|
| Yes, reuse evens/odds | Pairs naturally with t=0/t=1 shortcut. Compute once, use for both g_i eval and fold. | ✓ |
| No, keep separate slice ops | Fold still recomputes tbl[::2]/tbl[1::2] independently. | |

**User's choice:** Reuse evens/odds
**Notes:** Natural companion to the shortcut — once pre-computed for v=0/v=1 bypass, reusing in fold is free.

---

## JIT Placement

| Option | Description | Selected |
|--------|-------------|----------|
| Keep @jax.jit (current) | Entire loop compiled as one XLA program. Proven: 91 tests pass, test_jit_matches_eager passes. | ✓ |
| Remove @jax.jit (peer approach) | Python loop runs eagerly; only inner ops benefit from XLA. Would need to re-verify JIT test. | |

**User's choice:** Keep `@jax.jit`
**Notes:** Trade-off analysis applied. Peer's approach trades compile-time optimization for eager flexibility — not a win on CPU with fixed expression structures.

---

## Experiment Granularity

| Option | Description | Selected |
|--------|-------------|----------|
| One commit per change | t=0/t=1 shortcut → benchmark → record. Then evens/odds reuse → benchmark → record. Two clean git hashes. | ✓ |
| All changes together | Single commit, single benchmark. Simpler but can't isolate per-change impact. | |

**User's choice:** One commit per optimization
**Notes:** experiment.md was designed for this — each row has its own git hash. Two experiments planned.

---

## Claude's Discretion

- **D-07:** Run `uv run pytest --bits 32 --num-vars 4` as a correctness gate after each change before running the full benchmark suite.

## Deferred Ideas

- Barrett reduction — revisit if GPU/Modal benchmarks are needed
- 64-bit track — separate phase
- `jax.lax.scan` for fully-fused loop — deferred since Phase 2, still future option
