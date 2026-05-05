# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Assignment Goal

Implement the **prover side** of the SumCheck protocol over finite fields using JAX. The only file to edit is `student.py`.

## Setup

```bash
bash scripts/setup.sh        # installs dependencies via uv (auto-detects CUDA)
export JAX_PLATFORMS=cpu     # or gpu / tpu
```

If another virtualenv is active: `uv run --active ...`

## Running Tests

Staged correctness checks (required order — start small):
```bash
uv run pytest --bits 32 --num-vars 4
uv run pytest --bits 32 --num-vars 16
uv run pytest --bits 32 --num-vars 20
```

Focus on a single expression or case:
```bash
uv run pytest --bits 32 --num-vars 16 --expr-id "a*b + c"
uv run pytest --bits 32 --num-vars 16 --expr-id "a*b + c" --case-id v16_case32_2
```

Optional tracks:
```bash
uv run pytest --all-64 --enable-core64            # 64-bit core
uv run pytest --bits 32 --num-vars 20 --enable-challenge32  # advanced polynomials
uv run pytest -q tests/test_sumcheck.py::test_mod_add_128bit_edge_cases --include-128
```

## Benchmarking

```bash
uv run python -m tests.benchmark --bench --bits 32 --num-vars 4  --runs 8 --warmup 3
uv run python -m tests.benchmark --bench --bits 32 --num-vars 16 --runs 8 --warmup 3
uv run python -m tests.benchmark --bench --bits 32 --num-vars 20 --runs 8 --warmup 3
uv run python -m tests.benchmark --bench --bits 32 --num-vars 16 --expr "a*b" --runs 8 --warmup 3
```

## Submission

```bash
bash scripts/make_submission.sh   # runs required tests, produces code.zip
```

## Architecture

```
student.py          ← only file to modify
provided.py         ← read-only constants (EXPRESSIONS, VARIABLE_NAMES) + debug helper
sumcheck_utils.py   ← expression normalization utilities used by tests
tests/
  conftest.py       ← pytest CLI options/fixtures (--bits, --num-vars, --expr-id, etc.)
  test_sumcheck.py  ← public test suite
  benchmark.py      ← benchmark harness
  data_loader.py    ← discovers cases from tests/data/*/*_meta.json
  case_utils.py     ← loads tables and expected outputs
  case_selection.py ← filters cases by CLI options
scripts/
  custom_cases.py   ← generate/check/bench custom cases (independent of pytest)
  debug_round_trace.py
  setup.sh / make_submission.sh
```

## student.py API Contract

`sumcheck(eval_tables, *, q, expression, challenges, num_rounds, bit_width=32)` must return `(claim0, round_evals)` where **both are JAX arrays**.

- `challenges` has length `num_rounds - 1` — the final verifier-only challenge is excluded.
- `claim0` is the initial sum over the Boolean hypercube (scalar JAX array).
- `round_evals` is a 2D JAX array of shape `(num_rounds, degree+1)` — each row is the evaluations of the round polynomial `g_i` at points `0, 1, ..., d`.

Compulsory functions to implement (32-bit):
- `mod_add_32`, `mod_sub_32`, `mod_mul_32` — modular arithmetic on `uint32` inputs
- `mle_update_32(zero_eval, one_eval, target_eval, *, q)` — MLE table fold for one challenge
- `sumcheck_32(eval_tables, *, q, expression, challenges, num_rounds)` — full prover loop

## Expression Format

`list[list[str]]` — outer list is additive terms, inner list is multiplicative factors:
- `a*b` → `[["a", "b"]]`
- `a*b + c` → `[["a", "b"], ["c"]]`

Variable names come from `eval_tables` (a dict of `str → jax.Array`).

## Key Protocol Invariant

Rounds must be serialized — each round depends on the previous challenge. You cannot parallelize across rounds. Within a round, the table-fold computation across the remaining `2^(n-i)` entries is parallelizable.

## Debugging vars4 Cases

```python
import provided
trace = provided.expression_round_trace(2, case_id="v4_case32_0")
print(trace["expected_round_evals"])
for i, tables in enumerate(trace["round_tables"]):
    print(f"round {i}:", tables)
```

Only vars4 cases include round tables; vars16/vars20 cases omit them.

## Custom Cases

```bash
uv run python scripts/custom_cases.py generate --expression "a*b + c" --num-vars 6 --bits 32
uv run python scripts/custom_cases.py check --case-id custom_v6_32b_a_b_c_s0
uv run python scripts/custom_cases.py bench  --case-id custom_v6_32b_a_b_c_s0 --runs 8 --warmup 3
```

See `custom_cases.md` for full options.

## JAX Notes

- `jax.config.update("jax_enable_x64", True)` is already set in `student.py` and tests.
- The first JIT compile is slow; benchmark warmup runs account for this.
- `jax.block_until_ready()` is called by the benchmark harness — do not add it inside `sumcheck`.
