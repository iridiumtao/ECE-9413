# Requirements: ECE-9413 Assignment 2 — SumCheck Prover

**Defined:** 2026-04-30
**Core Value:** Correctly produce `(claim0, round_evals)` that passes all required 32-bit base-polynomial tests on vars4, vars16, and vars20.

## v1 Requirements

### Primitives

- [ ] **PRIM-01**: `mod_add_32(a, b, q)` returns `(a + b) % q` correctly for any uint32 inputs and 32-bit prime `q`
- [ ] **PRIM-02**: `mod_sub_32(a, b, q)` returns `(a - b) % q` correctly (no negative underflow)
- [ ] **PRIM-03**: `mod_mul_32(a, b, q)` returns `(a * b) % q` correctly (intermediate must not overflow uint32)

### MLE Update

- [ ] **MLE-01**: `mle_update_32(zero_eval, one_eval, target_eval, *, q)` computes linear interpolation to fold one Boolean variable — produces a table half the length of the input for a given challenge `r`

### SumCheck Prover

- [ ] **SC-01**: `sumcheck_32` computes `claim0` — the initial sum over the Boolean hypercube for the given expression
- [ ] **SC-02**: `sumcheck_32` computes per-round univariate evaluations `g_i(0)` and `g_i(1)` for each round
- [ ] **SC-03**: Round tables are correctly folded after each challenge is applied (`mle_update_32` used per round)
- [ ] **SC-04**: `sumcheck_32` returns `(claim0, round_evals)` both as JAX arrays, with `round_evals` shape `(num_rounds, 2)` or equivalent

### Correctness Verification

- [ ] **CORR-01**: All base expressions (`a`, `a*b`, `a*b+c`, `a*b*c`) pass on `vars4` (5 cases, 32-bit)
- [ ] **CORR-02**: All base expressions pass on `vars16` (5 cases, 32-bit)
- [ ] **CORR-03**: All base expressions pass on `vars20` (5 cases, 32-bit)

### Benchmarks

- [ ] **BENCH-01**: Benchmark results produced for vars4 (8 runs, 3 warmup)
- [ ] **BENCH-02**: Benchmark results produced for vars16 (8 runs, 3 warmup)
- [ ] **BENCH-03**: Benchmark results produced for vars20 (8 runs, 3 warmup)

### Submission

- [ ] **SUB-01**: `bash scripts/make_submission.sh` runs cleanly and produces `code.zip`

## v2 Requirements

### Extra Credit — Advanced Polynomials

- **ADV-01**: `a*a*b*b*c` passes on vars4/16/20 (32-bit, `--enable-challenge32`)
- **ADV-02**: `a*b*c + d*e` passes on vars4/16/20 (32-bit, `--enable-challenge32`)
- **ADV-03**: `a*b*c*g + d*e*g` passes on vars4/16/20 (32-bit, `--enable-challenge32`)

### Extra Credit — 64-bit Track

- **BIT64-01**: `mod_add_64 / mod_sub_64 / mod_mul_64` implemented
- **BIT64-02**: `mle_update_64` implemented
- **BIT64-03**: `sumcheck_64` passes all `--enable-core64` tests

### Extra Credit — Performance

- **PERF-01**: Hashing between rounds implemented and benchmarked
- **PERF-02**: GPU/TPU benchmark results included in report

## Out of Scope

| Feature | Reason |
|---------|--------|
| 128-bit track | Optional; not in required scope; much lower priority than extra credit |
| Interactive verifier integration | Assignment only requires prover side |
| Custom polynomial parsing | Expression format is fixed by harness (`list[list[str]]`) |
| Multi-round parallelism | Explicitly forbidden by assignment spec |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| PRIM-01 | Phase 1 | Pending |
| PRIM-02 | Phase 1 | Pending |
| PRIM-03 | Phase 1 | Pending |
| MLE-01 | Phase 1 | Pending |
| SC-01 | Phase 2 | Pending |
| SC-02 | Phase 2 | Pending |
| SC-03 | Phase 2 | Pending |
| SC-04 | Phase 2 | Pending |
| CORR-01 | Phase 2 | Pending |
| CORR-02 | Phase 2 | Pending |
| CORR-03 | Phase 2 | Pending |
| BENCH-01 | Phase 3 | Pending |
| BENCH-02 | Phase 3 | Pending |
| BENCH-03 | Phase 3 | Pending |
| SUB-01 | Phase 3 | Pending |

**Coverage:**
- v1 requirements: 15 total
- Mapped to phases: 15
- Unmapped: 0 ✓

---
*Requirements defined: 2026-04-30*
*Last updated: 2026-04-30 after initial definition*
