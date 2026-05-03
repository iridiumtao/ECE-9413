"""
Modal entrypoint for running SumCheck benchmarks on a GPU instance.

Usage:
    modal run modal_run.py                          # tests + bench all tiers on T4
    modal run modal_run.py --no-tests-only          # same (explicit)
    modal run modal_run.py --bench-only             # skip tests, bench all tiers on T4
    modal run modal_run.py --tests-only             # pytest only, no bench
    modal run modal_run.py --gpu A100               # run on A100
    modal run modal_run.py --gpu H100               # run on H100

Reproduce individual experiments:
    git checkout <hash> -- student.py
    modal run modal_run.py --bench-only
    git checkout HEAD -- student.py
"""

import modal

app = modal.App("ece9413-sumcheck")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "jax[cuda12]",
        "numpy",
        "rich",
        "pytest",
        "sympy",
    )
    .add_local_dir(
        ".",
        remote_path="/root/assignment2",
        ignore=[
            "*.egg-info", "__pycache__", ".git", "*.pyc",
            "code.zip", "*.pdf", "*.pptx",
            ".env", ".env*", "*.env.*",
            "*.key", "*.pem", "*.secret", "*.crt",
        ],
    )
)


def _run_sumcheck_impl(tests: bool, bench: bool, bits: int) -> None:
    """Shared implementation used by all GPU-specific Modal functions."""
    import subprocess
    import sys

    sys.path.insert(0, "/root/assignment2")

    if tests:
        ret = subprocess.run(
            [
                sys.executable, "-m", "pytest",
                "/root/assignment2/tests",
                "--bits", str(bits), "--num-vars", "4", "-v",
            ],
            cwd="/root/assignment2",
        )
        if ret.returncode != 0:
            raise RuntimeError(f"Tests failed (exit code {ret.returncode})")

    if bench:
        import jax

        device = jax.devices()[0]
        print(f"Device: {device.platform} ({device.device_kind})")

        for n in [4, 16, 20]:
            result = subprocess.run(
                [
                    sys.executable, "-m", "tests.benchmark",
                    "--bench", "--bits", str(bits),
                    "--num-vars", str(n),
                    "--runs", "8", "--warmup", "3",
                ],
                capture_output=True, text=True,
                cwd="/root/assignment2",
            )
            print(f"\n--- num-vars {n} ---")
            print(result.stdout)
            if result.returncode != 0:
                print(result.stderr)
                raise RuntimeError(f"Benchmark failed for num-vars {n}")


@app.function(image=image, gpu="T4", timeout=900)
def run_sumcheck_t4(tests: bool = True, bench: bool = True, bits: int = 32):
    return _run_sumcheck_impl(tests=tests, bench=bench, bits=bits)


@app.function(image=image, gpu="A100", timeout=900)
def run_sumcheck_a100(tests: bool = True, bench: bool = True, bits: int = 32):
    return _run_sumcheck_impl(tests=tests, bench=bench, bits=bits)


@app.function(image=image, gpu="H100", timeout=900)
def run_sumcheck_h100(tests: bool = True, bench: bool = True, bits: int = 32):
    return _run_sumcheck_impl(tests=tests, bench=bench, bits=bits)


_GPU_FUNCTIONS = {
    "T4": run_sumcheck_t4,
    "A100": run_sumcheck_a100,
    "H100": run_sumcheck_h100,
}


@app.local_entrypoint()
def main(
    tests_only: bool = False,
    bench_only: bool = False,
    gpu: str = "T4",
    bits: int = 32,
):
    gpu_upper = gpu.upper()
    if gpu_upper not in _GPU_FUNCTIONS:
        raise ValueError(f"Unknown GPU: {gpu!r}. Choose from: {sorted(_GPU_FUNCTIONS)}")
    _GPU_FUNCTIONS[gpu_upper].remote(
        tests=not bench_only,
        bench=not tests_only,
        bits=bits,
    )
