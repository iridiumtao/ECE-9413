"""Generate ablation bar charts for report §5.4.1.

Per D-07, shows vars20 median latency for the four base polynomials
across optimization states.  Two charts are produced:
  - report/ablation_chart_cpu.png  (CPU: Baseline / Exp01 / Exp02 / Exp03)
  - report/ablation_chart_gpu.png  (GPU: Baseline / Exp01 / Exp02)

Source numbers: experiment.md (CPU) and experiment_gpu.md (GPU).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no DISPLAY required
import matplotlib.pyplot as plt
import numpy as np


EXPRESSIONS = ["a", "a*b", "a*b + c", "a*b*c"]

# ── CPU results (experiment.md) ──────────────────────────────────────────────

CPU_BASELINE = {"a": 2.098, "a*b": 6.571, "a*b + c": 15.693, "a*b*c": 9.968}
CPU_EXP01    = {"a": 1.145, "a*b": 3.864, "a*b + c":  5.241, "a*b*c": 6.508}
CPU_EXP02    = {"a": 1.091, "a*b": 3.800, "a*b + c":  5.538, "a*b*c": 6.667}
CPU_EXP03    = {"a": 1.071, "a*b": 2.887, "a*b + c":  4.848, "a*b*c": 5.050}
CPU_EXP04    = {"a": 1.021, "a*b": 3.476, "a*b + c":  4.630, "a*b*c": 6.766}

# ── GPU results (experiment_gpu.md, Modal T4) ────────────────────────────────

GPU_BASELINE = {"a": 0.464, "a*b": 0.822, "a*b + c": 1.326, "a*b*c": 1.344}
GPU_EXP01    = {"a": 0.464, "a*b": 0.712, "a*b + c": 1.176, "a*b*c": 1.258}
GPU_EXP02    = {"a": 0.561, "a*b": 0.823, "a*b + c": 1.380, "a*b*c": 1.408}
GPU_EXP03    = {"a": 0.465, "a*b": 0.684, "a*b + c": 1.187, "a*b*c": 1.136}
GPU_EXP04    = {"a": 0.450, "a*b": 0.729, "a*b + c": 1.335, "a*b*c": 1.107}


def _annotate_bars(ax, bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
        )


def plot_cpu() -> Path:
    x = np.arange(len(EXPRESSIONS))
    width = 0.15

    fig, ax = plt.subplots(figsize=(10, 5))
    offsets = [-2, -1, 0, 1, 2]
    datasets = [
        (CPU_BASELINE, "Baseline"),
        (CPU_EXP01,    "Exp01 (t=0/t=1 shortcut)"),
        (CPU_EXP02,    "Exp02 (evens/odds reuse)"),
        (CPU_EXP03,    "Exp03 (filter + no-mul)"),
        (CPU_EXP04,    "Exp04 (diff-share + term-by-term)"),
    ]
    for (data, label), off in zip(datasets, offsets):
        bars = ax.bar(x + off * width, [data[e] for e in EXPRESSIONS], width, label=label)
        _annotate_bars(ax, bars)

    ax.set_xlabel("Expression")
    ax.set_ylabel("Median latency (ms)")
    ax.set_title("Ablation — CPU, vars20 median latency across optimization states")
    ax.set_xticks(x)
    ax.set_xticklabels(EXPRESSIONS)
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    out = Path("report") / "ablation_chart_cpu.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_gpu() -> Path:
    x = np.arange(len(EXPRESSIONS))
    width = 0.15

    fig, ax = plt.subplots(figsize=(10, 5))
    offsets = [-2, -1, 0, 1, 2]
    datasets = [
        (GPU_BASELINE, "Baseline"),
        (GPU_EXP01,    "Exp01 (t=0/t=1 shortcut)"),
        (GPU_EXP02,    "Exp02 (evens/odds reuse — regresses on GPU)"),
        (GPU_EXP03,    "Exp03 (filter + no-mul)"),
        (GPU_EXP04,    "Exp04 (diff-share + term-by-term + Barrett)"),
    ]
    for (data, label), off in zip(datasets, offsets):
        bars = ax.bar(x + off * width, [data[e] for e in EXPRESSIONS], width, label=label)
        _annotate_bars(ax, bars)

    ax.set_xlabel("Expression")
    ax.set_ylabel("Median latency (ms)")
    ax.set_title("Ablation — GPU (T4), vars20 median latency across optimization states")
    ax.set_xticks(x)
    ax.set_xticklabels(EXPRESSIONS)
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    out = Path("report") / "ablation_chart_gpu.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    print(f"wrote {plot_cpu()}")
    print(f"wrote {plot_gpu()}")


if __name__ == "__main__":
    main()
