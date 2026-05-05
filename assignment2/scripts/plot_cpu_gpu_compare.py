"""CPU vs GPU comparison chart — vars20 median latency for Baseline, Exp03, Exp04.

6 bars per expression group (CPU solid / GPU hatched, one shade per state).
Log-scale y-axis handles the 35× range between CPU Baseline and GPU Exp04.
Output: report/cpu_gpu_compare.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXPRESSIONS = ["a", "a*b", "a*b + c", "a*b*c"]

# CPU medians — experiment.md (Apple Silicon)
CPU_BASELINE = {"a": 2.098, "a*b": 6.571, "a*b + c": 15.693, "a*b*c":  9.968}
CPU_EXP03    = {"a": 1.071, "a*b": 2.887, "a*b + c":  4.848, "a*b*c":  5.050}
CPU_EXP04    = {"a": 1.021, "a*b": 3.476, "a*b + c":  4.630, "a*b*c":  6.766}

# GPU medians — experiment_gpu.md (Modal T4, our run)
GPU_BASELINE = {"a": 0.464, "a*b": 0.822, "a*b + c":  1.326, "a*b*c":  1.344}
GPU_EXP03    = {"a": 0.465, "a*b": 0.684, "a*b + c":  1.187, "a*b*c":  1.136}
GPU_EXP04    = {"a": 0.450, "a*b": 0.729, "a*b + c":  1.335, "a*b*c":  1.107}

# One hue per optimization state; solid = CPU, hatched = GPU.
_COLORS = {
    "Baseline": "#4C72B0",
    "Exp03":    "#DD8452",
    "Exp04":    "#55A868",
}

_DATASETS = [
    (CPU_BASELINE, "CPU — Baseline", "Baseline", False),
    (GPU_BASELINE, "GPU — Baseline", "Baseline", True),
    (CPU_EXP03,    "CPU — Exp03",    "Exp03",    False),
    (GPU_EXP03,    "GPU — Exp03",    "Exp03",    True),
    (CPU_EXP04,    "CPU — Exp04",    "Exp04",    False),
    (GPU_EXP04,    "GPU — Exp04",    "Exp04",    True),
]

# Bar offsets: pairs [CPU_base, GPU_base] | [CPU_exp03, GPU_exp03] | [CPU_exp04, GPU_exp04]
# with a small visual gap between state pairs.
_OFFSETS = [-2.6, -1.7, -0.55, 0.35, 1.5, 2.4]
_WIDTH   = 0.13


def _annotate(ax, bars):
    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            f"{h:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=5.5,
            rotation=90,
        )


def plot_cpu_gpu_compare() -> Path:
    x = np.arange(len(EXPRESSIONS))

    fig, ax = plt.subplots(figsize=(12, 6))

    for (data, label, state, is_gpu), off in zip(_DATASETS, _OFFSETS):
        color = _COLORS[state]
        bars = ax.bar(
            x + off * _WIDTH,
            [data[e] for e in EXPRESSIONS],
            _WIDTH,
            label=label,
            color=color,
            hatch="////" if is_gpu else "",
            edgecolor="white",
            linewidth=0.4,
        )
        _annotate(ax, bars)

    ax.set_yscale("log")
    ax.set_xlabel("Expression")
    ax.set_ylabel("Median latency (ms, log scale)")
    ax.set_title("CPU vs GPU — vars20 median latency across Baseline / Exp03 / Exp04\n"
                 "Solid = CPU (Apple Silicon)   Hatched = GPU (Modal T4)")
    ax.set_xticks(x)
    ax.set_xticklabels(EXPRESSIONS)
    ax.legend(fontsize=8, ncol=3, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    out = Path("report") / "cpu_gpu_compare.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    print(f"wrote {plot_cpu_gpu_compare()}")


if __name__ == "__main__":
    main()
