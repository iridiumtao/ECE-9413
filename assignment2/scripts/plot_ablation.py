"""Generate the ablation bar chart for report §5.4.1.

Per D-07, shows vars20 median latency for the four base polynomials
across baseline / Exp01 / Exp03 git-tag states.

Source numbers: experiment.md (baseline + Exp01 + Exp03 sections).
Output: report/ablation_chart.png.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no DISPLAY required
import matplotlib.pyplot as plt
import numpy as np


EXPRESSIONS = ["a", "a*b", "a*b + c", "a*b*c"]

# Median latency in milliseconds at num-vars 20.
# Source: experiment.md §Baseline → Results — num-vars 20
BASELINE = {
    "a": 2.098,
    "a*b": 6.571,
    "a*b + c": 15.693,
    "a*b*c": 9.968,
}

# Source: experiment.md §Experiment 01 → Results — num-vars 20
EXP01 = {
    "a": 1.145,
    "a*b": 3.864,
    "a*b + c": 5.241,
    "a*b*c": 6.508,
}

# Source: experiment.md §Experiment 03 → Results — num-vars 20
EXP03 = {
    "a": 1.071,
    "a*b": 2.887,
    "a*b + c": 4.848,
    "a*b*c": 5.050,
}


def main() -> None:
    x = np.arange(len(EXPRESSIONS))
    width = 0.27

    fig, ax = plt.subplots(figsize=(8.5, 5))
    bars_baseline = ax.bar(
        x - width, [BASELINE[e] for e in EXPRESSIONS], width, label="Baseline"
    )
    bars_exp01 = ax.bar(
        x, [EXP01[e] for e in EXPRESSIONS], width, label="Exp01 (t=0/t=1 shortcut)"
    )
    bars_exp03 = ax.bar(
        x + width,
        [EXP03[e] for e in EXPRESSIONS],
        width,
        label="Exp03 (filter + no-mul)",
    )

    # Annotate each bar with its numeric value above the top edge.
    for bars in (bars_baseline, bars_exp01, bars_exp03):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xlabel("Expression")
    ax.set_ylabel("Median latency (ms)")
    ax.set_title("Ablation: vars20 median latency across optimization states")
    ax.set_xticks(x)
    ax.set_xticklabels(EXPRESSIONS)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    out_path = Path("report") / "ablation_chart.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
