from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
BLUE = "#2B6CB0"
ORANGE = "#D97706"
GRAY = "#6B7280"
DARK = "#1F2937"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def finish(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def learning_curve() -> None:
    rows = read_csv(ROOT / "results/statistics/learning_curve.csv")
    order = ["0720k", "2000k", "full"]
    labels = ["0.72M", "2.0M", "Full"]
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["scale"]].append(float(row["verb_class_accuracy"]) * 100)
    means = [np.mean(grouped[key]) for key in order]
    sds = [np.std(grouped[key], ddof=1) for key in order]
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    x = np.arange(len(order))
    for seed in sorted({row["seed"] for row in rows}):
        values = [float(next(row for row in rows if row["scale"] == key and row["seed"] == seed)["verb_class_accuracy"]) * 100 for key in order]
        ax.plot(x, values, color=GRAY, alpha=.45, linewidth=1, marker="o", markersize=3)
    ax.errorbar(x, means, yerr=sds, color=BLUE, linewidth=2.2, marker="o", markersize=6, capsize=4, label="Mean ± seed SD")
    ax.axhline(20, color=DARK, linestyle="--", linewidth=1, label="Chance")
    ax.set_xticks(x, labels)
    ax.set_xlabel("Training scale")
    ax.set_ylabel("Verb-class accuracy (%)")
    ax.set_ylim(15, 78)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    finish(fig, "learning_curve")


def baseline_gradient() -> None:
    rows = read_csv(ROOT / "results/statistics/baselines.csv")
    random_init_rows = read_csv(ROOT / "results/statistics/random_init.csv")
    random_values = [float(row["verb_class_accuracy"]) * 100 for row in random_init_rows]
    deterministic = {row["baseline"]: float(row["accuracy"]) * 100 for row in rows if row["seed"] == "deterministic"}
    model_rows = read_csv(ROOT / "results/statistics/learning_curve.csv")
    full = np.mean([float(row["verb_class_accuracy"]) * 100 for row in model_rows if row["scale"] == "full"])
    labels = ["Random init", "AUX only", "POS frame", "Lexical context", "Full MLM"]
    values = [np.mean(random_values), deterministic["aux_only_nb"], deterministic["pos_frame_count"], deterministic["lexical_cooccurrence_nb"], full]
    colors = [GRAY, ORANGE, ORANGE, ORANGE, BLUE]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, height=.62)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Verb-class accuracy (%)")
    ax.set_xlim(0, 80)
    ax.axvline(20, color=DARK, linestyle="--", linewidth=1)
    for yi, value in zip(y, values):
        ax.text(value + 1, yi, f"{value:.1f}", va="center", fontsize=9)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    finish(fig, "baseline_gradient")


def nonce_effects() -> None:
    rows = [row for row in read_csv(ROOT / "results/nonce/hierarchical_bootstrap.csv") if row["metric"] == "verb_class_accuracy"]
    names = {
        "zero_shot": "Zero-shot",
        "aux_ablated_exposure": "AUX-identity ablation",
        "aux_shuffled_exposure": "AUX-identity shuffle",
        "matched_function_ablated_exposure": "Matched-function ablation",
    }
    rows.sort(key=lambda row: ["zero_shot", "aux_ablated_exposure", "aux_shuffled_exposure", "matched_function_ablated_exposure"].index(row["comparison"]))
    effects = [-float(row["delta_comparison_minus_reference"]) * 100 for row in rows]
    lows = [-float(row["ci95_high"]) * 100 for row in rows]
    highs = [-float(row["ci95_low"]) * 100 for row in rows]
    labels = [names[row["comparison"]] for row in rows]
    fig, ax = plt.subplots(figsize=(5.3, 3.2))
    y = np.arange(len(rows))
    xerr = np.vstack([np.array(effects) - np.array(lows), np.array(highs) - np.array(effects)])
    ax.errorbar(effects, y, xerr=xerr, fmt="o", color=BLUE, ecolor=BLUE, capsize=4, markersize=6)
    ax.axvline(0, color=DARK, linewidth=1, linestyle="--")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Intact exposure advantage (accuracy points)")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    finish(fig, "nonce_effects")


def cue_behavior() -> None:
    rows = read_csv(ROOT / "results/statistics/item_cue_features.csv")
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    for seed, color in zip(sorted({row["seed"] for row in rows}), [BLUE, ORANGE, GRAY]):
        subset = [row for row in rows if row["seed"] == seed]
        x = np.array([float(row["aux_log_odds"]) for row in subset])
        y = np.array([float(row["model_class_preference_score"]) for row in subset])
        chunks = np.array_split(np.argsort(x), 10)
        bx = np.array([x[index].mean() for index in chunks])
        by = np.array([y[index].mean() for index in chunks])
        ax.plot(bx, by, marker="o", markersize=4, linewidth=1.5, color=color, label=f"Seed {seed}")
    ax.axhline(0, color=DARK, linewidth=.8, alpha=.6)
    ax.set_xlabel("AUX log-odds score for target class (decile means)")
    ax.set_ylabel("Mean class preference score")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    finish(fig, "cue_behavior")


def main() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "pdf.fonttype": 42,
    })
    learning_curve()
    baseline_gradient()
    nonce_effects()
    cue_behavior()
    print(f"Wrote figures to {OUT}")


if __name__ == "__main__":
    main()
