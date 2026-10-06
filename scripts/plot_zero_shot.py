"""Plot a step-2 result: ROC, the confusion matrix at score 0, and the score separation.

    python scripts/plot_zero_shot.py <evaluation output folder> [--out zero_shot_report.png]

Reads `metrics.json`, `predictions.csv` and `roc_curve.json` as written by score_zero_shot.py,
so it also works on a folder downloaded from an earlier job - which is how prompt variants get
compared without re-running anything.

`report(result_dir)` returns the figure, for use from the notebook.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CLASSES = ("Non-Responder", "Responder")


def plot_roc(axis, curve, metrics):
    auc = metrics["roc_auc"]
    axis.plot([0, 1], [0, 1], "--", color="0.7", linewidth=1, label="Random ranking (0.5)")
    axis.step(curve["fpr"], curve["tpr"], where="post", color="#007f78", linewidth=2,
              label=f"Zero-shot AUC = {auc:.3f}")
    positives = metrics["positive_count"]
    axis.set_title(f"Image-level ROC  (n={metrics['n_images']}: {positives} Responder / "
                   f"{metrics['n_images'] - positives} Non-Responder)", fontsize=10)
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_xlim(-0.02, 1.02)
    axis.set_ylim(-0.02, 1.02)
    axis.set_aspect("equal")
    axis.legend(loc="lower right", fontsize=8)
    if auc < 0.5:
        # Worth saying out loud: the prompts rank the two classes the wrong way round.
        axis.text(0.03, 0.95, "AUC < 0.5:\nranking is inverted", transform=axis.transAxes,
                  va="top", fontsize=8, color="#b22222")


def plot_confusion(axis, metrics):
    matrix = np.asarray(metrics["confusion_matrix"])
    axis.imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.max())
    for (row, column), count in np.ndenumerate(matrix):
        axis.text(column, row, str(count), ha="center", va="center", fontsize=16,
                  color="white" if count > matrix.max() / 2 else "black")
    axis.set_xticks([0, 1], CLASSES, fontsize=8)
    axis.set_yticks([0, 1], CLASSES, rotation=90, va="center", fontsize=8)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title(f"Confusion matrix at score > 0\n"
                   f"sens={metrics['recall']:.3f} spec={metrics['specificity']:.3f}   |   "
                   f"all-Responder accuracy = {metrics['majority_baseline_accuracy']:.3f}",
                   fontsize=9)


def plot_scores(axis, frame):
    jitter = np.random.default_rng(0)
    for value, colour in ((0, "#d62728"), (1, "#2ca02c")):
        selected = frame.loc[frame["y_true"] == value, "score"].to_numpy()
        axis.scatter(selected,
                     np.full(len(selected), value) + jitter.uniform(-0.12, 0.12, len(selected)),
                     color=colour, s=45, alpha=0.8, label=f"{CLASSES[value]} (n={len(selected)})")
    axis.axvline(0.0, color="0.3", linestyle="--", linewidth=1.2)
    axis.set_yticks([0, 1], CLASSES, fontsize=8)
    axis.set_ylim(-0.5, 1.5)
    axis.set_xlabel("cosine(Responder prompt) - cosine(Non-Responder prompt), averaged over tiles")
    axis.set_title("Score separation  (dashed = the threshold used above)", fontsize=10)
    axis.legend(fontsize=8, loc="lower right")


def report(result_dir):
    """Build the three-panel figure from a step-2 output folder."""
    result_dir = Path(result_dir)
    metrics = json.loads((result_dir / "metrics.json").read_text(encoding="utf-8"))
    curve = json.loads((result_dir / "roc_curve.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(result_dir / "predictions.csv", float_precision="round_trip")

    figure, axes = plt.subplots(1, 3, figsize=(16, 5))
    plot_roc(axes[0], curve, metrics)
    plot_confusion(axes[1], metrics)
    plot_scores(axes[2], frame)
    figure.suptitle(
        f"MedImageInsight zero-shot - {metrics['model']}, {metrics['n_images']} images, "
        f"{metrics['n_tiles']} tiles, no training\n"
        f"Responder prompt: \"{metrics['prompts']['Responder']}\"", fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path, help="step-2 output folder")
    parser.add_argument("--out", type=Path, default=Path("zero_shot_report.png"))
    args = parser.parse_args()

    figure = report(args.result_dir)
    figure.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")

    metrics = json.loads((args.result_dir / "metrics.json").read_text(encoding="utf-8"))
    print(f"AUC={metrics['roc_auc']:.4f}  accuracy={metrics['accuracy']:.3f}  "
          f"balanced={metrics['balanced_accuracy']:.3f}  n={metrics['n_images']}")


if __name__ == "__main__":
    main()
