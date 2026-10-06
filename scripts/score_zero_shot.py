"""Step 2 - score the saved tile embeddings against two text prompts (CPU only, seconds).

Zero-shot means there is no training: the only thing standing in for a classifier is the pair
of sentences in `--prompt_negative` / `--prompt_positive`. Each tile's embedding is compared
with both, and a slide's score is

    mean over tiles of cosine(tile, positive prompt) - mean over tiles of cosine(tile, negative)

Positive is Responder. Higher means "looks more like the responder sentence". It is a ranking
score, not a probability, and the confusion matrix simply cuts it at 0.

This step re-reads step 1's embeddings, so trying a different pair of prompts costs one text
embedding request instead of re-embedding every tile.
"""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, roc_auc_score, roc_curve)

from cohort import CLASS_NAMES, read_cohort_csv
from mii_client import PremiumClient, resource_endpoint, write_json


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings_dir", type=str, required=True)
    parser.add_argument("--cohort_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--endpoint", type=str, required=True)
    parser.add_argument("--model", type=str, default="MedImageInsight-Premium")
    parser.add_argument("--prompt_negative", type=str, required=True)
    parser.add_argument("--prompt_positive", type=str, required=True)
    parser.add_argument("--managed_identity_client_id", type=str, default="")
    return parser.parse_args()


def log_metrics(metrics):
    """mlflow is absent from a plain laptop checkout; this step must still run there."""
    try:
        import mlflow
    except ImportError:
        print("mlflow not installed - skipping metric logging")
        return
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value):
            mlflow.log_metric(key, value)


def evaluate(y_true, scores):
    """ROC/AUC and the fixed-threshold confusion matrix. `scores` are margins, not probabilities."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    if y_true.shape != scores.shape or not len(y_true):
        raise ValueError("Labels and scores must be equal-sized and nonempty.")
    if set(y_true.tolist()) != {0, 1} or not np.isfinite(scores).all():
        raise ValueError("Both classes and finite scores are required.")

    predicted = (scores > 0.0).astype(int)
    matrix = confusion_matrix(y_true, predicted, labels=[0, 1])
    (true_negative, false_positive), (false_negative, true_positive) = matrix
    fpr, tpr, thresholds = roc_curve(y_true, scores, drop_intermediate=False)

    metrics = {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(true_positive / (true_positive + false_negative)),
        "specificity": float(true_negative / (true_negative + false_positive)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "n_images": int(len(y_true)),
        "positive_count": int(y_true.sum()),
        # Predicting the majority class for everything already scores this much.
        "majority_baseline_accuracy": float(max(y_true.mean(), 1 - y_true.mean())),
        "confusion_matrix": matrix.tolist(),
        "class_order": CLASS_NAMES,
        "positive_class": "Responder",
        "threshold": 0.0,
        "score": "mean_cosine_positive_prompt_minus_mean_cosine_negative_prompt",
        "training": False,
        "probability": False,
    }
    curve = {"fpr": fpr.tolist(), "tpr": tpr.tolist(),
             "thresholds": [float(value) if np.isfinite(value) else None for value in thresholds]}
    return metrics, curve


def write_predictions(path, rows):
    fields = ["image_id", "file_name", "label", "y_true", "predicted_label", "score",
              "cosine_non_responder", "cosine_responder", "tile_count"]
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    from azure.identity import ManagedIdentityCredential

    args = parse_args()
    for name, value in vars(args).items():
        print(f"{name}: {value}")
    if args.prompt_negative.strip() == args.prompt_positive.strip():
        raise ValueError("The two class prompts must differ.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir = Path(args.embeddings_dir)

    rows = read_cohort_csv(args.cohort_csv)
    images = json.loads((embeddings_dir / "images.json").read_text(encoding="utf-8"))["images"]
    by_name = {row["file_name"].casefold(): row for row in rows}
    if {image["file_name"].casefold() for image in images} != set(by_name):
        raise ValueError("Step 1 embedded a different set of images than the cohort file lists.")

    client_id = args.managed_identity_client_id or os.getenv("DEFAULT_IDENTITY_CLIENT_ID")
    credential = ManagedIdentityCredential(client_id=client_id or None)
    client = PremiumClient(resource_endpoint(args.endpoint), args.model, credential)
    try:
        # Prompt order must match CLASS_NAMES: index 0 negative, index 1 positive.
        text_vectors = client.embed("texts", [args.prompt_negative, args.prompt_positive])
    finally:
        client.close()
    np.save(output_dir / "text_embeddings.npy", text_vectors)

    predictions = []
    for image in images:
        payload = np.load(embeddings_dir / "embeddings" / f"{image['image_id']}.npz")
        tile_vectors = payload["image_embeddings"]
        if tile_vectors.shape != (image["tile_count"], 1024):
            raise ValueError(f"{image['image_id']}: embedding shape does not match tiles.json.")
        # Both sides are unit-normalised, so the dot product is the cosine similarity.
        similarities = np.clip(tile_vectors @ text_vectors.T, -1.0, 1.0)
        mean_negative, mean_positive = similarities.mean(axis=0)
        margin = float(mean_positive - mean_negative)
        label = by_name[image["file_name"].casefold()]["label"]
        predictions.append({
            "image_id": image["image_id"],
            "file_name": image["file_name"],
            "label": label,
            "y_true": CLASS_NAMES.index(label),
            "predicted_label": CLASS_NAMES[int(margin > 0)],
            "score": margin,
            "cosine_non_responder": float(mean_negative),
            "cosine_responder": float(mean_positive),
            "tile_count": int(image["tile_count"]),
        })

    write_predictions(output_dir / "predictions.csv", predictions)
    metrics, curve = evaluate([row["y_true"] for row in predictions],
                              [row["score"] for row in predictions])
    metrics.update({
        "model": args.model,
        "prompts": {"Non-Responder": args.prompt_negative, "Responder": args.prompt_positive},
        "n_tiles": sum(row["tile_count"] for row in predictions),
        "job_name": os.getenv("AZUREML_RUN_ID"),
        "clinical_use": False,
    })
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "roc_curve.json", curve)

    log_metrics(metrics)
    print(json.dumps({key: value for key, value in metrics.items()
                      if key != "prompts"}, indent=2))


if __name__ == "__main__":
    main()
