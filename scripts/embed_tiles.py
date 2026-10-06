"""Step 1 - tile each slide and turn every tile into a MedImageInsight embedding (CPU only).

This is the step that costs money: one Foundry request per batch of tiles. It is also the
step whose output can be reused, so it deliberately knows **nothing** about prompts, classes
or the ROC. It reads the cohort file only for the list of file names to process; the `label`
column is not opened here.

Output:
    embeddings/<image_id>.npz   image_embeddings [n_tiles, 1024], unit-normalised
    tiles.json                  per-slide magnification and tile coordinates
    images.json                 image_id -> file name, plus the API call log
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from cohort import read_cohort_csv
from mii_client import PremiumClient, png_data_url, resource_endpoint, write_json
from tile_wsi import WSI_SUFFIXES, tile_slide


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--cohort_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--endpoint", type=str, required=True)
    parser.add_argument("--model", type=str, default="MedImageInsight-Premium")
    parser.add_argument("--target_mpp", type=float, default=0.5)
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--occupancy_threshold", type=float, default=0.20)
    parser.add_argument("--max_tiles", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=8)
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
        mlflow.log_metric(key, value)


def resolve_slides(rows, images_dir):
    """Map each cohort file name to exactly one slide file. Never skip a missing one."""
    found = {}
    for path in Path(images_dir).rglob("*"):
        if path.is_file() and path.suffix.lower() in WSI_SUFFIXES:
            found.setdefault(path.name.casefold(), []).append(path)
    matches = [(row, found.get(row["file_name"].casefold(), [])) for row in rows]
    missing = [row["file_name"] for row, paths in matches if len(paths) != 1]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} cohort image(s) have no unique slide file: {missing}. "
            "Fix the data asset or the cohort list; images are not dropped silently."
        )
    return [(row, paths[0]) for row, paths in matches]


def embed_tiles(client, tiles, batch_size):
    """Embed a slide's tiles, a few per request. Returns [n_tiles, 1024]."""
    batches = []
    for start in range(0, len(tiles), batch_size):
        batches.append(client.embed("images", [png_data_url(tile)
                                               for tile in tiles[start:start + batch_size]]))
    vectors = np.concatenate(batches, axis=0)
    if vectors.shape != (len(tiles), 1024):
        raise ValueError(f"Expected {(len(tiles), 1024)} embeddings, got {vectors.shape}.")
    return vectors


def main():
    from azure.identity import ManagedIdentityCredential

    args = parse_args()
    for name, value in vars(args).items():
        print(f"{name}: {value}")

    output_dir = Path(args.output_dir)
    (output_dir / "embeddings").mkdir(parents=True, exist_ok=True)

    rows = read_cohort_csv(args.cohort_csv)
    slides = resolve_slides(rows, args.images_dir)
    print(f"{len(slides)} slide(s) to embed")

    # The job authenticates as the cluster's managed identity. No local login or .env is
    # sent to the compute; the identity needs Cognitive Services User on the Foundry resource.
    client_id = args.managed_identity_client_id or os.getenv("DEFAULT_IDENTITY_CLIENT_ID")
    credential = ManagedIdentityCredential(client_id=client_id or None)
    client = PremiumClient(resource_endpoint(args.endpoint), args.model, credential)

    tiling, images = [], []
    try:
        for index, (row, path) in enumerate(slides):
            image_id = f"image_{index:03d}"
            tiles, metadata = tile_slide(
                path,
                target_mpp=args.target_mpp,
                tile_size=args.tile_size,
                occupancy_threshold=args.occupancy_threshold,
                max_tiles=args.max_tiles,
            )
            vectors = embed_tiles(client, tiles, args.batch_size)
            np.savez_compressed(output_dir / "embeddings" / f"{image_id}.npz",
                                image_embeddings=vectors)

            tiling.append({"image_id": image_id, "file_name": row["file_name"], **metadata})
            images.append({"image_id": image_id, "file_name": row["file_name"],
                           "tile_count": len(tiles)})
            # Written after every slide so a failure still leaves the finished work behind.
            write_json(output_dir / "tiles.json", tiling)
            write_json(output_dir / "images.json",
                       {"model": args.model, "images": images, "api_calls": client.calls})
            print(f"[{image_id}] {row['file_name']}: {len(tiles)} tiles "
                  f"(of {metadata['candidate_tiles']} tissue candidates), "
                  f"effective_mpp={metadata['effective_mpp']:.4f}", flush=True)
    finally:
        client.close()

    log_metrics({"n_images": len(images),
                 "n_tiles": sum(row["tile_count"] for row in images),
                 "n_api_calls": len(client.calls)})
    print(json.dumps({"images": len(images), "api_calls": len(client.calls)}, indent=2))


if __name__ == "__main__":
    main()
