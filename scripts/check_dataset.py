"""Preflight: do the labels and the slides in Blob storage line up? Lists names only.

No image is downloaded and no model is called. The default check makes storage requests;
`--csv-only` skips Azure entirely and just checks the local sheet.

    python scripts/check_dataset.py [--csv-only]
"""

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from email.utils import formatdate
from pathlib import Path, PureWindowsPath

import requests
from azure.identity import AzureCliCredential
from dotenv import load_dotenv

from cohort import read_cohort
from mii_client import ensure_azure_cli_path, write_json

HERE = Path(__file__).resolve().parent.parent


def list_blobs(container, prefix, tenant_id):
    ensure_azure_cli_path()
    with AzureCliCredential(tenant_id=tenant_id or None) as credential:
        token = credential.get_token("https://storage.azure.com/.default")
        response = requests.get(
            container,
            params={"restype": "container", "comp": "list", "prefix": prefix, "maxresults": 1000},
            headers={"Authorization": "Bearer " + token.token,
                     "x-ms-version": "2023-11-03", "x-ms-date": formatdate(usegmt=True)},
            timeout=(15, 60), allow_redirects=False,
        )
    if response.status_code != 200:
        raise RuntimeError(f"Blob HTTP {response.status_code}: "
                           f"{response.headers.get('x-ms-error-code')}")
    root = ET.fromstring(response.content)
    if root.findtext("NextMarker"):
        raise RuntimeError("More than one page of blobs; the count would be incomplete.")
    return [name for name in (blob.findtext("Name") for blob in root.findall("./Blobs/Blob"))
            if name and not name.endswith("/")]


def main():
    load_dotenv(HERE / ".env", override=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sheet", type=Path,
                        default=HERE / (os.getenv("LABELS_CSV") or "data/labels.csv"))
    parser.add_argument("--csv-only", action="store_true")
    args = parser.parse_args()

    rows = read_cohort(args.sheet)
    report = {
        "evaluation_unit": "image_lesion",
        "subject_id_used_for_deduplication": False,
        "sheet_rows": len(rows),
        "label_counts": dict(Counter(row["label"] or "unlabeled" for row in rows)),
        "labeled_rows": sum(row["label"] is not None for row in rows),
        "image_downloads": 0,
        "model_calls": 0,
    }

    if not args.csv_only:
        container = os.environ["HNE_BLOB_CONTAINER_URL"]
        blobs = list_blobs(container, os.getenv("HNE_BLOB_PREFIX", ""),
                           os.getenv("TENANT_ID"))
        lookup = {}
        for blob in blobs:
            lookup.setdefault(PureWindowsPath(blob).name.casefold(), []).append(blob)
        labeled = [row for row in rows if row["label"] is not None]
        report.update({
            "blob_file_count": len(blobs),
            "blob_extension_counts": dict(Counter(PureWindowsPath(b).suffix.lower() for b in blobs)),
            "matched_labeled_images": sum(len(lookup.get(row["file_name"].casefold(), [])) == 1
                                          for row in labeled),
            "missing_labeled_images": [row["file_name"] for row in labeled
                                       if not lookup.get(row["file_name"].casefold())],
            "ambiguous_labeled_images": [row["file_name"] for row in labeled
                                         if len(lookup.get(row["file_name"].casefold(), [])) > 1],
        })

    output = HERE / "outputs" / datetime.now(timezone.utc).strftime("dataset-check-%Y%m%dT%H%M%S%fZ.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("Report:", output)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR ({type(error).__name__}): {error}", file=sys.stderr)
        sys.exit(1)
