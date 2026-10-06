"""Extract a private `file_name,label` cohort from an authorized H&E label sheet.

Keep the source sheet in the ignored data/ folder and generated cohorts in outputs/.
File names can contain patient or specimen identifiers; reducing columns is not anonymization.

    python scripts/cohort.py data/labels.csv --out outputs/cohort.csv
"""

import argparse
import csv
import re
from collections import Counter
from pathlib import Path, PureWindowsPath

CLASS_NAMES = ["Non-Responder", "Responder"]
FILE_COLUMN = "H&E Images"
LABEL_COLUMN = "Responder/Not Responder"

# The sheet spells the negative class several ways; normalise before comparing.
LABEL_SPELLINGS = {"responder": "Responder", "nonresponder": "Non-Responder", "": None}


def read_cohort(path):
    """Read every row. Rows keep `label=None` when the sheet has no response recorded."""
    rows, seen = [], set()
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            # The sheet stores a Windows path from the acquisition machine; only the name joins.
            file_name = PureWindowsPath(row[FILE_COLUMN].strip()).name
            if not file_name or file_name.casefold() in seen:
                raise ValueError("Empty or duplicate image file name in the sheet.")
            seen.add(file_name.casefold())
            spelling = re.sub(r"[\s_-]+", "", row[LABEL_COLUMN].casefold())
            if spelling not in LABEL_SPELLINGS:
                raise ValueError(f"Unrecognised response label: {row[LABEL_COLUMN]!r}")
            # One row is one image, i.e. one lesion site. Rows are never merged by subject:
            # two slides from the same subject are different sites, not one lesion twice.
            rows.append({"file_name": file_name, "label": LABEL_SPELLINGS[spelling]})
    return rows


def labeled_only(rows):
    """Drop rows with no response label. Unlabeled slides are not embedded and not scored."""
    return [row for row in rows if row["label"] is not None]


def write_cohort(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["file_name", "label"])
        writer.writeheader()
        writer.writerows(rows)


def read_cohort_csv(path):
    """Read back a cohort.csv written by write_cohort, validating the labels."""
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or any(row["label"] not in CLASS_NAMES for row in rows):
        raise ValueError(f"cohort.csv needs a nonempty file_name and a label in {CLASS_NAMES}.")
    names = [row["file_name"].casefold() for row in rows]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate file_name in cohort.csv.")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sheet", type=Path, help="the clinical label sheet (CSV)")
    parser.add_argument("--out", type=Path, default=Path("outputs/cohort.csv"))
    args = parser.parse_args()

    rows = read_cohort(args.sheet)
    labeled = labeled_only(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_cohort(args.out, labeled)
    print(f"{len(rows)} rows in the sheet, {len(labeled)} labeled -> {args.out}")
    print(dict(Counter(row["label"] for row in labeled)))


if __name__ == "__main__":
    main()
