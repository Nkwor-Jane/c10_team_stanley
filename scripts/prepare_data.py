"""
Prepare Complaint Sense working dataset from CFPB Consumer Complaint Database.

Matches the data card:
  - Filter to 11 Issue categories
  - Require non-empty consumer complaint narratives
  - Remove exact-duplicate narratives
  - Build a balanced modelling subset (~30 per class, min available for smallest)
"""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

TARGET_ISSUES = [
    "Incorrect information on your report",
    "Improper use of your report",
    "Managing an account",
    "Problem with a company's investigation into an existing problem",
    "Problem with a purchase shown on your statement",
    "Attempts to collect debt not owed",
    "Trouble during payment process",
    "False statements or representation",
    "Struggling to pay mortgage",
    "Written notification about debt",
    "Took or threatened to take negative or legal action",
]

NARRATIVE_COL = "Consumer complaint narrative"
ISSUE_COL = "Issue"
DATE_COL = "Date received"
CFPB_ZIP_URL = "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"


def resolve_data_dir() -> Path:
    """
    Resolve the project's data/ directory, always as a sibling of scripts/
    (never nested inside it) — matches the notebook's ROOT resolution so
    both tools read/write the same data/ folder regardless of whether this
    script is run from Tri-ai/ or Tri-ai/scripts/.
    """
    root = Path(".").resolve()
    if root.name == "scripts":
        root = root.parent
    return root / "data"


def download_cfpb(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading CFPB complaints (~large) from {CFPB_ZIP_URL} ...")
    resp = requests.get(CFPB_ZIP_URL, timeout=600, stream=True)
    resp.raise_for_status()
    raw = resp.content
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise RuntimeError("No CSV found inside CFPB zip.")
        csv_name = names[0]
        out = dest if dest.suffix.lower() == ".csv" else dest / "complaints.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(csv_name) as src, open(out, "wb") as dst:
            dst.write(src.read())
    print(f"Saved raw CSV to {out}")
    return out


def load_raw(path: Path) -> pd.DataFrame:
    print(f"Loading {path} ...")
    df = pd.read_csv(path, low_memory=False)
    return df


def prepare_working_set(
    df: pd.DataFrame,
    start_date: str | None,
    end_date: str | None,
    per_class: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    required = [NARRATIVE_COL, ISSUE_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    work = df.copy()
    work[NARRATIVE_COL] = work[NARRATIVE_COL].astype(str).str.strip()
    work = work[work[NARRATIVE_COL].notna()]
    work = work[work[NARRATIVE_COL].str.len() > 20]
    work = work[~work[NARRATIVE_COL].isin(["", "nan", "None"])]

    if DATE_COL in work.columns and (start_date or end_date):
        work[DATE_COL] = pd.to_datetime(work[DATE_COL], errors="coerce")
        if start_date:
            work = work[work[DATE_COL] >= pd.Timestamp(start_date)]
        if end_date:
            work = work[work[DATE_COL] <= pd.Timestamp(end_date)]

    work = work[work[ISSUE_COL].isin(TARGET_ISSUES)].copy()
    exported_n = len(work)

    work = work.drop_duplicates(subset=[NARRATIVE_COL], keep="first")
    unique_n = len(work)

    parts = []
    for issue in TARGET_ISSUES:
        subset = work[work[ISSUE_COL] == issue]
        n = min(per_class, len(subset))
        if n == 0:
            print(f"WARNING: no rows for issue: {issue}")
            continue
        parts.append(subset.sample(n=n, random_state=seed))

    balanced = pd.concat(parts, ignore_index=True)
    balanced = balanced.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    stats = {
        "exported_filtered": exported_n,
        "unique_after_dedupe": unique_n,
        "balanced_size": len(balanced),
        "class_counts": balanced[ISSUE_COL].value_counts().to_dict(),
    }
    return work, balanced, stats


def main() -> None:
    data_dir = resolve_data_dir()

    parser = argparse.ArgumentParser(description="Prepare Complaint Sense dataset")
    parser.add_argument(
        "--input",
        type=Path,
        default=data_dir / "complaints.csv",
        help="Path to raw CFPB CSV (downloaded if missing and --download is set). "
        "Defaults to <project_root>/data/complaints.csv.",
    )
    parser.add_argument("--download", action="store_true", help="Download full CFPB CSV zip if input missing")
    parser.add_argument("--start-date", default=None, help="Optional start date YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, help="Optional end date YYYY-MM-DD")
    parser.add_argument("--per-class", type=int, default=30, help="Max samples per Issue class")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=data_dir,
        help="Where prepared CSVs are written. Defaults to <project_root>/data "
        "(a sibling of scripts/, never nested inside it), regardless of whether "
        "this script is run from the project root or from scripts/.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.input.exists():
        if args.download:
            args.input = download_cfpb(args.input)
        else:
            raise FileNotFoundError(
                f"{args.input} not found. Place a CFPB CSV there or re-run with --download."
            )

    raw = load_raw(args.input)
    pool, balanced, stats = prepare_working_set(
        raw,
        start_date=args.start_date,
        end_date=args.end_date,
        per_class=args.per_class,
        seed=args.seed,
    )

    keep_cols = [
        c
        for c in [
            "Complaint ID",
            NARRATIVE_COL,
            "Product",
            "Sub-product",
            ISSUE_COL,
            "Sub-issue",
            "Company",
            DATE_COL,
            "State",
            "ZIP code",
            "Company response to consumer",
            "Submitted via",
        ]
        if c in balanced.columns
    ]

    pool_path = args.out_dir / "complaint_sense_pool.csv"
    balanced_path = args.out_dir / "complaint_sense_working.csv"
    pool[keep_cols].to_csv(pool_path, index=False)
    balanced[keep_cols].to_csv(balanced_path, index=False)

    print("\n=== Preparation summary ===")
    for k, v in stats.items():
        print(f"{k}: {v}")
    print(f"\nWrote pool      -> {pool_path}")
    print(f"Wrote working   -> {balanced_path}")


if __name__ == "__main__":
    main()