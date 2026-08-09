#!/usr/bin/env python3
"""
Merge multiple results_merged.json files into one CSV with added metadata.

Expected folder names (in the same parent directory as this script, unless --root is used):
- results_prune_notUniform
- results_notPrune_notUniform
- results_prune_uniform
- results_notPrune_uniform

Each folder contains: results_merged.json (a JSON list of objects)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def infer_flags_from_folder(folder_name: str) -> Dict[str, Any]:
    """
    Given folder name like 'results_prune_notUniform', infer:
      - experimental_condition: 'prune_notUniform'
      - prune: bool
      - uniform: bool
    """
    if not folder_name.startswith("results_"):
        raise ValueError(f"Folder name must start with 'results_': {folder_name}")

    experimental_condition = folder_name[len("results_") :]

    lower = folder_name.lower()

    # prune flag
    if "notprune" in lower:
        prune = False
    elif "prune" in lower:
        prune = True
    else:
        raise ValueError(f"Cannot infer prune/notPrune from folder name: {folder_name}")

    # uniform flag
    # we treat both 'uniform' and 'notUniform' (case-insensitive)
    if "notuniform" in lower:
        uniform = False
    elif "uniform" in lower:
        uniform = True
    else:
        raise ValueError(f"Cannot infer uniform/notUniform from folder name: {folder_name}")

    return {
        "experimental_condition": experimental_condition,
        "prune": prune,
        "uniform": uniform,
    }


def load_and_transform(json_path: Path, folder_name: str) -> List[Dict[str, Any]]:
    """
    Load list of records from results_merged.json and apply:
      - rename 'condition' -> 'method'
      - add experimental_condition/prune/uniform
    """
    meta = infer_flags_from_folder(folder_name)

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {json_path}, got {type(data)}")

    out: List[Dict[str, Any]] = []
    for i, rec in enumerate(data):
        if not isinstance(rec, dict):
            raise ValueError(
                f"Expected each item to be an object/dict in {json_path} at index {i}, got {type(rec)}"
            )

        new_rec = dict(rec)

        # rename condition -> method
        if "condition" in new_rec:
            new_rec["method"] = new_rec.pop("condition")
        else:
            # If already method, keep it; otherwise error
            if "method" not in new_rec:
                raise KeyError(f"Record missing 'condition' (and no 'method') in {json_path} at index {i}")

        # add metadata
        new_rec.update(meta)

        out.append(new_rec)

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge results_merged.json files into one CSV.")
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Root directory containing the results_* folders (default: current directory).",
    )
    parser.add_argument(
        "--folders",
        type=str,
        nargs="*",
        default=[
            "results_prune_notUniform",
            "results_notPrune_notUniform",
            "results_prune_uniform",
            "results_notPrune_uniform",
        ],
        help="Folders to process (default: the four standard folders).",
    )
    parser.add_argument(
        "--input_name",
        type=str,
        default="results_merged.json",
        help="Input JSON filename inside each folder (default: results_merged.json).",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="results_all_merged.csv",
        help="Output CSV path (default: results_all_merged.csv in root).",
    )

    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    all_records: List[Dict[str, Any]] = []

    for folder in args.folders:
        folder_path = root / folder
        json_path = folder_path / args.input_name

        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        if not json_path.exists():
            raise FileNotFoundError(f"Input file not found: {json_path}")

        records = load_and_transform(json_path, folder_name=folder_path.name)
        all_records.extend(records)
        print(f"Loaded {len(records):,} records from {json_path}")

    if not all_records:
        raise RuntimeError("No records were loaded. Check your folders and input files.")

    df = pd.DataFrame(all_records)

    out_csv_path = (root / args.out_csv).resolve()
    df.to_csv(out_csv_path, index=False)
    print(f"\nWrote {len(df):,} rows to: {out_csv_path}")


if __name__ == "__main__":
    main()
