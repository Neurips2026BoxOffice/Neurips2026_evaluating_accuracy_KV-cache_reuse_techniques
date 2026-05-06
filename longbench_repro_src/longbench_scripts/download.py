#!/usr/bin/env python3
"""
Download original LongBench datasets for 2wikimqa, hotpotqa, musique.

LongBench is hosted on HuggingFace: THUDM/LongBench
The data is in a data.zip containing per-dataset JSONL files.
Each has fields: input, context, answers, length, dataset, language, all_classes, _id

The "200 splits" are simply the full dataset — LongBench ships exactly 200 samples
per QA dataset. There is no further subsampling; 200 IS the full LongBench test set
for each task.

Source: https://huggingface.co/datasets/THUDM/LongBench
Paper: "LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding"
       (Bai et al., 2023)

Usage:
    python download.py
    python download.py --output-dir ../data/longbench_raw
"""

import argparse
import json
import os
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ZIP_URL = "https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip"
WANTED = {"2wikimqa.jsonl", "hotpotqa.jsonl", "musique.jsonl"}
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "data" / "longbench_raw"


def download_longbench(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    existing = {f.name for f in output_dir.glob("*.jsonl")}
    if WANTED.issubset(existing):
        print("All files already exist. Skipping download.")
    else:
        # Download zip
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            tmp_path = tmp.name
        print(f"Downloading {ZIP_URL}...", flush=True)
        urllib.request.urlretrieve(ZIP_URL, tmp_path)
        print(f"  Saved to {tmp_path}", flush=True)

        # Extract wanted files
        with zipfile.ZipFile(tmp_path) as zf:
            for member in zf.namelist():
                basename = os.path.basename(member)
                if basename in WANTED:
                    out_path = output_dir / basename
                    with zf.open(member) as src, open(out_path, "wb") as dst:
                        dst.write(src.read())
                    print(f"  Extracted {basename}")

        os.unlink(tmp_path)

    # Summary
    print(f"\n{'='*60}")
    print("LongBench dataset sizes (these ARE the full test sets):")
    for name in sorted(WANTED):
        p = output_dir / name
        if p.exists():
            rows = []
            with open(p) as f:
                for line in f:
                    if line.strip():
                        rows.append(json.loads(line))
            print(f"  {name}: {len(rows)} samples")
            print(f"    Keys: {list(rows[0].keys())}")
            print(f"    First query: {rows[0]['input'][:80]}")
            print(f"    First answer: {rows[0]['answers']}")
        else:
            print(f"  {name}: MISSING")
    print()
    print("Note: LongBench ships exactly 200 samples per QA dataset.")
    print("The '200 split' is the COMPLETE test set, not a subsample.")


def main():
    ap = argparse.ArgumentParser(description="Download LongBench QA datasets.")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()
    download_longbench(args.output_dir)


if __name__ == "__main__":
    main()
