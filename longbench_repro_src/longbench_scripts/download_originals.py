#!/usr/bin/env python3
"""
Download the ORIGINAL datasets (with supporting_facts / golden annotations)
that LongBench sampled from. Uses direct parquet/JSON downloads to avoid
datasets library compatibility issues.

Sources:
  - 2WikiMultiHopQA: https://huggingface.co/datasets/xanhho/2WikiMultihopQA
  - HotpotQA:        https://huggingface.co/datasets/hotpotqa/hotpot_qa
  - MuSiQue:         https://huggingface.co/datasets/dgslibiern/MuSiQue

Usage:
    python download_originals.py
    python download_originals.py --output-dir ../data/originals
"""

import argparse
import json
import os
import tempfile
import urllib.request
from pathlib import Path
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Handle numpy types in JSON serialization."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)

DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "data" / "originals"


def download_file(url: str, dest: Path):
    """Download a file with progress."""
    print(f"    Downloading {url}...", flush=True)
    urllib.request.urlretrieve(url, dest)
    size_mb = dest.stat().st_size / 1024 / 1024
    print(f"    Saved {size_mb:.1f}MB to {dest}", flush=True)


def download_2wiki(output_dir: Path):
    out = output_dir / "2wikimqa_dev.json"
    if out.exists():
        print(f"  [skip] {out}")
        return

    import pandas as pd

    # LongBench sampled from the DEV split, not test
    parquet_url = "https://huggingface.co/datasets/xanhho/2WikiMultihopQA/resolve/main/dev.parquet"
    tmp_parquet = output_dir / "_2wiki_test.parquet"
    download_file(parquet_url, tmp_parquet)

    df = pd.read_parquet(tmp_parquet)
    print(f"  Loaded {len(df)} rows, columns: {list(df.columns)}")

    rows = []
    for _, r in df.iterrows():
        rows.append({
            "_id": str(r.get("_id", "")),
            "type": str(r.get("type", "")),
            "question": str(r["question"]),
            "answer": str(r["answer"]),
            "context": r["context"],
            "supporting_facts": r["supporting_facts"],
            "evidences": r.get("evidences", []),
        })

    with open(out, "w") as f:
        json.dump(rows, f, ensure_ascii=False, cls=NumpyEncoder)
    tmp_parquet.unlink()
    print(f"  Saved {len(rows)} samples to {out}")
    print(f"  Sample: q='{rows[0]['question'][:60]}' a='{rows[0]['answer']}'")
    sf = rows[0]["supporting_facts"]
    print(f"  supporting_facts type: {type(sf)}, preview: {str(sf)[:100]}")


def download_hotpotqa(output_dir: Path):
    out = output_dir / "hotpotqa_validation.json"
    if out.exists():
        print(f"  [skip] {out}")
        return

    import pandas as pd

    # HotpotQA distractor validation
    parquet_url = "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/distractor/validation-00000-of-00001.parquet"
    tmp_parquet = output_dir / "_hotpotqa_val.parquet"
    download_file(parquet_url, tmp_parquet)

    df = pd.read_parquet(tmp_parquet)
    print(f"  Loaded {len(df)} rows, columns: {list(df.columns)}")

    rows = []
    for _, r in df.iterrows():
        rows.append({
            "id": str(r["id"]),
            "type": str(r.get("type", "")),
            "level": str(r.get("level", "")),
            "question": str(r["question"]),
            "answer": str(r["answer"]),
            "context": r["context"],
            "supporting_facts": r["supporting_facts"],
        })

    with open(out, "w") as f:
        json.dump(rows, f, ensure_ascii=False, cls=NumpyEncoder)
    tmp_parquet.unlink()
    print(f"  Saved {len(rows)} samples to {out}")
    print(f"  Sample: q='{rows[0]['question'][:60]}' a='{rows[0]['answer']}'")


def download_musique(output_dir: Path):
    out = output_dir / "musique_validation.json"
    if out.exists():
        print(f"  [skip] {out}")
        return

    # fladhak/musique has dev.json with is_supporting annotations (public, no auth)
    url = "https://huggingface.co/datasets/fladhak/musique/resolve/main/dev.json"
    tmp = output_dir / "_musique_dev.json"
    download_file(url, tmp)

    with open(tmp) as f:
        rows = json.load(f)
    tmp.unlink()

    print(f"  Loaded {len(rows)} samples")
    print(f"  Keys: {list(rows[0].keys())}")

    with open(out, "w") as f:
        json.dump(rows, f, ensure_ascii=False)
    print(f"  Saved {len(rows)} samples to {out}")
    print(f"  Sample: q='{rows[0]['question'][:60]}' a='{rows[0]['answer']}'")
    paras = rows[0].get("paragraphs", [])
    if paras:
        n_sup = sum(1 for p in paras if p.get("is_supporting"))
        print(f"  First sample: {len(paras)} paragraphs, {n_sup} supporting")


def main():
    ap = argparse.ArgumentParser(description="Download original datasets with golden annotations.")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=== 2WikiMultiHopQA ===")
    download_2wiki(args.output_dir)
    print()

    print("=== HotpotQA ===")
    download_hotpotqa(args.output_dir)
    print()

    print("=== MuSiQue ===")
    download_musique(args.output_dir)
    print()

    print("Done. Now run: python enrich.py")


if __name__ == "__main__":
    main()
