#!/usr/bin/env python3
"""
Match LongBench queries to original datasets and produce _lb.json with golden annotations.

For each LongBench query, matches by question text to the original dataset,
recovers supporting_facts / is_supporting, and outputs our format:
  {query_id, query, context (list), golden (list of indices), answer, answers_all}

Usage:
    python enrich.py
    python enrich.py --longbench-dir ../data/longbench_raw \\
        --originals-dir ../data/originals --output-dir ../data/enriched
"""

import argparse
import json
import re
import string
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
DEFAULT_LB = DATA_DIR / "longbench_raw"
DEFAULT_ORIG = DATA_DIR / "originals"
DEFAULT_OUT = DATA_DIR / "enriched"


def normalize_question(q: str) -> str:
    """Normalize question for matching."""
    q = q.strip().lower()
    q = re.sub(r"\s+", " ", q)
    q = q.rstrip("?").strip()
    return q


def parse_supporting_facts(sf) -> Set[str]:
    """Extract golden titles from supporting_facts in any format.

    Handles:
      - str: JSON-encoded list of [title, sent_id] pairs
      - list: list of [title, sent_id] pairs
      - dict: {"title": [...], "sent_id": [...]}
    """
    if isinstance(sf, str):
        try:
            sf = json.loads(sf)
        except (json.JSONDecodeError, TypeError):
            return set()
    if isinstance(sf, dict):
        return set(str(t) for t in sf.get("title", []))
    if isinstance(sf, list):
        return set(str(pair[0]) for pair in sf if isinstance(pair, (list, tuple)) and len(pair) >= 1)
    return set()


def parse_longbench_passages(context_str: str, query_id: str = "") -> Tuple[List[Dict[str, str]], List[str]]:
    """Split LongBench context string into passages, removing duplicates.

    Returns (passages, removed_log) where removed_log lists what was filtered.
    """
    parts = re.split(r"\nPassage \d+:\n", "\n" + context_str)
    passages = []
    seen_titles: Set[str] = set()
    removed_log = []
    for idx, p in enumerate(parts):
        p = p.strip()
        if not p:
            continue
        lines = p.split("\n", 1)
        title = lines[0].strip()
        text = lines[1].strip() if len(lines) > 1 else ""
        if title in seen_titles:
            removed_log.append(f"  query={query_id}: removed duplicate passage #{idx} title='{title}'")
            continue
        seen_titles.add(title)
        passages.append({"title": title, "text": text, "full": p})
    return passages, removed_log


def enrich_2wiki(lb_dir: Path, orig_dir: Path, out_dir: Path):
    lb_path = lb_dir / "2wikimqa.jsonl"
    orig_path = orig_dir / "2wikimqa_dev.json"
    out_path = out_dir / "2wikimultihopqa_lb.json"

    if not lb_path.exists():
        print(f"  SKIP: {lb_path} not found")
        return
    if not orig_path.exists():
        print(f"  SKIP: {orig_path} not found")
        return

    # Load LongBench
    lb_rows = []
    with open(lb_path) as f:
        for line in f:
            if line.strip():
                lb_rows.append(json.loads(line))

    # Load original, index by normalized question
    with open(orig_path) as f:
        orig_data = json.load(f)
    orig_by_q: Dict[str, dict] = {}
    for r in orig_data:
        orig_by_q[normalize_question(r["question"])] = r

    print(f"  LongBench: {len(lb_rows)}, Original: {len(orig_data)}")

    # Build supporting_facts lookup for each original
    converted = []
    matched = 0
    for i, lb in enumerate(lb_rows):
        question = lb["input"].strip()
        answers = lb.get("answers", [])

        # Parse LB passages
        qid = lb.get("_id", str(i))
        passages, dup_log = parse_longbench_passages(lb["context"], query_id=qid)
        for msg in dup_log:
            print(msg)

        # Match to original
        orig = orig_by_q.get(normalize_question(question))
        golden_indices = []

        if orig is not None:
            matched += 1
            golden_titles = parse_supporting_facts(orig.get("supporting_facts", {}))

            # Match LB passages to golden titles
            for pi, passage in enumerate(passages):
                if passage["title"] in golden_titles:
                    golden_indices.append(pi)

        context = [p["full"] for p in passages]
        answer = answers[0] if answers else (orig["answer"] if orig else "")

        converted.append({
            "query_id": qid,
            "query": question,
            "context": context,
            "golden": golden_indices,
            "answer": answer,
            "answers_all": answers,
            "source": "longbench_2wikimqa",
            "matched_original": orig is not None,
        })

    with open(out_path, "w") as f:
        json.dump(converted, f, indent=2, ensure_ascii=False)

    n_golden = sum(1 for c in converted if c["golden"])
    avg_golden = sum(len(c["golden"]) for c in converted) / len(converted) if converted else 0
    print(f"  Matched: {matched}/{len(lb_rows)}")
    print(f"  With golden: {n_golden}/{len(converted)}, avg golden: {avg_golden:.1f}")
    print(f"  Saved: {out_path}")


def enrich_hotpotqa(lb_dir: Path, orig_dir: Path, out_dir: Path):
    lb_path = lb_dir / "hotpotqa.jsonl"
    orig_path = orig_dir / "hotpotqa_validation.json"
    out_path = out_dir / "hotpotqa_lb.json"

    if not lb_path.exists():
        print(f"  SKIP: {lb_path} not found")
        return
    if not orig_path.exists():
        print(f"  SKIP: {orig_path} not found")
        return

    lb_rows = []
    with open(lb_path) as f:
        for line in f:
            if line.strip():
                lb_rows.append(json.loads(line))

    with open(orig_path) as f:
        orig_data = json.load(f)
    orig_by_q: Dict[str, dict] = {}
    for r in orig_data:
        orig_by_q[normalize_question(r["question"])] = r

    print(f"  LongBench: {len(lb_rows)}, Original: {len(orig_data)}")

    converted = []
    matched = 0
    for i, lb in enumerate(lb_rows):
        question = lb["input"].strip()
        answers = lb.get("answers", [])
        qid = lb.get("_id", str(i))
        passages, dup_log = parse_longbench_passages(lb["context"], query_id=qid)
        for msg in dup_log:
            print(msg)

        orig = orig_by_q.get(normalize_question(question))
        golden_indices = []

        if orig is not None:
            matched += 1
            sf = orig.get("supporting_facts", {})
            if isinstance(sf, dict):
                golden_titles = set(str(t) for t in sf.get("title", []))
            elif isinstance(sf, list):
                golden_titles = set(str(pair[0]) for pair in sf)
            else:
                golden_titles = set()

            for pi, passage in enumerate(passages):
                if passage["title"] in golden_titles:
                    golden_indices.append(pi)

        context = [p["full"] for p in passages]
        answer = answers[0] if answers else (orig["answer"] if orig else "")

        converted.append({
            "query_id": qid,
            "query": question,
            "context": context,
            "golden": golden_indices,
            "answer": answer,
            "answers_all": answers,
            "source": "longbench_hotpotqa",
            "matched_original": orig is not None,
        })

    with open(out_path, "w") as f:
        json.dump(converted, f, indent=2, ensure_ascii=False)

    n_golden = sum(1 for c in converted if c["golden"])
    avg_golden = sum(len(c["golden"]) for c in converted) / len(converted) if converted else 0
    print(f"  Matched: {matched}/{len(lb_rows)}")
    print(f"  With golden: {n_golden}/{len(converted)}, avg golden: {avg_golden:.1f}")
    print(f"  Saved: {out_path}")


def enrich_musique(lb_dir: Path, orig_dir: Path, out_dir: Path):
    lb_path = lb_dir / "musique.jsonl"
    orig_path = orig_dir / "musique_validation.json"
    out_path = out_dir / "musique_lb.json"

    if not lb_path.exists():
        print(f"  SKIP: {lb_path} not found")
        return
    if not orig_path.exists():
        print(f"  SKIP: {orig_path} not found")
        return

    lb_rows = []
    with open(lb_path) as f:
        for line in f:
            if line.strip():
                lb_rows.append(json.loads(line))

    with open(orig_path) as f:
        orig_data = json.load(f)
    orig_by_q: Dict[str, dict] = {}
    for r in orig_data:
        orig_by_q[normalize_question(r["question"])] = r

    print(f"  LongBench: {len(lb_rows)}, Original: {len(orig_data)}")

    converted = []
    matched = 0
    for i, lb in enumerate(lb_rows):
        question = lb["input"].strip()
        answers = lb.get("answers", [])
        qid = lb.get("_id", str(i))
        passages, dup_log = parse_longbench_passages(lb["context"], query_id=qid)
        for msg in dup_log:
            print(msg)

        orig = orig_by_q.get(normalize_question(question))
        golden_indices = []

        if orig is not None:
            matched += 1
            # MuSiQue uses paragraphs with is_supporting flag
            paras = orig.get("paragraphs", [])
            if isinstance(paras, str):
                try:
                    paras = json.loads(paras)
                except (json.JSONDecodeError, TypeError):
                    paras = []
            golden_titles = set()
            for p in paras:
                if isinstance(p, str):
                    try:
                        p = json.loads(p)
                    except:
                        continue
                if p.get("is_supporting", False):
                    golden_titles.add(str(p.get("title", "")))

            for pi, passage in enumerate(passages):
                if passage["title"] in golden_titles:
                    golden_indices.append(pi)

        context = [p["full"] for p in passages]
        answer = answers[0] if answers else (orig.get("answer", "") if orig else "")

        converted.append({
            "query_id": qid,
            "query": question,
            "context": context,
            "golden": golden_indices,
            "answer": answer,
            "answers_all": answers,
            "source": "longbench_musique",
            "matched_original": orig is not None,
        })

    with open(out_path, "w") as f:
        json.dump(converted, f, indent=2, ensure_ascii=False)

    n_golden = sum(1 for c in converted if c["golden"])
    avg_golden = sum(len(c["golden"]) for c in converted) / len(converted) if converted else 0
    print(f"  Matched: {matched}/{len(lb_rows)}")
    print(f"  With golden: {n_golden}/{len(converted)}, avg golden: {avg_golden:.1f}")
    print(f"  Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Enrich LongBench with golden annotations from originals.")
    ap.add_argument("--longbench-dir", type=Path, default=DEFAULT_LB)
    ap.add_argument("--originals-dir", type=Path, default=DEFAULT_ORIG)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    print("=== 2WikiMultiHopQA ===")
    enrich_2wiki(args.longbench_dir, args.originals_dir, args.output_dir)
    print()

    print("=== HotpotQA ===")
    enrich_hotpotqa(args.longbench_dir, args.originals_dir, args.output_dir)
    print()

    print("=== MuSiQue ===")
    enrich_musique(args.longbench_dir, args.originals_dir, args.output_dir)
    print()


if __name__ == "__main__":
    main()
