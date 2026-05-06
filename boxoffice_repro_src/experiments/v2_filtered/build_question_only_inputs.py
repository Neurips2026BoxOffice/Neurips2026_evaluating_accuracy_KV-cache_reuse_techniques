#!/usr/bin/env python3
"""Strip a release dataset down to question-only prompts."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def convert(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        new_row = json.loads(json.dumps(row))
        question_segment = f"Question: {new_row['question']}"
        new_row["passages"] = []
        new_row["ctxs"] = []
        new_row["prompt_segments"] = [question_segment]
        new_row["prompt_text"] = question_segment
        out.append(new_row)
    return out


def main() -> None:
    if len(sys.argv) != 5:
        raise SystemExit("usage: build_question_only_inputs.py full_in eval_in full_out eval_out")
    full_in = Path(sys.argv[1])
    eval_in = Path(sys.argv[2])
    full_out = Path(sys.argv[3])
    eval_out = Path(sys.argv[4])
    write_jsonl(full_out, convert(load_jsonl(full_in)))
    write_jsonl(eval_out, convert(load_jsonl(eval_in)))


if __name__ == "__main__":
    main()
