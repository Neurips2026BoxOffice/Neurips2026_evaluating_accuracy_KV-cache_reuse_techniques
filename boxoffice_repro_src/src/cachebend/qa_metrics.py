"""Shared QA-style metrics used by pilot and LongBench evaluations."""

from __future__ import annotations

import collections
import re
import string
from typing import Any, List, Sequence


def parse_generation(text: str) -> str:
    text = (text or "").lstrip("\n").split("\n")[0]
    if text.startswith("Yes") or text.startswith("yes"):
        return "Yes"
    pieces = text.split()
    if pieces and (pieces[0].startswith("No") or pieces[0].startswith("no")):
        return "No"
    return text


def normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    def lower(value: str) -> str:
        return value.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text or ""))))


def flatten_answers(raw_answers: Any) -> List[str]:
    if raw_answers is None:
        return []
    if isinstance(raw_answers, str):
        return [raw_answers]

    flattened: List[str] = []
    if isinstance(raw_answers, Sequence):
        for item in raw_answers:
            if isinstance(item, str):
                flattened.append(item)
            elif isinstance(item, Sequence):
                flattened.extend(str(x) for x in item)
            else:
                flattened.append(str(item))
    else:
        flattened.append(str(raw_answers))
    return [value for value in flattened if value]


def compute_f1(prediction: str, gold: str | List[str], tokenizer: Any) -> float:
    if not isinstance(gold, str):
        return max((compute_f1(prediction, answer, tokenizer) for answer in gold), default=0.0)

    pred_tokens = tokenizer.encode(normalize_answer(parse_generation(prediction)), add_special_tokens=False)
    gold_tokens = tokenizer.encode(normalize_answer(gold), add_special_tokens=False)
    common = collections.Counter(gold_tokens) & collections.Counter(pred_tokens)
    num_same = sum(common.values())
    if len(gold_tokens) == 0 or len(pred_tokens) == 0:
        return float(gold_tokens == pred_tokens)
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2.0 * precision * recall) / (precision + recall)


def compute_normalized_em(prediction: str, answers: List[str]) -> float:
    pred_norm = normalize_answer(parse_generation(prediction))
    return float(any(normalize_answer(answer) == pred_norm for answer in answers))
