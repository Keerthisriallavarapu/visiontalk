"""VQA evaluation.

VQAv2's grader is the standard: for each (question, image) you get up to
10 ground-truth answers; the predicted answer is graded as
min(matching_gt_answers / 3, 1.0). So 3+ annotators agreeing = 1.0,
1 = 0.33, 0 = 0.0.

GQA grader is exact match after normalization (lowercase, strip punct).

We implement both and apply whichever the dataset uses.
"""
from __future__ import annotations

import logging
import re
import string
from dataclasses import dataclass

log = logging.getLogger(__name__)


_PUNCT_RE = re.compile(f"[{re.escape(string.punctuation)}]")
_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")


def normalize_vqa(s: str) -> str:
    """Standard VQA normalization: lowercase, strip articles+punct, collapse spaces."""
    s = s.lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _ARTICLES_RE.sub(" ", s)
    s = " ".join(s.split())
    return s


def vqa_v2_score(prediction: str, gt_answers: list[str]) -> float:
    """VQAv2 grader: min(matches / 3, 1.0)."""
    pred_norm = normalize_vqa(prediction)
    matches = sum(1 for gt in gt_answers if normalize_vqa(gt) == pred_norm)
    return min(matches / 3.0, 1.0)


def gqa_score(prediction: str, gt_answer: str) -> float:
    return 1.0 if normalize_vqa(prediction) == normalize_vqa(gt_answer) else 0.0


@dataclass
class EvalResult:
    n_examples: int
    accuracy: float  # mean score
    by_category: dict[str, float] | None = None


def evaluate_vqa_v2(model, examples: list[dict], max_examples: int | None = None) -> EvalResult:
    """examples: list of {image, question, answers}.
    model: anything with `.generate(image, prompt, max_new_tokens)`.
    """
    from tqdm import tqdm

    if max_examples:
        examples = examples[:max_examples]

    scores = []
    by_cat: dict[str, list[float]] = {}
    for ex in tqdm(examples, desc="VQAv2 eval"):
        pred = model.generate(
            ex["image"], ex["question"], max_new_tokens=24, do_sample=False
        )
        score = vqa_v2_score(pred, ex["answers"])
        scores.append(score)
        cat = ex.get("question_type", "all")
        by_cat.setdefault(cat, []).append(score)

    return EvalResult(
        n_examples=len(scores),
        accuracy=sum(scores) / max(len(scores), 1),
        by_category={k: sum(v) / len(v) for k, v in by_cat.items()},
    )
