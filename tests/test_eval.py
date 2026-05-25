"""Tests for VQA graders."""
from __future__ import annotations

from visiontalk.eval import gqa_score, normalize_vqa, vqa_v2_score


def test_normalize_strips_punctuation_and_articles():
    assert normalize_vqa("The cat, in the hat.") == "cat in hat"
    assert normalize_vqa("A dog!") == "dog"
    assert normalize_vqa("YES.") == "yes"


def test_vqa_v2_three_or_more_matches_perfect():
    score = vqa_v2_score("yes", ["yes", "yes", "yes", "no"])
    assert score == 1.0


def test_vqa_v2_one_match_partial():
    score = vqa_v2_score("blue", ["blue", "green", "green", "green"])
    assert abs(score - (1 / 3)) < 1e-6


def test_vqa_v2_no_match_zero():
    score = vqa_v2_score("purple", ["red", "blue", "green"])
    assert score == 0.0


def test_vqa_v2_normalizes_pred_and_gt():
    score = vqa_v2_score("The Cat.", ["a cat", "a cat", "a cat"])
    assert score == 1.0


def test_gqa_strict_normalized_match():
    assert gqa_score("Yes", "yes") == 1.0
    assert gqa_score("The cat", "a cat") == 1.0
    assert gqa_score("dog", "cat") == 0.0
