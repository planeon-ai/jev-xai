"""Rank correlation with tie handling. SciPy is optional and is not required."""

from __future__ import annotations

import math
from collections import Counter


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(values):
        end = cursor
        while end + 1 < len(values) and values[order[end + 1]] == values[order[cursor]]:
            end += 1
        average = (cursor + end) / 2 + 1
        for position in range(cursor, end + 1):
            ranks[order[position]] = average
        cursor = end + 1
    return ranks


def _pearson(left: list[float], right: list[float]) -> float:
    count = len(left)
    if count == 0:
        return 0.0
    mean_left = sum(left) / count
    mean_right = sum(right) / count
    numerator = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right, strict=True))
    spread_left = math.sqrt(sum((a - mean_left) ** 2 for a in left))
    spread_right = math.sqrt(sum((b - mean_right) ** 2 for b in right))
    if spread_left == 0 or spread_right == 0:
        return 0.0
    return numerator / (spread_left * spread_right)


def spearman(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    return _pearson(_ranks(left), _ranks(right))


def _tie_pairs(values: list[float]) -> float:
    total = 0.0
    for count in Counter(values).values():
        total += count * (count - 1) / 2
    return total


def kendall_tau_b(left: list[float], right: list[float]) -> float:
    """Kendall tau-b, including the tie correction."""

    count = len(left)
    if count < 2 or count != len(right):
        return 0.0
    concordant = 0
    discordant = 0
    for i in range(count):
        for j in range(i + 1, count):
            dx = left[i] - left[j]
            dy = right[i] - right[j]
            if dx == 0 or dy == 0:
                continue
            if dx * dy > 0:
                concordant += 1
            else:
                discordant += 1
    n0 = count * (count - 1) / 2
    n1 = _tie_pairs(left)
    n2 = _tie_pairs(right)
    denom = math.sqrt((n0 - n1) * (n0 - n2))
    if denom == 0:
        return 0.0
    return (concordant - discordant) / denom


def rank_correlation(left: list[float], right: list[float], metric: str) -> float:
    """In-house Spearman or Kendall tau-b. SciPy is not required at runtime."""

    if metric == "kendall":
        return kendall_tau_b(left, right)
    return spearman(left, right)


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def sample_stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)
