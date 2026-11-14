"""Wilson score interval for binomial proportions."""

import math


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    radicand = max(0.0, p * (1 - p) / n + z * z / (4 * n * n))
    half = (z * math.sqrt(radicand)) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return (lo, hi)


def wilson_lo(successes: int, n: int, z: float = 1.96) -> float:
    return wilson_ci(successes, n, z)[0]


def wilson_hi(successes: int, n: int, z: float = 1.96) -> float:
    return wilson_ci(successes, n, z)[1]
