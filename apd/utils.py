"""Utility functions for Active Protocol Discovery."""

from __future__ import annotations

import math
from typing import Sequence


def kl_gaussian(mu: float, sigma: float) -> float:
    """KL divergence D_KL(N(mu,σ²) || N(0,σ²)) = μ²/(2σ²)."""
    return mu ** 2 / (2.0 * sigma ** 2)


def sprt_thresholds(alpha: float, beta: float) -> tuple[float, float]:
    """Compute Wald SPRT thresholds.

    Returns
    -------
    (upper, lower) : tuple[float, float]
        upper = log((1-β)/α), lower = log(β/(1-α))
    """
    upper = math.log((1.0 - beta) / alpha)
    lower = math.log(beta / (1.0 - alpha))
    return upper, lower


def expected_samples_gaussian(mu: float, sigma: float, alpha: float, beta: float) -> float:
    """Approximate expected number of samples for Wald SPRT under H1.

    Uses Wald's approximation:
        E[N|H1] ≈ ((1-β)log((1-β)/α) + β·log(β/(1-α))) / D_KL

    where D_KL = μ²/(2σ²).
    """
    d_kl = kl_gaussian(mu, sigma)
    if d_kl == 0:
        return float("inf")
    upper, lower = sprt_thresholds(alpha, beta)
    return ((1.0 - beta) * upper + beta * lower) / d_kl


def format_results_table(
    rows: Sequence[dict],
    columns: Sequence[str] | None = None,
) -> str:
    """Format a list of dicts as a markdown table."""
    if not rows:
        return ""
    columns = columns or list(rows[0].keys())
    widths = {c: max(len(c), max(len(str(r.get(c, ""))) for r in rows)) for c in columns}

    header = "| " + " | ".join(c.ljust(widths[c]) for c in columns) + " |"
    sep = "|-" + "-|-".join("-" * widths[c] for c in columns) + "-|"
    lines = [header, sep]
    for r in rows:
        line = "| " + " | ".join(str(r.get(c, "")).ljust(widths[c]) for c in columns) + " |"
        lines.append(line)
    return "\n".join(lines)
