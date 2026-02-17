"""Benchmark: passive vs beacon vs APD across parameter sweeps."""

from __future__ import annotations

import csv
import sys
import numpy as np

from apd.models import GaussianWorld
from apd.policy import KLOptimalPolicy, FixedPolicy
from apd.sprt import WaldSPRT
from apd.apd import APDEngine


def benchmark_single(
    sigma: float,
    probe_strength: float,
    method: str,
    probes: list[float],
    alpha: float = 0.01,
    beta: float = 0.01,
    n_trials: int = 200,
    max_steps: int = 10000,
) -> dict:
    """Run benchmark for a single configuration."""
    steps_list = []
    detections = 0

    for trial in range(n_trials):
        world = GaussianWorld(sigma=sigma, adaptive=True)
        sprt = WaldSPRT(alpha=alpha, beta=beta)

        if method == "passive":
            policy = FixedPolicy(0.0)
        elif method == "beacon":
            policy = FixedPolicy(probe_strength)
        elif method == "apd":
            policy = KLOptimalPolicy(probes=probes, sigma=sigma)
        else:
            raise ValueError(f"Unknown method: {method}")

        engine = APDEngine(world, policy, sprt)
        result = engine.run(max_steps=max_steps, seed=trial)
        steps_list.append(result.steps)
        if result.decision == 1:
            detections += 1

    return {
        "sigma": sigma,
        "probe_strength": probe_strength,
        "method": method,
        "num_probes": len(probes),
        "mean_steps": np.mean(steps_list),
        "median_steps": np.median(steps_list),
        "detection_rate": detections / n_trials,
    }


def main() -> None:
    """Run full benchmark sweep and output CSV."""
    sigmas = [0.5, 1.0, 2.0]
    probe_strengths = [0.2, 0.5, 1.0]
    probe_sets = {
        3: [0.2, 0.5, 1.0],
        6: [0.1, 0.3, 0.5, 0.8, 1.0, 1.5],
    }

    writer = csv.DictWriter(
        sys.stdout,
        fieldnames=[
            "sigma", "probe_strength", "method", "num_probes",
            "mean_steps", "median_steps", "detection_rate",
        ],
    )
    writer.writeheader()

    for sigma in sigmas:
        for ps in probe_strengths:
            # Passive
            row = benchmark_single(sigma, ps, "passive", [0.0])
            writer.writerow({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in row.items()})

            # Beacon
            row = benchmark_single(sigma, ps, "beacon", [ps])
            writer.writerow({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in row.items()})

            # APD with different probe set sizes
            for n, probes in probe_sets.items():
                row = benchmark_single(sigma, ps, "apd", probes)
                writer.writerow({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in row.items()})

    sys.stdout.flush()


if __name__ == "__main__":
    main()
