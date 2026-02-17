"""Gaussian closed-form demo — the killer demo for Active Protocol Discovery.

Compares passive monitoring, fixed beacon, and KL-optimal APD for detecting
an adaptive shift in a Gaussian observation model.
"""

from __future__ import annotations

import sys
import numpy as np

from apd.models import GaussianWorld
from apd.policy import KLOptimalPolicy, FixedPolicy, RandomPolicy
from apd.sprt import WaldSPRT
from apd.apd import APDEngine
from apd.utils import format_results_table


def run_trials(
    world_cls,
    policy_cls,
    sprt_kwargs: dict,
    n_trials: int = 1000,
    max_steps: int = 10000,
    **world_kwargs,
) -> dict:
    """Run multiple trials and collect statistics."""
    steps_list = []
    decisions = []

    for trial in range(n_trials):
        world = world_cls(**world_kwargs)
        sprt = WaldSPRT(**sprt_kwargs)
        policy = policy_cls()
        engine = APDEngine(world, policy, sprt)
        result = engine.run(max_steps=max_steps, seed=trial)
        steps_list.append(result.steps)
        decisions.append(result.decision)

    h1_rate = sum(1 for d in decisions if d == 1) / n_trials
    h0_rate = sum(1 for d in decisions if d == 0) / n_trials
    undecided = sum(1 for d in decisions if d is None) / n_trials
    mean_steps = np.mean(steps_list)

    return {
        "mean_steps": mean_steps,
        "detection_rate": h1_rate,
        "null_rate": h0_rate,
        "undecided_rate": undecided,
    }


def main() -> None:
    """Run the full Gaussian APD demo."""
    alpha = 0.01
    beta = 0.01
    sigma = 1.0
    n_trials = 1000
    max_steps = 10000
    sprt_kwargs = {"alpha": alpha, "beta": beta}

    print("=" * 70)
    print("Active Protocol Discovery — Gaussian Demo")
    print("=" * 70)
    print(f"\nParameters: sigma={sigma}, alpha={alpha}, beta={beta}")
    print(f"Trials per method: {n_trials}, max steps: {max_steps}")
    print()

    # --- H1: Adaptive world (adversary present) ---
    print("─" * 70)
    print("H1: ADAPTIVE WORLD (adversary present)")
    print("─" * 70)

    methods_h1 = []

    # 1. Passive monitoring (u=0)
    stats = run_trials(
        GaussianWorld,
        lambda: FixedPolicy(0.0),
        sprt_kwargs,
        n_trials=n_trials,
        max_steps=max_steps,
        sigma=sigma,
        adaptive=True,
    )
    methods_h1.append({
        "Method": "Passive (u=0)",
        "Mean Steps": f"{stats['mean_steps']:.1f}",
        "Detection Rate": f"{stats['detection_rate']:.3f}",
        "Undecided": f"{stats['undecided_rate']:.3f}",
    })

    # 2. Weak beacon (u=0.2)
    stats = run_trials(
        GaussianWorld,
        lambda: FixedPolicy(0.2),
        sprt_kwargs,
        n_trials=n_trials,
        max_steps=max_steps,
        sigma=sigma,
        adaptive=True,
    )
    methods_h1.append({
        "Method": "Weak beacon (u=0.2)",
        "Mean Steps": f"{stats['mean_steps']:.1f}",
        "Detection Rate": f"{stats['detection_rate']:.3f}",
        "Undecided": f"{stats['undecided_rate']:.3f}",
    })

    # 3. Strong beacon (u=1.0)
    stats = run_trials(
        GaussianWorld,
        lambda: FixedPolicy(1.0),
        sprt_kwargs,
        n_trials=n_trials,
        max_steps=max_steps,
        sigma=sigma,
        adaptive=True,
    )
    methods_h1.append({
        "Method": "Strong beacon (u=1.0)",
        "Mean Steps": f"{stats['mean_steps']:.1f}",
        "Detection Rate": f"{stats['detection_rate']:.3f}",
        "Undecided": f"{stats['undecided_rate']:.3f}",
    })
    strong_beacon_steps = float(stats["mean_steps"])

    # 4. APD with KL-optimal (probes: 0.2, 1.0)
    stats = run_trials(
        GaussianWorld,
        lambda: KLOptimalPolicy(probes=[0.2, 1.0], sigma=sigma),
        sprt_kwargs,
        n_trials=n_trials,
        max_steps=max_steps,
        sigma=sigma,
        adaptive=True,
    )
    methods_h1.append({
        "Method": "APD KL-optimal {0.2, 1.0}",
        "Mean Steps": f"{stats['mean_steps']:.1f}",
        "Detection Rate": f"{stats['detection_rate']:.3f}",
        "Undecided": f"{stats['undecided_rate']:.3f}",
    })

    # 5. APD with larger probe set
    stats = run_trials(
        GaussianWorld,
        lambda: KLOptimalPolicy(
            probes=[0.1, 0.3, 0.5, 0.8, 1.0, 1.5], sigma=sigma
        ),
        sprt_kwargs,
        n_trials=n_trials,
        max_steps=max_steps,
        sigma=sigma,
        adaptive=True,
    )
    methods_h1.append({
        "Method": "APD KL-optimal {0.1..1.5}",
        "Mean Steps": f"{stats['mean_steps']:.1f}",
        "Detection Rate": f"{stats['detection_rate']:.3f}",
        "Undecided": f"{stats['undecided_rate']:.3f}",
    })
    apd_full_steps = float(stats["mean_steps"])

    print(format_results_table(methods_h1))
    print()

    # --- H0: Null world (no adversary) ---
    print("─" * 70)
    print("H0: NULL WORLD (no adversary) — false positive check")
    print("─" * 70)

    methods_h0 = []

    for label, policy_fn in [
        ("Strong beacon (u=1.0)", lambda: FixedPolicy(1.0)),
        ("APD KL-optimal {0.1..1.5}", lambda: KLOptimalPolicy(
            probes=[0.1, 0.3, 0.5, 0.8, 1.0, 1.5], sigma=sigma
        )),
    ]:
        stats = run_trials(
            GaussianWorld,
            policy_fn,
            sprt_kwargs,
            n_trials=n_trials,
            max_steps=max_steps,
            sigma=sigma,
            adaptive=False,
        )
        methods_h0.append({
            "Method": label,
            "Mean Steps": f"{stats['mean_steps']:.1f}",
            "False Positive Rate": f"{stats['detection_rate']:.4f}",
            "Correct Null": f"{stats['null_rate']:.3f}",
        })

    print(format_results_table(methods_h0))
    print()

    print("─" * 70)
    print(f"APD selects u=1.5 (strongest probe) automatically from the candidate set.")
    print(f"Expected false positive rate ≈ alpha = {alpha}")
    print("=" * 70)


if __name__ == "__main__":
    main()
