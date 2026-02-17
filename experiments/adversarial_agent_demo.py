"""Adversarial agent demo — APD discovers hidden adaptive direction."""

from __future__ import annotations

import numpy as np

from apd.models import AdaptiveAgentWorld
from apd.policy import KLOptimalPolicy, RandomPolicy
from apd.sprt import WaldSPRT
from apd.apd import APDEngine


def main() -> None:
    """Run the adversarial agent demo."""
    dim = 5
    sigma = 1.0
    alpha = 0.01
    beta = 0.01
    n_trials = 200
    max_steps = 5000

    print("=" * 70)
    print("Adversarial Agent Demo — Multi-dimensional APD")
    print("=" * 70)
    print(f"\nDimensions: {dim}, sigma={sigma}, alpha={alpha}, beta={beta}")
    print(f"Trials: {n_trials}")

    # Generate candidate probes: unit vectors along each axis + some diagonals
    probes = []
    for i in range(dim):
        e = np.zeros(dim)
        e[i] = 1.0
        probes.append(e)
    # Add some diagonal probes
    diag = np.ones(dim) / np.sqrt(dim)
    probes.append(diag)

    # Fixed hidden direction for reproducibility
    rng = np.random.default_rng(42)
    hidden = rng.standard_normal(dim)
    hidden /= np.linalg.norm(hidden)

    print(f"Hidden direction: [{', '.join(f'{x:.2f}' for x in hidden)}]")
    print()

    # Random probing
    random_steps = []
    random_detections = 0
    for trial in range(n_trials):
        world = AdaptiveAgentWorld(
            dim=dim, sigma=sigma, adaptive=True,
            hidden_direction=hidden, sensitivity=1.0,
        )
        policy = RandomPolicy(probes)
        sprt = WaldSPRT(alpha=alpha, beta=beta)
        engine = APDEngine(world, policy, sprt, mu_h1_fn=world.mu)
        result = engine.run(max_steps=max_steps, seed=trial)
        random_steps.append(result.steps)
        if result.decision == 1:
            random_detections += 1

    print(f"Random probing:     mean steps = {np.mean(random_steps):.1f}, "
          f"detection = {random_detections/n_trials:.3f}")

    # KL-optimal probing
    kl_steps = []
    kl_detections = 0
    for trial in range(n_trials):
        world = AdaptiveAgentWorld(
            dim=dim, sigma=sigma, adaptive=True,
            hidden_direction=hidden, sensitivity=1.0,
        )
        policy = KLOptimalPolicy(probes, sigma=sigma, mu_fn=world.mu)
        sprt = WaldSPRT(alpha=alpha, beta=beta)
        engine = APDEngine(world, policy, sprt, mu_h1_fn=world.mu)
        result = engine.run(max_steps=max_steps, seed=trial)
        kl_steps.append(result.steps)
        if result.decision == 1:
            kl_detections += 1

    print(f"KL-optimal probing: mean steps = {np.mean(kl_steps):.1f}, "
          f"detection = {kl_detections/n_trials:.3f}")

    # Show which probe the KL policy selects
    world = AdaptiveAgentWorld(
        dim=dim, sigma=sigma, adaptive=True,
        hidden_direction=hidden, sensitivity=1.0,
    )
    policy = KLOptimalPolicy(probes, sigma=sigma, mu_fn=world.mu)
    scores = policy.kl_scores()
    print("\nKL scores per probe:")
    for p, s in scores.items():
        label = np.array2string(np.asarray(p), precision=2)
        print(f"  {label}: {s:.4f}")

    best = max(scores, key=scores.get)
    print(f"\nBest probe: [{', '.join(f'{x:.2f}' for x in np.asarray(best))}]")
    print(f"(Projection onto hidden direction: "
          f"{abs(np.dot(np.asarray(best), hidden)):.3f})")

    if np.mean(kl_steps) < np.mean(random_steps):
        speedup = np.mean(random_steps) / np.mean(kl_steps)
        print(f"\nKL-optimal is {speedup:.1f}x faster than random probing.")

    print("=" * 70)


if __name__ == "__main__":
    main()
