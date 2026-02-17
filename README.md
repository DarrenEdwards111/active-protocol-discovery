<p align="center">
  <img src="apd-logo.jpg" alt="Mikoshi Active Protocol Discovery" width="600">
</p>

# Active Protocol Discovery (APD)

A lightweight framework for detecting hidden adaptive structure in noisy environments via sequential, KL-optimal probing.

Rather than passively monitoring a system, APD:

1. Injects structured perturbations
2. Measures statistical deviation
3. Updates posterior belief
4. Selects next probe to maximise expected evidence gain

## Applications

- Hidden adversarial agent detection
- Distributed AI system discovery
- Network bot detection
- Adaptive environment testing
- Multi-agent inference research

## Installation

```bash
pip install git+https://github.com/DarrenEdwards111/active-protocol-discovery.git
```

Or clone and install locally:

```bash
git clone https://github.com/DarrenEdwards111/active-protocol-discovery.git
cd active-protocol-discovery
pip install -e ".[dev]"
```

## Quick Start

```python
from apd import APDEngine, GaussianWorld, KLOptimalPolicy, WaldSPRT

world = GaussianWorld(sigma=1.0, adaptive=True)
policy = KLOptimalPolicy(probes=[0.2, 0.5, 1.0])
sprt = WaldSPRT(alpha=0.01, beta=0.01)

engine = APDEngine(world, policy, sprt)
result = engine.run(max_steps=1000)

print(f"Decision: {'Adaptive' if result.decision else 'Null'}")
print(f"Steps: {result.steps}")
```

## How It Works

APD implements a sequential likelihood ratio framework (Wald SPRT) with probe selection based on KL-separability. At each step the engine:

1. **Selects a probe** — the perturbation that maximises expected KL divergence between H0 and H1
2. **Observes the response** — samples from the true environment
3. **Updates the log-likelihood ratio** — accumulates evidence for/against adaptive structure
4. **Decides or continues** — stops when the SPRT threshold is crossed

## Demo Results

Results from `experiments/gaussian_demo.py` (1000 trials, σ=1.0, α=β=0.01):

### H1: Adaptive world (adversary present)

| Method                    | Mean Steps | Detection Rate | Undecided |
|---------------------------|------------|----------------|-----------|
| Passive (u=0)             | 10000.0    | 0.000          | 1.000     |
| Weak beacon (u=0.2)       | 231.5      | 0.987          | 0.000     |
| Strong beacon (u=1.0)     | 10.4       | 0.996          | 0.000     |
| APD KL-optimal {0.2, 1.0} | 10.4       | 0.996          | 0.000     |
| APD KL-optimal {0.1..1.5} | 5.1        | 0.993          | 0.000     |

### H0: Null world (false positive check)

| Method                    | Mean Steps | False Positive Rate | Correct Null |
|---------------------------|------------|---------------------|--------------|
| Strong beacon (u=1.0)     | 10.5       | 0.003               | 0.997        |
| APD KL-optimal {0.1..1.5} | 5.1        | 0.002               | 0.998        |

**Key takeaway:** APD automatically selects the strongest available probe (u=1.5), achieving 2× faster detection than a strong fixed beacon while maintaining false positive rates below α.

## Theory

### Wald SPRT

The Sequential Probability Ratio Test accumulates log-likelihood ratios until crossing a decision boundary:

- **Log-likelihood ratio:** Λ(y|u) = log p(y|H₁,u) − log p(y|H₀)
- **Upper threshold (accept H₁):** A = log((1−β)/α)
- **Lower threshold (accept H₀):** B = log(β/(1−α))

### KL Divergence as Probe Quality

For the Gaussian case, the KL divergence between H₁ and H₀ given probe u is:

D(u) = μ(u)² / (2σ²)

This quantifies how much information a single observation provides about the hypothesis. APD selects the probe maximising D(u), achieving the fastest possible sequential detection.

### Expected Sample Complexity

Under H₁, Wald's approximation gives:

E[N|H₁] ≈ ((1−β)·A + β·B) / D(u)

Stronger probes (higher D(u)) yield fewer required samples.

## Package Structure

```
apd/
├── models.py    — World models (Gaussian, AdaptiveAgent, Network)
├── policy.py    — Probe selection (KL-optimal, fixed, random, ε-greedy)
├── sprt.py      — Wald Sequential Probability Ratio Test
├── apd.py       — Main APD engine
└── utils.py     — Statistics helpers
```

## Running Experiments

```bash
# Gaussian demo (the killer demo)
python -m experiments.gaussian_demo

# Adversarial agent in multi-dimensional space
python -m experiments.adversarial_agent_demo

# Full benchmark sweep (outputs CSV)
python -m experiments.benchmark > results.csv
```

## Tests

```bash
pytest tests/ -v
```

## Citation

```bibtex
@software{apd2026,
  author = {Mikoshi Ltd},
  title = {Active Protocol Discovery},
  year = {2026},
  url = {https://github.com/DarrenEdwards111/active-protocol-discovery}
}
```

## Licence

MIT — Mikoshi Ltd, 2026
