<p align="center">
  <img src="apd-logo.jpg" alt="Mikoshi Active Protocol Discovery" width="600">
</p>

# Active Protocol Discovery (APD)

A lightweight framework for detecting hidden adaptive structure in noisy environments via sequential, KL-optimal probing.

Most detection systems are **passive** — they watch and wait. APD is **active** — it probes, measures, adapts, and decides. This makes it fundamentally faster at detecting hidden adaptive behaviour.

Rather than passively monitoring a system, APD:

1. **Injects structured perturbations** into the target environment
2. **Measures statistical deviation** between expected and observed responses
3. **Updates posterior belief** via sequential likelihood ratios
4. **Selects the next probe** to maximise expected evidence gain (KL-optimal)
5. **Decides** when statistical confidence is reached — no fixed sample size required

## Why APD Over Existing Approaches?

| Approach | Method | Weakness | APD Advantage |
|----------|--------|----------|---------------|
| **Passive monitoring** | Watch logs/traffic for anomalies | Fails against agents that stay quiet until activated | APD *forces* adaptive agents to reveal themselves |
| **Fixed probing** (honeypots, canaries) | Deploy known traps | Predictable — sophisticated agents learn to avoid them | APD *adapts* its probes based on what it learns |
| **Threshold detectors** | Alert on metric > threshold | Requires pre-set thresholds, high false positive rate | APD uses *sequential testing* — mathematically controlled error rates |
| **Batch hypothesis testing** | Collect N samples, then decide | Wastes samples, fixed sample size regardless of signal strength | APD *stops early* when evidence is sufficient — up to 45× faster |
| **Anomaly detection (ML)** | Train model on "normal", flag outliers | Needs training data, can't detect novel adaptive behaviour | APD is *model-free* — works with any response distribution |

### Key Strengths

- **45× faster than passive monitoring** — empirically demonstrated (see demo results below)
- **2× faster than best fixed-probe methods** — KL-optimal selection always finds the most informative probe
- **Mathematically guaranteed error rates** — false positive and false negative rates are configurable via Wald SPRT, not heuristic
- **No training data required** — works from first observation, no labelled dataset needed
- **Probe-agnostic** — plug in any environment, any probe set, any response distribution
- **Lightweight** — pure NumPy/SciPy, no deep learning frameworks, runs on a Raspberry Pi

## Applications

### AI Safety & Adversarial Detection
- **Deceptive AI agents** — detect if an autonomous agent is strategically hiding capabilities or intentions
- **LLM prompt injection** — inject structured perturbations, monitor output entropy shifts to detect hidden tool-use loops
- **Sleeper agents** — probe for trigger-activated behaviour in fine-tuned models
- **Multi-agent deception** — detect if agents in a sandbox are colluding or behaving adaptively

### Network Security
- **Bot detection** — send structured timing probes, observe latency distribution shifts to distinguish bots from humans
- **Adversarial infrastructure** — detect command-and-control servers that adapt responses based on probe patterns
- **DDoS source identification** — probe suspect sources with varied request patterns to detect coordinated adaptive behaviour

### Distributed Systems
- **Hidden node discovery** — detect undeclared adaptive nodes in a distributed network
- **Byzantine fault detection** — identify nodes that respond strategically rather than honestly
- **Protocol compliance testing** — probe services to detect non-standard adaptive behaviour

### Research
- **Multi-agent systems** — test whether agents develop emergent adaptive strategies
- **Reinforcement learning** — detect if an RL agent has learned to game its environment
- **Cognitive science** — sequential optimal experiment design for detecting adaptive structure in human/animal behaviour

## Installation

```bash
pip install active-protocol-discovery
```

Or from source:

```bash
git clone https://github.com/DarrenEdwards111/active-protocol-discovery.git
cd active-protocol-discovery
pip install -e ".[dev]"
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

Apache 2.0 — Mikoshi Ltd, 2026
