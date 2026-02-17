"""Active Protocol Discovery — detect hidden adaptive structure via sequential KL-optimal probing."""

from apd.models import GaussianWorld, AdaptiveAgentWorld, NetworkWorld
from apd.policy import KLOptimalPolicy, FixedPolicy, RandomPolicy, EpsilonGreedyPolicy
from apd.sprt import WaldSPRT
from apd.apd import APDEngine, APDResult

__version__ = "0.1.0"

__all__ = [
    "GaussianWorld",
    "AdaptiveAgentWorld",
    "NetworkWorld",
    "KLOptimalPolicy",
    "FixedPolicy",
    "RandomPolicy",
    "EpsilonGreedyPolicy",
    "WaldSPRT",
    "APDEngine",
    "APDResult",
]
