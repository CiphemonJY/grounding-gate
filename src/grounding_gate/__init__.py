"""Grounding Gate — zero-token structural verifier for agent loops.

One choke point at the submit boundary enforces:
  G (grounding): terminal claims require a qualifying observation this turn
     (novel AND relevant AND consequence-tier-correct).
  B (budget): grounded steps refill rope, pure reasoning decrements; exhaustion
     halts to {qualifying call | typed `unverified` terminal}.

Hash/set/integer operations only — no LLM calls anywhere in the gate.
"""

from .state import PRESETS, GateState, extract_identifiers, normalize
from .classifier import classify_observation
from .boundary import ACCEPT, LEGAL_NEXT, REJECT, boundary_check, turn_loop

__version__ = "0.3.0"

__all__ = [
    "ACCEPT",
    "LEGAL_NEXT",
    "PRESETS",
    "REJECT",
    "GateState",
    "boundary_check",
    "classify_observation",
    "extract_identifiers",
    "normalize",
    "turn_loop",
    "__version__",
]
