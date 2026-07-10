"""Gate state, model-class presets, and the normalization helpers.

Spec modules 1 (state container) and 6 (per-model-class presets): fleet
variance is absorbed as integers, not prose.
"""

import re
from dataclasses import dataclass, field

# Per-model-class presets. Two documented agent failure modes get their own
# tuning: "skipper" models emit confident terminals without observing reality
# (strict grounding), "diverger" models reason in closed context until a
# confident wrong answer ships (small budget, small refill — starves loops).
PRESETS = {
    "skipper":  {"CAP": 5, "REFILL": 2, "strict_g": True},
    "diverger": {"CAP": 4, "REFILL": 1, "strict_g": False},
    "default":  {"CAP": 6, "REFILL": 2, "strict_g": False},
}


@dataclass
class GateState:
    budget: int
    cap: int
    refill: int
    # strict G (skipper preset): even assertions require the verified tier
    strict_g: bool = False
    # novelty
    recent_result_hashes: set = field(default_factory=set)
    novelty_exempt: set = field(default_factory=set)
    # relevance
    claim_surface: set = field(default_factory=set)
    # consequence
    last_mutation_step: int = 0   # 0 = no mutation has EVER occurred
    current_step: int = 0
    # per-turn latches
    grounded_this_turn: bool = False
    verified_this_turn: bool = False
    # declarative rails
    verified_signals: set = field(default_factory=set)
    goal_predicates: list = field(default_factory=list)
    halted: bool = False

    @classmethod
    def for_model_class(cls, model_class="default", **kw):
        p = PRESETS[model_class]
        kw.setdefault("strict_g", p["strict_g"])
        return cls(budget=p["CAP"], cap=p["CAP"], refill=p["REFILL"], **kw)


def normalize(text):
    """Strip nondeterminism before hashing. Reference: ISO timestamps + hex ids.

    Real deployments extend this per-tool; too-weak normalization means novelty
    never fires on noisy tools (the no-op defense weakens).
    """
    text = re.sub(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}\S*", "<TS>", str(text))
    return re.sub(r"\b[0-9a-f]{8,}\b", "<HEX>", text)


def extract_identifiers(args, result):
    """Conservative token extraction for the relevance check.

    Over-extraction leaks relevance; under-extraction false-rejects
    cross-cutting work (open risk, flagged in docs/module-2-classifier.md).
    """
    return set(re.findall(r"[\w.\-/]+", f"{args} {result}"))
