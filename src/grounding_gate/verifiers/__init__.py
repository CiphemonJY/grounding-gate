"""Optional ``verify_with`` verifier tier — spec module 5 (declarative rails).

This subpackage is the escalation tier the design's leak audit names for
"semantic misreads / relevance-spoofing / adversarial self-deception": failures
the ZERO-LLM floor deliberately does not chase. It is wired at the submit
boundary (``boundary.boundary_check`` / ``turn_loop``) as an escalation that can
ONLY downgrade a structural ACCEPT to the typed ``unverified`` path — it can
never upgrade a structural REJECT. The floor runs first and independently; a
verifier is a no-op unless one is supplied.

This base module is STDLIB-ONLY (it imports only ``typing``) so that
``import grounding_gate`` and ``from grounding_gate.verifiers import ...`` pull
in zero third-party dependencies. The optional reference implementation that
talks to an LLM lives in :mod:`grounding_gate.verifiers.llm` and imports its SDK
lazily; it is NOT imported here, so the SACRED zero-LLM story of the floor holds
for anything reachable from this package's ``__init__``.

Typing is used here (unlike the annotation-free floor) because that is the
idiom for the optional subpackage; the ``Verifier`` Protocol documents the one
method a custom verifier must provide.
"""

from typing import Protocol, runtime_checkable

# Default score-granularity threshold. A verifier confidence STRICTLY below this
# downgrades a structural ACCEPT to the typed ``unverified`` path. Mirrored by
# ``GateState.verify_threshold`` (state.py keeps the literal 0.5 rather than
# importing this optional module, so the floor never reaches into the tier).
GRANULARITY = 0.5


@runtime_checkable
class Verifier(Protocol):
    """Structural contract for a ``verify_with`` verifier.

    ``score(claim, observations, criteria)`` returns one of:
      * a ``float`` in ``[0, 1]`` — a single confidence for the whole claim;
      * a ``dict`` ``{criterion_name: float}`` — one confidence per criterion
        (the weakest governs; see :func:`aggregate`);
      * ``None`` — ABSTAIN (the floor's ACCEPT is left standing).

    ``claim`` is the terminal attempt dict (``{"claim_type", "content"}``),
    ``observations`` is the list of this turn's qualifying observations the
    wiring layer retained (``GateState.turn_observations``), and ``criteria`` is
    the tuple of ``(name, question)`` pairs from :func:`criteria_for`.
    """

    def score(self, claim, observations, criteria):  # pragma: no cover - protocol
        ...


# Criteria decompositions, keyed off the consequence tier the claim is making.
# Each is a tuple of (name, natural-language question) pairs; a verifier scores
# each question independently (DECOMPOSITION) and the WEAKEST sub-score governs.
_COMPLETION_CRITERIA = (
    ("effect_shown",
     "Do the cited observations show the claimed change actually took effect?"),
    ("no_overreach",
     "Does the claim stay within what the cited observations support, "
     "without asserting more than was observed?"),
)
_ASSERTION_CRITERIA = (
    ("claim_supported",
     "Do the cited observations support the asserted fact?"),
    ("no_overreach",
     "Does the claim stay within what the cited observations support, "
     "without asserting more than was observed?"),
)


def criteria_for(claim_type, strict_g=False):
    """Return the ``(name, question)`` criteria tuple for a claim tier.

    ``completion`` -> effect-shown + no-overreach; ``assertion`` ->
    claim-supported + no-overreach. Under the skipper preset's strict-G an
    ``assertion`` is held to the same VERIFIED (completion-tier) bar the floor
    already imposes on it, so it routes to the completion criteria. Anything
    else (including ``unverified`` / ``none``, which the boundary never escalates
    anyway) returns ``()`` — no criteria, so a verifier has nothing to score.
    """
    if claim_type == "completion":
        return _COMPLETION_CRITERIA
    if claim_type == "assertion":
        return _COMPLETION_CRITERIA if strict_g else _ASSERTION_CRITERIA
    return ()


def aggregate(score):
    """Collapse a verifier's ``score`` return into a single confidence or None.

    ``dict`` -> ``min`` of its values (the WEAKEST sub-criterion governs, so a
    claim that fails any one decomposed check is downgraded). An EMPTY dict or
    ``None`` -> ``None`` (ABSTAIN: the floor's ACCEPT stands). A bare ``float``
    passes through unchanged. This is the ONLY place the per-criterion scores
    are combined, so the "weakest link" policy lives in one spot.
    """
    if score is None:
        return None
    if isinstance(score, dict):
        if not score:
            return None
        return min(score.values())
    return score


class StubVerifier:
    """Deterministic, network-free verifier for tests and pinning.

    Either returns a fixed ``score`` for every claim, or delegates to ``rule``
    (``rule(claim, observations, criteria) -> score``) for exact control. It
    imports nothing beyond this module, so it exercises the whole ``verify_with``
    wiring with zero third-party dependencies and zero network — the hermetic
    stand-in the SACRED test suite uses in place of :class:`~.llm.LLMVerifier`.
    """

    def __init__(self, score=1.0, rule=None):
        self._score = score
        self._rule = rule

    def score(self, claim, observations, criteria):
        if self._rule is not None:
            return self._rule(claim, observations, criteria)
        return self._score


__all__ = ["Verifier", "StubVerifier", "criteria_for", "aggregate", "GRANULARITY"]
