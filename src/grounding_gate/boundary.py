"""Submit-boundary choke point and reference turn loop — spec module 4.

``boundary_check`` must be the SOLE path to any terminal output. ``turn_loop``
is the reference wiring that drives a scripted agent through the gate — use it
as the template for integrating the gate into a real agent loop (and in tests
and demos, where scripted steps make behavior deterministic).

Four corrections survived adversarial review of the original draft — all in
``turn_loop``, between the individually-passing acceptance cases:
  (C1) grounding flags are LATCHES within a turn, not last-call assignments —
       otherwise a later non-completion-grade call overwrites a valid
       verification back to false (false-rejects legitimate work).
  (C2) halt is cleared ONLY by a QUALIFYING observation — clearing on any tool
       call lets a halted model escape via a novelty-defeated no-op read.
  (C4) refused reasoning still decrements budget and surfaces ``legal_next``
       (re-prompt) — otherwise a halted spinner livelocks for free.

Optional verify_with tier (spec module 5). ``boundary_check`` and ``turn_loop``
take an OPTIONAL ``verifier``. It is consulted at exactly the two claim-bearing
structural-ACCEPT points, and only through the sole helper ``_maybe_downgrade``,
so a rejected terminal is NEVER handed to it. The tier can therefore only ever
DOWNGRADE a structural ACCEPT to the typed ``unverified`` path (a REJECT with the
same ``LEGAL_NEXT``) — it can never upgrade a structural REJECT into an accept.
With ``verifier=None`` (the default) behaviour is byte-identical to the floor, so
the zero-LLM story and the reference suite are untouched. The verifier path
REQUIRES the wiring layer (``turn_loop`` / the adapter) to have populated
``state.turn_observations``; a raw ``boundary_check(..., verifier=X)`` with no
wiring scores the claim against ``[]`` and is a caller error.
"""

from .classifier import classify_observation

ACCEPT, REJECT = "ACCEPT", "REJECT"
LEGAL_NEXT = ["qualifying_tool_call", "unverified_terminal"]
CLAIM_TYPES = ("none", "assertion", "completion", "unverified")


def boundary_check(terminal_attempt, state, verifier=None):
    """THE choke point — must be the sole path to any terminal output.

    ``verifier`` is the OPTIONAL verify_with escalation (see module docstring):
    when supplied, a structural ACCEPT of a claim-bearing terminal may be
    DOWNGRADED to the typed ``unverified`` path via ``_maybe_downgrade``. A
    structural REJECT is returned before the verifier is ever consulted, and the
    ``unverified`` / ``none`` exits below never escalate — the floor is binding.
    """
    ct = terminal_attempt["claim_type"]        # none|assertion|completion|unverified
    if ct not in CLAIM_TYPES:
        # fail CLOSED: an unknown claim type (a typo like "assertoin") would
        # otherwise fall through to the exempt ``none`` ACCEPT below
        raise ValueError("unknown claim_type %r (expected one of %s)"
                         % (ct, ", ".join(CLAIM_TYPES)))

    if ct == "unverified":                     # universal escape hatch (typed)
        state.halted = False                   # never escalated — the honest exit
        return {"verdict": ACCEPT, "legal_next": []}

    claim_bearing = ct in ("assertion", "completion")

    if claim_bearing and state.budget <= 0:
        state.rejection_count += 1
        state.halted = True
        return {"verdict": REJECT, "legal_next": LEGAL_NEXT}

    if ct == "completion":
        if not state.verified_this_turn or any(
                s not in state.verified_signals for s in state.goal_predicates):
            state.rejection_count += 1
            state.halted = True
            return {"verdict": REJECT, "legal_next": LEGAL_NEXT}
        # structural ACCEPT — the ONLY place the verifier may see a completion
        return _maybe_downgrade(terminal_attempt, state, verifier)

    if ct == "assertion":
        # strict G (skipper preset): observed tier is not enough — even an
        # assertion needs verified-tier grounding (spec module 6)
        grounded = state.verified_this_turn if state.strict_g else state.grounded_this_turn
        if not grounded:
            state.rejection_count += 1
            state.halted = True
            return {"verdict": REJECT, "legal_next": LEGAL_NEXT}
        # structural ACCEPT — the ONLY place the verifier may see an assertion
        return _maybe_downgrade(terminal_attempt, state, verifier)

    return {"verdict": ACCEPT, "legal_next": []}   # ct == none: exempt, never escalated


def _maybe_downgrade(terminal_attempt, state, verifier):
    """The SOLE verifier entry point — floor-first, downgrade-only.

    Reached ONLY from the two claim-bearing structural-ACCEPT points above, so
    the claim it sees has already cleared the floor. It can return exactly two
    things: the floor's ACCEPT, or a stricter REJECT. There is NO path here that
    produces an ACCEPT for a claim the floor rejected — that is what makes the
    tier downgrade-only. Keep this the sole caller of the verifier.

    With ``verifier is None`` this reproduces the floor's original two lines
    EXACTLY (``state.halted = False`` then the plain ACCEPT), so a gate wired
    without a verifier is byte-identical to before this tier existed.

    Otherwise the ``.verifiers`` base is imported LAZILY (defence-in-depth for
    the zero-LLM story; a top-level import would also be safe since the base is
    stdlib). The verifier scores the claim against this turn's retained
    observations; ``aggregate`` collapses per-criterion scores to one confidence
    (weakest link) or ``None`` (ABSTAIN). A confidence STRICTLY below
    ``state.verify_threshold`` DOWNGRADES to REJECT + ``LEGAL_NEXT`` — which IS
    the typed-unverified path: ``LEGAL_NEXT`` already offers the
    ``unverified_terminal`` move, and a subsequently-submitted ``unverified``
    terminal is ACCEPTed WITHOUT consulting the verifier, so the tier can force
    the honest exit but can never block it. ``None`` (abstain) leaves the floor's
    ACCEPT standing. Reusing REJECT (not a new verdict) means every existing
    consumer — turn_loop's sole-emit gate, the adapter's block/budget/escape
    machinery — handles a downgrade unchanged, so the model is never trapped.
    """
    if verifier is None:
        state.halted = False
        return {"verdict": ACCEPT, "legal_next": []}

    from .verifiers import aggregate, criteria_for
    criteria = criteria_for(terminal_attempt["claim_type"], state.strict_g)
    conf = aggregate(verifier.score(
        terminal_attempt, list(state.turn_observations), criteria))
    if conf is not None and conf < state.verify_threshold:
        state.rejection_count += 1
        state.halted = True
        return {"verdict": REJECT, "legal_next": LEGAL_NEXT,
                "downgraded_by_verifier": True, "confidence": conf}
    state.halted = False
    return {"verdict": ACCEPT, "legal_next": []}


def turn_loop(script, state, verifier=None):
    """Drive a scripted agent through the gate. Returns ``(emitted, trace)``.

    Script steps:
      {"type": "reasoning"}
      {"type": "tool_call", "tool", "args", "result",
       "mutating": bool = False, "signals": list = None, "exit_ok": bool = True}
      {"type": "terminal", "attempt": {"claim_type", "content"}}

    ``verifier`` (optional) is forwarded to ``boundary_check`` at terminal steps;
    the qualifying observations this loop retains in ``state.turn_observations``
    are what the verifier scores against. With ``verifier=None`` the loop is
    byte-identical to the reference wiring.
    """
    trace = []
    for step in script:
        if step["type"] == "reasoning":
            # floored at 0: the divergence defense only ever tests <= 0 at
            # the boundary, so a deeper pit adds nothing — it only makes
            # recovery after entry-heavy reasoning cost more than one
            # qualifying observation
            state.budget = max(state.budget - 1, 0)               # (C4) refusals starve too
            if state.halted:
                trace.append(("refused_reasoning", LEGAL_NEXT))
                continue
            trace.append(("reasoning", None))
            continue

        if step["type"] == "tool_call":
            state.current_step += 1
            obs = classify_observation(step["tool"], step["args"], step["result"],
                                       state, read_only=not step.get("mutating", False))
            qualifying = obs["grounds_assertion"] or obs["grounds_completion"]
            state.grounded_this_turn |= obs["grounds_assertion"]   # (C1) latch
            state.verified_this_turn |= obs["grounds_completion"]  # (C1) latch
            if step.get("signals") and step.get("exit_ok", True):
                # declarative-rails signal mapper (reference: script-declared)
                state.verified_signals |= set(step["signals"])
            if qualifying:
                state.budget = min(state.budget + state.refill, state.cap)
                state.halted = False                               # (C2) qualifying only
                # retain the qualifying observation for the verifier tier +
                # telemetry (WIRING layer only — classify_observation stays
                # byte-identical and zero-LLM)
                state.turn_observations.append(
                    {"tool": step["tool"], "args": step["args"],
                     "result": step["result"],
                     "tier": "verified" if obs["grounds_completion"] else "observed"})
                if obs["grounds_completion"]:
                    state.last_verification_step = state.current_step
            if step.get("mutating"):
                state.last_mutation_step = state.current_step
                # a NEW mutation invalidates prior verification — the
                # verifying observation must postdate the LAST mutation
                state.verified_this_turn = False
            trace.append(("tool_call", obs))
            continue

        if step["type"] == "terminal":
            v = boundary_check(step["attempt"], state, verifier)
            trace.append(("terminal", v["verdict"]))
            if v["verdict"] == ACCEPT:
                return step["attempt"]["content"], trace           # SOLE terminal emit
            continue
    return None, trace
