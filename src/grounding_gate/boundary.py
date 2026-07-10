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
"""

from .classifier import classify_observation

ACCEPT, REJECT = "ACCEPT", "REJECT"
LEGAL_NEXT = ["qualifying_tool_call", "unverified_terminal"]


def boundary_check(terminal_attempt, state):
    """THE choke point — must be the sole path to any terminal output."""
    ct = terminal_attempt["claim_type"]        # none|assertion|completion|unverified

    if ct == "unverified":                     # universal escape hatch (typed)
        state.halted = False
        return {"verdict": ACCEPT, "legal_next": []}

    claim_bearing = ct in ("assertion", "completion")

    if claim_bearing and state.budget <= 0:
        state.halted = True
        return {"verdict": REJECT, "legal_next": LEGAL_NEXT}

    if ct == "completion":
        if not state.verified_this_turn or any(
                s not in state.verified_signals for s in state.goal_predicates):
            state.halted = True
            return {"verdict": REJECT, "legal_next": LEGAL_NEXT}
        state.halted = False
        return {"verdict": ACCEPT, "legal_next": []}

    if ct == "assertion":
        # strict G (skipper preset): observed tier is not enough — even an
        # assertion needs verified-tier grounding (spec module 6)
        grounded = state.verified_this_turn if state.strict_g else state.grounded_this_turn
        if not grounded:
            state.halted = True
            return {"verdict": REJECT, "legal_next": LEGAL_NEXT}
        state.halted = False
        return {"verdict": ACCEPT, "legal_next": []}

    return {"verdict": ACCEPT, "legal_next": []}   # ct == none: exempt


def turn_loop(script, state):
    """Drive a scripted agent through the gate. Returns ``(emitted, trace)``.

    Script steps:
      {"type": "reasoning"}
      {"type": "tool_call", "tool", "args", "result",
       "mutating": bool = False, "signals": list = None, "exit_ok": bool = True}
      {"type": "terminal", "attempt": {"claim_type", "content"}}
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
            if step.get("mutating"):
                state.last_mutation_step = state.current_step
                # a NEW mutation invalidates prior verification — the
                # verifying observation must postdate the LAST mutation
                state.verified_this_turn = False
            trace.append(("tool_call", obs))
            continue

        if step["type"] == "terminal":
            v = boundary_check(step["attempt"], state)
            trace.append(("terminal", v["verdict"]))
            if v["verdict"] == ACCEPT:
                return step["attempt"]["content"], trace           # SOLE terminal emit
            continue
    return None, trace
