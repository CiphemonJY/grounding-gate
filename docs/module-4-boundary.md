# Module 4 — Submit-boundary choke point (corrected)

`boundary_check` is the single decision that matters: it must be the **sole path to any
terminal output**. `turn_loop` is the reference wiring of the whole gate around an agent.

## Review history — leaks live BETWEEN the test cases

The draft of this module passed all 8 authored boundary cases (R1–A3) on the first try —
`boundary_check` itself survived review unchanged. Every bug was in `turn_loop`, in the
seams the per-case suite didn't cover:

- **(C1)** The draft assigned grounding flags from the *last* call
  (`verified_this_turn = obs.grounds_completion`), so any later non-completion-grade call
  overwrote a valid verification back to `false` — false-rejecting legitimate work.
  Flags are **latches** within a turn (`|=`), cleared only at turn boundaries.
- **(C2)** The draft cleared `halted` on *any* tool call — a halted model could escape via
  a novelty-defeated no-op read, reintroducing the no-op gaming vector at loop level.
  Halt is cleared **only by a qualifying observation** (or a typed `unverified` exit).
- **(C3)** The draft invented an `obs.signals` field that Module 2 does not return.
  Signal population goes through the declarative-rails signal mapper (declared command
  outcome → signal name), never through the classifier.
- **(C4)** Refused reasoning while halted must surface `legal_next` to the agent
  (re-prompt) or the loop livelocks; refusals also decrement budget, so a halted spinner
  still starves its way to the typed `unverified` exit.

The takeaway generalizes: a module can score 100% on its authored acceptance cases and
still leak — integration cases (L1–L3 below) exist precisely to cover the seams.

## Corrected pseudocode

```text
function boundary_check(terminal_attempt, state):
    ct = terminal_attempt.claim_type          # none | assertion | completion | unverified

    if ct == unverified:                      # universal escape hatch (typed, not content-matched)
        state.halted = false
        return { verdict: ACCEPT, legal_next: [] }

    is_claim_bearing = (ct == assertion) OR (ct == completion)

    if is_claim_bearing AND state.budget <= 0:
        state.halted = true
        return { verdict: REJECT, legal_next: [qualifying_tool_call, unverified_terminal] }

    if ct == completion:
        if NOT state.verified_this_turn:
            reject as above
        for sig in state.goal_predicates:
            if sig NOT in state.verified_signals:
                reject as above
        state.halted = false
        return { verdict: ACCEPT, legal_next: [] }

    if ct == assertion:
        grounded = state.verified_this_turn if state.strict_g    # strict G (module 6):
                   else state.grounded_this_turn                 # skipper preset requires
        if NOT grounded:                                         # verified tier even here
            reject as above
        state.halted = false
        return { verdict: ACCEPT, legal_next: [] }

    return { verdict: ACCEPT, legal_next: [] }   # ct == none: exempt
```

`turn_loop` wiring (see [boundary.py](../src/grounding_gate/boundary.py) for the runnable
version): reasoning decrements budget (refused or not); tool calls are classified, latch
the flags, refill budget and clear halt only when qualifying, and record mutation steps;
terminal attempts go through `boundary_check`, and an ACCEPT is the loop's only emit.

## Integration cases the corrections cover

| # | Case | Verdict |
|---|------|---------|
| L1 | edit → re-read (verified) → novelty-defeated repeat read → completion | ACCEPT (was falsely REJECTED pre-C1) |
| L2 | halted → novelty-defeated repeat read → reasoning | reasoning still refused (was allowed pre-C2) |
| L3 | halted → reasoning × N | budget starves to typed `unverified` (was free spin pre-C4) |
