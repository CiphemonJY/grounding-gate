# Design specification — Grounding Gate

A zero-token structural verifier for the agent loop.

## Problem this solves

Agents fail two ways, per-model, both bad:

- **Skip-and-hallucinate**: emit a confident terminal answer without ever observing reality.
- **Reason-and-diverge**: loop in closed context until a confident WRONG answer ships.

Prose reminders ("remember to verify") cost tokens every turn, are model-variable, and are
skippable. This gate moves enforcement into **control flow** (hash/set/integer ops, no LLM
calls) so it costs ~zero marginal tokens and no model can skip it. Semantic correctness is
explicitly OUT of scope — punted to a separate declared verifier tier.

## Core idea

One choke point wraps the submit/conclude boundary. A model may exit only if BOTH:

- **G (grounding)**: a *qualifying* observation happened this turn, OR the turn makes no factual claim.
- **B (budget)**: reasoning-rope remains (`budget > 0`).

Fail either → reject the terminal output, force a halt transition. `unverified` is a
first-class legal exit.

---

## Module 1 — State container

Per-task (persists across turns): `budget:int`, `recent_result_hashes:set`,
`goal_predicates:[Pred]`, `verified_signals:set`.
Per-turn (reset each turn): `grounded_this_turn:bool=false`, `verified_this_turn:bool=false`,
`claim_surface:set` (identifiers/paths the terminal answer references).

## Module 2 — Observation classifier (THE hard part; must be airtight)

Given a completed tool call `(tool, args, result)`, decide if it flips grounding. A call
qualifies only if **novel ∧ relevant ∧ consequence-tier-correct**:

1. **Novelty** — `h = hash(tool + normalize(args) + normalize(result))`. If
   `h ∈ recent_result_hashes` → NO credit (defeats no-op / re-read loops). Else add to set.
   `normalize()` strips timestamps/rng; tools flagged `novelty_exempt` bypass hashing
   (small audited allowlist).
2. **Relevance** — the call's args/result identifier-set must intersect `claim_surface`.
   If the answer asserts something about file X but no qualifying call this turn touched X
   → NOT relevant. Set intersection only.
3. **Consequence tier**:
   - read returning fresh relevant data → sets `observed` (grounds an *assertion*: "X is true").
   - observation taken *after* a mutating action, showing intended effect → sets `verified`
     (grounds a *completion*: "I changed X"). A mutating call alone NEVER self-grounds its
     own effect.

`grounded_this_turn = novel ∧ relevant`. `verified_this_turn` additionally requires the
post-mutation tier.

See [module-2-classifier.md](module-2-classifier.md) for the corrected pseudocode and the
review history.

## Module 3 — Budget engine (divergence killer, net token-NEGATIVE)

Per step: if step ended on a qualifying grounding obs → `budget = min(budget+REFILL, CAP)`;
else (pure reasoning / repeat / no-op) → `budget -= 1`. If `budget <= 0` → HALT.
Invariant: `1 <= REFILL < CAP`. Grounded work → effectively unbounded; closed-loop
reasoning → hard floor.

## Module 4 — Boundary check (the ONE choke point)

At every submit/conclude attempt — and this MUST be the only path to a terminal output:

- claim-bearing completion requires `verified_this_turn` (or declared verified signals).
- claim-bearing assertion requires `grounded_this_turn`.
- requires `budget > 0`.
- else → reject; legal next actions are ONLY {a qualifying tool call, emit `unverified`
  terminal}. Reasoning refused.

See [module-4-boundary.md](module-4-boundary.md) for the corrected pseudocode and the
turn-loop leak analysis.

## Module 5 — Declarative rails seam

A skill/task declares (frontmatter/manifest):

```yaml
gate:
  requires: [tests_passed, diff_nonempty]   # signal names that must be in verified_signals
  verify_with: <verifier-tier>              # escalation tier when floor is ambiguous (optional)
  irreversible: false
  model_class: diverger|skipper|default     # selects CAP/REFILL/G-strictness preset
```

The skill maps a real command's exit code / output to a signal name; the wrapper enforces
presence. The wrapper never interprets meaning — only checks "signal named X exists at
verified tier."

## Module 6 — Per-model-class presets (fleet variance absorbed as integers, not prose)

- `skipper` (hallucinate-and-skip): strict G (require `verified` even for assertions),
  CAP low-ish, REFILL modest.
- `diverger` (reason-forever): low CAP, REFILL small → starves closed loops fast.
- `default`: middle.

Shipped starting guesses: skipper CAP=5/REFILL=2, diverger CAP=4/REFILL=1,
default CAP=6/REFILL=2. Tune empirically.

---

## Leak audit (residual holes, stated honestly)

| Hole | Status |
|------|--------|
| Model misreads a REAL result and asserts wrongly | Punted to `verify_with` — semantic judgment is out of the floor by design |
| Relevance spoofing (deliberately mention the claimed identifiers in an irrelevant call) | Floor defends lazy ungroundedness only; adversarial self-deception → `verify_with` |
| Nondeterministic tools defeat novelty hashing | Defended via per-tool `normalize()` extensions + audited `novelty_exempt` allowlist |

## Explicit non-goals

- No semantic/correctness judgment in the floor (that's `verify_with`).
- No per-turn prose injection.
- The wrapper never calls an LLM to interpret output — hash/set/integer only.

## Acceptance

Every adversarial transcript rejected, every valid transcript passes, choke point provably
the sole exit, zero LLM calls in Modules 1–4. Pinned by the 20-case suite in
[tests/test_gate.py](../tests/test_gate.py).
