# grounding-gate

[![ci](https://github.com/CiphemonJY/grounding-gate/actions/workflows/ci.yml/badge.svg)](https://github.com/CiphemonJY/grounding-gate/actions/workflows/ci.yml)

**Zero-token structural verifier for agent loops.** One choke point at the
submit boundary decides whether an agent is allowed to say "X is true" or
"I did X" — using hash, set, and integer operations only. No LLM calls, no
per-turn prompt injection, no dependencies.

```
pip install grounding-gate      # stdlib only, Python >= 3.9
```

The demo ships in the repo (not the wheel):

```
git clone https://github.com/CiphemonJY/grounding-gate && cd grounding-gate
python examples/demo.py         # the whole idea in 30 seconds
```

## The problem

Agents fail in two characteristic ways, and both ship *confident* wrong answers:

- **Skip-and-hallucinate** — emit a terminal claim ("done, config fixed")
  without ever observing reality after acting on it.
- **Reason-and-diverge** — loop in closed context, burning steps on
  reasoning about stale beliefs, until a confident wrong answer ships.

The standard fix is prose: "remember to verify your work" injected into every
turn. Prose costs tokens on every turn, behaves differently per model, and —
critically — is *skippable*. A reminder is not an invariant.

## The idea

Move enforcement out of the prompt and into **control flow**. A single gate
wraps the submit/conclude boundary, and a terminal output is emitted only if
both invariants hold:

- **G (grounding)** — a *qualifying* observation happened this turn, or the
  output makes no factual claim. Qualifying means **novel** (result hash not
  seen before, after stripping timestamps/ids) **∧ relevant** (touches the
  identifiers the claim is about) **∧ consequence-tier-correct** (see below).
- **B (budget)** — reasoning rope remains. Qualifying observations *refill*
  the budget (up to a cap); pure reasoning steps decrement it. Grounded work
  runs effectively unbounded; closed-loop reasoning hits a hard floor.

Fail either → the terminal is **rejected** and the agent is told its only
legal moves: make a qualifying tool call, or exit with a typed **`unverified`**
terminal. `unverified` is a first-class, always-legal escape hatch — the gate
never traps an agent, it only forbids *confident* ungrounded claims.

### Consequence tiers

The gate distinguishes what kind of claim an observation can support:

| Claim type   | Example                  | Requires |
|--------------|--------------------------|----------|
| `assertion`  | "X is true"              | a novel, relevant, read-only observation this turn |
| `completion` | "I changed X"            | a novel, relevant read taken **after** the mutation — a mutating call never self-grounds its own effect |
| `unverified` | "couldn't confirm X"     | nothing — always legal |
| `none`       | no factual claim         | nothing — exempt |

That second row is the heart of it: *writing a file and claiming success is
not verification; reading it back afterwards is.*

## Quickstart

```python
from grounding_gate import GateState, classify_observation, boundary_check

state = GateState.for_model_class("default", claim_surface={"app.cfg"})

# after EVERY tool call in your agent loop:
state.current_step += 1
obs = classify_observation(tool, args, result, state, read_only=not mutating)
state.grounded_this_turn |= obs["grounds_assertion"]
state.verified_this_turn |= obs["grounds_completion"]
if mutating:
    state.last_mutation_step = state.current_step   # a completion now needs a read AFTER this

# at every submit/conclude attempt — this must be the ONLY path to output:
verdict = boundary_check({"claim_type": "completion", "content": answer}, state)
if verdict["verdict"] == "REJECT":
    ...  # surface verdict["legal_next"] to the model and continue the loop
```

Note the mutation bookkeeping: without `last_mutation_step` ever being set, no
read can reach the verified tier and a `completion` can never be accepted —
that is the gate working as designed, not a bug.

`turn_loop` in [boundary.py](https://github.com/CiphemonJY/grounding-gate/blob/main/src/grounding_gate/boundary.py)
is the complete reference wiring (budget refill, mutation tracking, halt
semantics, signal mapping) — use it as the integration template. The
[demo](https://github.com/CiphemonJY/grounding-gate/blob/main/examples/demo.py)
runs the same scripted agent through an ungated and a gated loop, side by side.

## Model-class presets

Fleet variance is absorbed as integers, not prose. Pick the preset matching
how your model fails:

| Preset     | CAP | REFILL | Strict G | For |
|------------|-----|--------|----------|-----|
| `skipper`  | 5   | 2      | yes      | models that hallucinate-and-skip |
| `diverger` | 4   | 1      | no       | models that reason forever |
| `default`  | 6   | 2      | no       | everything else |

Strict G means even plain *assertions* require verified-tier grounding (a
post-mutation observation) — an observed-tier read is not enough. In a task
that never mutates anything, a strict-G agent can only exit via the typed
`unverified` terminal; that hard line is the point of the skipper preset, so
pick `default` for read-only/Q&A workloads.

## Declarative rails

A task can declare signals that must be verified before any completion is
accepted (`state.goal_predicates = ["tests_passed"]`). The gate never
interprets meaning — it only checks that a signal named `tests_passed` was
registered by a mapped, real command outcome. Semantic judgment stays out of
the floor by design.

## What the gate does NOT do

Honest scope, from the design's leak audit:

- **No semantic correctness.** A grounded claim can still be wrong (the model
  can misread a real result). That is punted to a declared verifier tier
  (`verify_with`), not smuggled into the floor.
- **Relevance can be spoofed** by a model that deliberately mentions the right
  identifiers in an irrelevant call. The floor defends against *lazy*
  ungroundedness, which is the overwhelmingly common failure; adversarial
  self-deception needs the verifier tier.
- **The completion tier checks freshness, not coverage.** A completion needs
  a novel, relevant observation taken after the last mutation — the gate
  cannot prove that observation was *of the mutated item* when the claim
  surface holds several identifiers (re-reading unchanged `a.cfg` after
  editing `b.cfg` passes if both are on the surface). The adapter narrows
  this by auto-adding mutated identifiers, but a broad user-seeded surface
  keeps the coarseness. Tracking per-mutation coverage is future work.
- **Nondeterministic tools defeat novelty unless you tell the gate about
  them.** The default `normalize()` strips the common timestamp shapes —
  ISO (second- or minute-precision), syslog and `ls -l` listings, RFC822/1123
  dates with day-of-week, bare and US dates, 12/24-hour clock times, relative
  times through years — plus UUIDs and hex/long-digit ids. But no fixed list
  covers every tool (short counters and digit runs glued into hex-letter
  words are known residuals), and a missed pattern fails toward wrong
  re-acceptance. Register a per-tool scrubber in `GateState.normalizers`
  (or `GateHooks(normalizers=...)`); it runs before the default, which
  always still applies — keep scrubbers deterministic.
- **Relevance can under-extract across lexical domains** — a tool returning
  an inode number never intersects a claim surface of file paths, and the
  gate false-rejects (blocked work, never wrong acceptance). Register a
  per-tool `GateState.extractors` entry mapping that tool's output back to
  surface identifiers. Registered extractors REPLACE the default and *are*
  the relevance gate for that tool: derive identifiers from what the call
  actually touched — an unconditional constant set makes every call
  "relevant" and reopens the wrong-acceptance door the default keeps shut.
- **The budget is a hard line, and it's tunable.** Presets are starting
  guesses: `diverger` (CAP 4) deliberately forces early grounding, so a model
  that front-loads reasoning wants
  `GateState.for_model_class("diverger", cap=10)` (the `1 <= refill < cap`
  invariant is enforced). The budget floors at zero — one qualifying
  observation restores assert-ability (under strict-G, that observation must
  be verified-tier, per the preset's rule). In the Agent SDK adapter the
  budget is secondary (no reasoning-step hook exists there); `max_blocks` is
  the operative floor.

## How this was built

The modules were drafted by different LLMs and adversarially reviewed before
assembly; the final behavior is pinned by a 30-case acceptance suite
([tests/test_gate.py](https://github.com/CiphemonJY/grounding-gate/blob/main/tests/test_gate.py))
that runs on bare Python with zero dependencies. Two review findings shaped
the method and are preserved in the docstrings:

- A drafting model shipped a consequence-tier bug **and authored the test that
  ratified it** — since then, expected outcomes are authored by the reviewer,
  never by the generator
  ([docs/module-2-classifier.md](https://github.com/CiphemonJY/grounding-gate/blob/main/docs/module-2-classifier.md)).
- The remaining leaks lived *between* individually-passing test cases —
  latch-vs-assignment, halt cleared by non-qualifying calls
  ([docs/module-4-boundary.md](https://github.com/CiphemonJY/grounding-gate/blob/main/docs/module-4-boundary.md)).

Full design spec:
[docs/spec.md](https://github.com/CiphemonJY/grounding-gate/blob/main/docs/spec.md).

## Claude Agent SDK adapter

`grounding_gate.adapters.claude_agent_sdk` wires the gate into a
[Claude Agent SDK](https://code.claude.com/docs/en/agent-sdk) agent using
hooks — `PostToolUse` classifies every successful tool result,
`PostToolUseFailure` conservatively records failed mutating calls (a failed
write may still have had an effect, so verification is demanded), `Stop` is
the submit boundary (a rejected finish is blocked and the model is told its
legal next moves), and `UserPromptSubmit` resets the per-turn latches and
budget:

```python
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions
from grounding_gate.adapters.claude_agent_sdk import GateHooks

gate = GateHooks(model_class="default")          # one instance per session
options = ClaudeAgentOptions(hooks=gate.as_options_hooks())

async with ClaudeSDKClient(options=options) as client:
    await client.query("Fix the timeout in app.cfg and confirm it took effect")
    ...
```

Identifiers touched by mutating tools join the claim surface automatically —
*you must verify what you changed* (values only, never JSON schema keys, so a
read of some unrelated file can't masquerade as verification) — and a new
mutation invalidates any earlier verification: an agent that edits `app.cfg`
and tries to finish without re-reading it gets blocked with an explanation,
and its completion is only accepted after a fresh read that postdates the
last change. Subagent tool events are excluded from the gate's state by
default (`gate_subagents=True` opts in).

Because the SDK has no typed terminals, the gate's `unverified` escape hatch
becomes an escape valve: after `max_blocks` rejected finishes — or when the
per-turn reasoning budget runs out, whichever comes first — the stop is
allowed, **`gate.exited_unverified` is set** (check this flag in headless
runs), and a `systemMessage` warning is returned. Per the SDK contract that
message is shown to the *user*, not the model, and appears in headless runs
only with `include_hook_events` enabled — the flag is the reliable marker.
The gate never traps an agent.

The adapter adds no dependency: grounding-gate stays stdlib-only, and only
`as_options_hooks()` requires `claude-agent-sdk` to be installed.

## Status & roadmap

This is the reference implementation — correct, minimal, and framework-free.
Shipped: the Claude Agent SDK hook adapter (above). Planned next:

- Adapters: LangGraph middleware, OpenAI Agents SDK.
- A real signal-mapper module (command exit code → declared signal).
- Empirical preset tuning across model classes.

## License

MIT
