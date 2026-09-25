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
if obs["grounds_completion"] and state.cover_pending(tool, args, result):
    state.verified_this_turn = True    # every file changed so far has been re-read
if mutating:
    state.note_mutation(args)          # a completion now needs a read of THIS change

# at every submit/conclude attempt — this must be the ONLY path to output:
verdict = boundary_check({"claim_type": "completion", "content": answer}, state)
if verdict["verdict"] == "REJECT":
    ...  # surface verdict["legal_next"] to the model and continue the loop
```

Note the mutation bookkeeping: without `note_mutation` ever being called, no
read can reach the verified tier and a `completion` can never be accepted —
that is the gate working as designed, not a bug. `note_mutation` records the
step, drops earlier verification, adds what was changed to the claim surface,
and marks each target it can identify (a `file_path`-style argument, or a
surface entry named in free-text args) as owed a re-read; `cover_pending`
settles those as reads arrive.

Relevance matches paths by trailing components, so `app.cfg`, `./app.cfg`,
`proj/app.cfg` and `/srv/proj/app.cfg` all name the same file, while
`/etc/app.cfg` and `/srv/proj/app.cfg` stay distinct (so do URL paths).

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
  identifiers in an irrelevant call, and a claim's relevance to *symbols*
  (not paths) is only as precise as the text: a note that mentions
  `parse_config` grounds a claim about it as well as its source does. The floor defends against *lazy*
  ungroundedness, which is the overwhelmingly common failure; adversarial
  self-deception needs the verifier tier.
- **Completion coverage is only as good as target identification.** Each
  change whose target the gate can name (a `file_path`-style argument, a
  surface entry in free-text args, or a shell `> f`, `>> f`, `tee f`,
  `sed -i ... f`) must be re-read after it happened (or deleted with `rm`;
  a move carries the debt to the new path; owed re-reads reset each turn), so re-reading `a.cfg`
  after editing `b.cfg` no longer passes. A change it can't name (a script,
  `mv`, `python fix.py`) falls back to freshness: any novel, relevant read
  after it counts. Treating those as owed would trap the agent on a target
  no read could ever match.
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

## The verify_with verifier tier

The holes above (semantic misreads, relevance spoofing, adversarial
self-deception) are, by design, punted out of the zero-LLM floor to an
**optional** escalation tier — the `verify_with` seam the leak audit names.
It lives in `grounding_gate.verifiers` and is opt-in:

```python
from grounding_gate import boundary_check
from grounding_gate.verifiers import StubVerifier          # deterministic, offline
# from grounding_gate.verifiers.llm import LLMVerifier     # optional, needs [llm]

verdict = boundary_check(attempt, state, verifier=StubVerifier(0.9))
```

- **`StubVerifier`** (stdlib, deterministic) returns a fixed score or delegates
  to a `rule(claim, observations, criteria)` — the hermetic stand-in used
  throughout the test suite, no network.
- **`LLMVerifier`** (optional, `pip install grounding-gate[llm]`) is a reference
  impl: it **decomposes** the claim into criteria and, per criterion, does
  **repeated evaluation** — `k` independent YES/NO samples of the API's inherent
  sampling distribution, averaged. That Monte-Carlo mean is an *estimator* of the
  probability the source paper reads off output logits; the Anthropic Messages
  API exposes no scoring-token logprobs, so we sample instead (cost = `k` model
  calls, no `temperature`/`top_p`/`top_k` — they 400 on current models). The SDK
  is imported lazily inside `LLMVerifier`, so `import grounding_gate` stays
  stdlib-only.

The wiring is **downgrade-only**: the verifier is consulted **only** at the two
points where the structural floor already decided ACCEPT for a claim-bearing
terminal. A confidence below `state.verify_threshold` (0.5) downgrades that
ACCEPT to the typed `unverified` path (a `REJECT` carrying
`downgraded_by_verifier`); an abstain (`None`) leaves the floor's ACCEPT
standing. It is **never** consulted on a structural REJECT, nor on the
`unverified`/`none` exits — so **the LLM can add strictness, never bypass the
gate**, and because a downgrade reuses the ordinary `REJECT` it flows through the
same budget/escape machinery and can never trap the agent. The floor runs first,
independently, and zero-LLM; `verifier=None` (the default) is a byte-identical
no-op. In the SDK adapter, pass `GateHooks(verifier=...)`.

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

Tools are sorted by what their output can prove. `Read`/`NotebookRead` are
*content* tools: they are relevant to the path they were given, not to names
their text mentions. `Glob` is a *listing* tool: it can ground an assertion
but never verify a change (it shows a file exists, not what it now says).
Shell commands are parsed by `grounding_gate.shell`, a small quote- and
heredoc-aware lexer (heredoc bodies are data, `>` inside quotes is text).
`Bash` counts as mutating, except for a command built only from known
read-only programs (`cat`, `head`, `grep`, `diff`, `git diff`, ...; no
redirects, `tee` or `$(...)`), which is treated as a read of the files it
operates on: `cat app.cfg` can verify an edit, `ls -l app.cfg` can only ground
an assertion, and `echo app.cfg` or the pattern in `grep app.cfg notes.txt`
counts for nothing. What counts is what reached the agent: the last stage
of a pipeline must pass lines through (`cat app.cfg | grep x` shows content;
`cat app.cfg | wc -l`, `head app.cfg > /dev/null` and `grep -q` do not).
`sed` and `awk` count as reads when they can only print (`sed -n '1,40p'`,
`awk 'NR<=20'`), as does `python -m json.tool FILE`. `cd` inside a command,
subshells and the session's working directory (the hook's `cwd`) are
followed, so `cd conf && cat app.cfg` reads `conf/app.cfg`, and `~/x`
resolves to the agent's home directory (`GateHooks(home=...)` if the agent
runs as another user). `git show
HEAD:app.cfg` is the committed copy, not the edit; a bare `git diff` counts
for the files in its `+++ b/` headers. `Grep` in its default
`files_with_matches` (or `count`) mode is a listing. The reference MCP
filesystem server's tools (`mcp__<server>__read_text_file`, `write_file`,
`list_directory`, ...) are classified like their built-in counterparts.
Symbols in content (`parse_config`, including the
`load_settings` in `cfg.load_settings()`) still ground claims about them;
file names a document merely mentions do not. Override with
`content_tools=` / `listing_tools=`.

The adapter adds no dependency: grounding-gate stays stdlib-only, and only
`as_options_hooks()` requires `claude-agent-sdk` to be installed.

## Measuring the error rate

[examples/hallucination_bench.py](https://github.com/CiphemonJY/grounding-gate/blob/main/examples/hallucination_bench.py)
drives labeled transcript families through `turn_loop` and the SDK adapter.
Each family's correct verdict comes from what actually happened in the
transcript (was the claimed file really read, after the change?), not from
the gate's own fields. A wrong ACCEPT is a *leak* (a hallucinated claim got
through) and a wrong REJECT is a *false reject* (honest work blocked). Each
family runs over seeded variations: path spellings, how the surface is
declared, noisy file contents.

The design set was used to choose changes. Each held-out set was written
after the previous round, measured before any change it motivated, and then
retired into the design pool, so the newest set is the real generalization
check. Structural error rate per set, 200 seeds per family:

| Set | Families | Before (0.4.1) | After |
|---|---|---|---|
| design | 15 | 30.0% | 0.0% |
| held-out 1 | 13 | 39.7% | 0.0% |
| held-out 2 | 15 | 20.0% | 0.0% |
| held-out 3 | 15 | 48.7% | 0.0% |
| held-out 4 | 15 | 35.6% | 0.0% |
| held-out 5 | 12 (+1 semantic) | 16.7% | 0.0% |
| held-out 6 | 12 (+1 semantic) | 41.7% | 0.0% |
| held-out 7 | 11 (+2 semantic) | 42.1% | 0.0% |
| held-out 8 | 17 | 57.1% | 0.0% |
| held-out 9 | 16 | 43.8% | 0.0% |
| held-out 10 | 12 | 56.7% | 0.0% |
| held-out 11 | 11 | 44.5% | 0.0% |
| held-out 12 (newest) | 12 | 47.7% | 0.0% |
| **all 180** | | **40.4%** | **0.0%** (1.1% with semantic pairs) |

Fresh sets keep finding new gaps: sets 8-12 each scored 7-42% against the
gate as it stood before they were written (Grep's file-list mode, `sed -n`
views, `cd` in commands, `git show REV:path`, output piped into `wc`). Each
gap was fixed in the round after it was found, so read "0.0%" as "every
known scenario", not as a bound on the next new one.

Four *semantic* families are reported but kept out of the structural rate.
They come in pairs whose transcripts are identical and whose right answer
depends only on what the claim says: reading a notes file that mentions
`parse_config` does ground "the notes mention parse_config" but not "the
function returns X", and a `Glob` hit grounds "settings.toml exists" but not
"its port is 8080". The floor never sees the answer text, so any structural
rule gets one of each pair wrong (both alternatives were tried and moved the
error rather than removing it). That is the `verify_with` tier's job.

CI runs `python examples/hallucination_bench.py --max-error 0.05` (fails if
any set's structural rate exceeds 5%). The families encode this README's
semantics, so 0% means "no known structural failure mode", not "no failure
mode". The limits above still apply.

## Preset tuning

Presets are starting guesses, and
[examples/tune_presets.py](https://github.com/CiphemonJY/grounding-gate/blob/main/examples/tune_presets.py)
is a reproducible harness for sweeping CAP/REFILL/strict-G over **seeded**
synthetic transcripts, ranking candidates against the shipped default by
paired-seed win-rate lower confidence bound via the sibling
[`lcb-gate`](https://pypi.org/project/lcb-gate/)'s `compare()` (common random
numbers). It tunes the structural floor only (no LLM), so it is offline and
deterministic. **No "tuned" numbers are committed** — the table is regenerated
on demand (`python examples/tune_presets.py --profile diverger --n 300`) and the
script writes nothing; `lcb-gate` is an optional example dependency
(`pip install grounding-gate[tuning]`) and the harness self-checks and exits 0
when it is absent.

## Status & roadmap

This is the reference implementation — correct, minimal, and framework-free.
Shipped: the Claude Agent SDK hook adapter (above), the optional `verify_with`
verifier tier (downgrade-only; `StubVerifier` + the `[llm]` `LLMVerifier`), and
`GateState.progress()` zero-token telemetry (surfaced via
`GateHooks.progress()` / opt-in `emit_progress`). Planned next:

- Adapters: LangGraph middleware, OpenAI Agents SDK.
- A real signal-mapper module (command exit code → declared signal).
- Empirical preset tuning across model classes (the harness above; committed
  results are regenerated, never fabricated).

## License

MIT
