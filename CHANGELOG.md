# Changelog

## Unreleased

- `boundary_check` now fails CLOSED on an unknown `claim_type` (raises
  `ValueError`). Previously a typo such as `"assertoin"` fell through to the
  exempt `none` branch and was ACCEPTed with no grounding.
- `grounding_gate.__version__` reported `0.4.0` in the 0.4.1 release; it now
  matches `pyproject.toml`, and a test pins the two together.
- README quickstart: the mutation bookkeeping now also clears
  `verified_this_turn`, matching `turn_loop`. Without it, a verification taken
  before a second mutation stayed latched and could ground a completion.
- New labeled benchmark, `examples/hallucination_bench.py`: 180 transcript
  families (design set + 12 held-out sets written round by round), each run
  over seeded variations through `turn_loop` and the Agent SDK adapter. The
  structural error rate (leaked ungrounded claims + blocked grounded ones)
  goes from 40.4% on 0.4.1 to 0.0%; CI fails above 5%. Changes it drove:
  - Relevance matches paths by trailing components (`./app.cfg`,
    `proj/app.cfg`, `/srv/proj/app.cfg`), but not other directories or URLs.
  - Per-target completion coverage: every change with a nameable target must
    be re-read after it (new `GateState.pending_verification`,
    `note_mutation`, `cover_pending`; shown in `progress()`). Re-reading one
    of two edited files, or an unchanged neighbour, no longer verifies. `rm`
    (globs included) discharges a deleted file.
  - `turn_loop` adds mutated identifiers to the claim surface (as the adapter
    already did) and gives a failed call (`exit_ok: False`) no completion
    credit.
  - Adapter: `Glob` is a listing tool (assertion tier only); `Read` and
    `NotebookRead` are content tools, relevant to the path read and the
    symbols in its text, not to file names it mentions; read-only shell
    commands (`cat`, `grep`, `git diff`, ...) count as reads of their file
    operands instead of as mutations; shell writes via `>`, `>>`, `tee` and
    `sed -i` are tracked as targets. New `content_tools=` / `listing_tools=`
    arguments.
  - Shell reads follow what reaches the agent: `sed -n`/`awk` print-only
    views and `python -m json.tool FILE` count; output piped into `wc`,
    sent to `/dev/null`, or from `grep -q/-c/-l` does not. `cd`, subshells
    and the hook's `cwd` are followed when resolving paths; `..` is
    normalized. `git show REV:path` never verifies; a bare `git diff`
    credits the files in its `+++ b/` headers.
  - Adapter: `Grep` in `files_with_matches`/`count` mode is a listing; the
    reference MCP filesystem server's tools are classified (content /
    listing / mutating). `turn_loop` tracks list-form args as targets.
  - Behavior change: a user-declared *path* on the claim surface is no
    longer satisfied by a `Read` of some other file whose text mentions it.
- CI tests every supported Python (3.9-3.13), builds the sdist/wheel with
  `twine check`, and imports the installed wheel with no extras. The release
  workflow refuses a tag that doesn't match the package version.

## 0.4.1 — 2026-09-15

- Packaging only; no functional change. Corrects the package author metadata,
  which named `Ciphemon` rather than James Yeung, and relicenses nothing (MIT
  throughout). Published so PyPI reflects the correct author.

## 0.4.0 — 2026-07-12

Adds an optional verifier tier, zero-token progress telemetry, and a
reproducible preset-tuning harness. Modules 1-4 remain ZERO-LLM and
stdlib-only; nothing optional is imported by the core (`import grounding_gate`
works with no extra deps), exactly as the Claude Agent SDK adapter already is.

- Optional `verify_with` verifier tier — the new `grounding_gate.verifiers`
  subpackage. `verifiers/__init__.py` is stdlib-only (a `runtime_checkable`
  `Verifier` Protocol, a deterministic `StubVerifier`, plus `criteria_for` /
  `aggregate` / `GRANULARITY`); `verifiers/llm.py` adds an `LLMVerifier`
  reference impl that lazily imports the Anthropic SDK behind the new `[llm]`
  extra (criteria DECOMPOSITION + K-sample REPEATED EVALUATION — an honest
  Monte-Carlo estimate, since the Messages API exposes no scoring-token
  logprobs; no `temperature`/`top_p`/`top_k`, which 400 on current models). The
  tier is wired at the submit boundary (`boundary_check` / `turn_loop` /
  `GateHooks` gain an optional `verifier=`) as an escalation that can ONLY
  downgrade a structural ACCEPT to the typed `unverified` path, never upgrade a
  structural REJECT — the floor runs first and independently, and `verifier=None`
  is a byte-identical no-op. Reached only through the sole helper
  `_maybe_downgrade` at the two claim-bearing ACCEPT points, so a rejected
  terminal is never handed to a verifier. This implements the leak-audit
  escalation the design named for semantic misreads / relevance-spoofing /
  adversarial self-deception.
- `GateState.progress()` — a zero-token, pure-function telemetry snapshot
  (budget/headroom, step, grounding latches, task-cumulative `rejection_count`,
  monotonic steps-since-last-mutation/verification, unmet signals). Surfaced by
  the adapter's `GateHooks.progress()` accessor (merged with adapter-only
  counters — the reliable programmatic surface) and an opt-in `emit_progress`
  event that suffixes the UNVERIFIED escape-valve `systemMessage` with a compact
  summary. New defaulted `GateState` fields (`turn_observations`,
  `verify_threshold`, `last_verification_step`, `rejection_count`) keep
  `for_model_class(**kw)` unaffected; observation retention lives in the wiring
  layer (`turn_loop` / adapter), so Module 2 (`classify_observation`) stays
  byte-identical and zero-LLM.
- Reproducible preset-tuning harness `examples/tune_presets.py` (not shipped in
  the wheel) that dogfoods the sibling `lcb-gate` package: it sweeps
  CAP/REFILL/strict_g over SEEDED synthetic transcripts and ranks candidates by
  paired-seed win-rate LCB via `lcb_gate.compare()` (common random numbers). No
  fabricated "tuned" numbers are committed — it prints a ranked table on demand
  and writes nothing; `lcb-gate` is imported lazily behind an optional `tuning`
  extra and the script self-checks + exits 0 when it is absent.
- Suite grows to 42 core + 28 adapter + 9 verifier cases, all hermetic
  (`StubVerifier` only, no network); new cases pin the invariants: the floor
  stays zero-LLM (`'anthropic'` absent after `import grounding_gate`), the
  verifier downgrades ACCEPT->unverified below threshold but can NEVER upgrade a
  REJECT, `unverified`/`none` are never escalated, and the `progress()` shape.

## 0.3.0 — 2026-07-10

Addresses three reviewer findings on novelty coverage, relevance domains,
and budget ergonomics:

- `normalize()` now strips syslog and `ls -l` listing timestamps (both
  forms), RFC822/1123 dates with optional day-of-week, bare/US dates,
   12/24-hour clock times (AM/PM marker consumed), minute-precision ISO,
  relative times through years, dashed UUIDs, and case-insensitive hex —
  in addition to second-precision ISO + long-digit ids. Stale re-reads in
  these formats no longer re-ground (wrong-acceptance fixes), and the ISO
  tail is bounded so payload glued to a timestamp survives (wrong-rejection
  fix). For formats beyond the defaults, per-tool scrubbers register in
  `GateState.normalizers` / `GateHooks(normalizers=...)`; custom scrubbers
  run first and the default floor always applies after.
- The mutation epoch moved INTO the novelty hash tuple (core + adapter):
  the reference `turn_loop` can now verify date-only/idempotent changes
  (previously impossible there), and custom normalizers can no longer
  corrupt the epoch (it isn't in the text anymore).
- `for_model_class` enforces the `1 <= refill < cap` invariant; extractor
  return values are coerced to sets (any iterable; a bare string counts as
  ONE identifier, not a character set).
- From the post-fix skeptic pass: `0x`-prefixed addresses (Python reprs)
  are scrubbed; month-name rules stop at line breaks (end-of-line counters
  survive); US dates require a 19xx/20xx year (block sizes survive). Bare
  24-hour `HH:MM` deliberately survives (scores/ratios ambiguity) and is
  documented as a residual for per-tool normalizers, as is the completion
  tier's freshness-not-coverage semantics.
- Per-tool relevance extractors (`GateState.extractors` /
  `GateHooks(extractors=...)`) bridge lexical-domain mismatches (inodes,
  opaque handles) that the default token extractor can never intersect.
- Budget floors at zero: entry-heavy reasoning no longer digs an
  unrecoverable pit — one qualifying observation restores assert-ability.
  `for_model_class()` accepts `cap=`/`refill=` overrides, and the adapter
  documents the SDK budget asymmetry (no reasoning-step hook exists there;
  `max_blocks` is the operative floor).
- Suite grows to 26 core + 22 adapter cases.

## 0.2.0 — 2026-07-10

- Claude Agent SDK adapter (`grounding_gate.adapters.claude_agent_sdk`):
  `GateHooks` maps the gate onto SDK hooks — `PostToolUse` observation
  classification with automatic claim-surface accumulation from mutating
  tools (values only, never JSON schema keys), `PostToolUseFailure`
  conservatively recording failed mutations, `Stop` as the submit boundary
  (`decision: block` + legal-next reason on REJECT), `UserPromptSubmit`
  turn reset (latches + budget), mutation-epoch novelty so idempotent
  writes remain verifiable, subagent events excluded by default, and an
  UNVERIFIED escape valve (`gate.exited_unverified` flag + user-facing
  `systemMessage`) after `max_blocks` rejected finishes or budget
  exhaustion. No new dependencies; the SDK is only needed for
  `as_options_hooks()`. Hardened by a 13-agent adversarial review whose
  reproduced findings are pinned as tests.
- Core `turn_loop`: a new mutation now invalidates prior verified-tier
  grounding — the verifying observation must postdate the LAST mutation
  (acceptance case M1; suite is now 20 cases).

## 0.1.0 — 2026-07-10

Initial release.

- `classify_observation` — novelty ∧ relevance ∧ consequence-tier observation
  classifier (corrected C1/C3 behavior preserved from review).
- `boundary_check` / `turn_loop` — submit-boundary choke point and reference
  loop wiring (latched flags, qualifying-only halt clear, starving refusals).
- `GateState.for_model_class` with `skipper` / `diverger` / `default` presets,
  including strict-G enforcement for the skipper class (assertions require
  verified-tier grounding).
- 19-case acceptance suite, runnable on bare Python with zero dependencies.
