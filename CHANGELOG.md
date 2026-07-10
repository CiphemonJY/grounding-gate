# Changelog

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
