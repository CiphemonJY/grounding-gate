# Changelog

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
