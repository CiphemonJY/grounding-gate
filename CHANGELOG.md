# Changelog

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
