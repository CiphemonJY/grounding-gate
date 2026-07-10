# Module 2 — Observation classifier (corrected)

The classifier decides whether one completed tool call `(tool, args, result)` grounds a
claim. A call qualifies only if **novel ∧ relevant ∧ consequence-tier-correct**.

## Review history — and why it shaped the method

The first draft of this module was produced by one model and reviewed by another. The
review found:

- **(C1)** The consequence tier let a plain read ground a **completion** ("I changed X")
  when nothing had ever mutated: with `last_mutation_step = 0`, the step comparison was
  trivially true. The fix requires a mutation to have *occurred*
  (`last_mutation_step > 0`) before any read can count as post-mutation verification.
- **(C2)** The drafting model had also **authored the test case that ratified its own
  bug** — its test #1 expected `{true, true}` for a read with no prior mutation. The
  expectation was corrected to `{true, false}`.
- **(C3)** The draft recorded the novelty hash before the relevance check — a
  novel-but-irrelevant read burned its hash and was wrongly denied credit if the same
  read later became relevant. The hash is now recorded only after the relevance gate
  passes.

C2 is the important lesson, and it became a standing rule for this project: **expected
outcomes are authored by the reviewer, never by the generator.** A model grading its own
work will happily certify its own bug.

## Corrected pseudocode

```text
function classify_observation(tool, args, result, state) -> {grounds_assertion, grounds_completion}:
    ret = {grounds_assertion: false, grounds_completion: false}

    # 1. NOVELTY (check only; do not record yet — see C3)
    h = null
    if tool not in state.novelty_exempt:
        h = hash(tool + normalize(args) + normalize(result))
        if h in state.recent_result_hashes:
            return ret

    # 2. RELEVANCE
    identifiers = extract_identifiers(args, result)
    if not intersects(identifiers, state.claim_surface):
        return ret

    # record novelty only for calls that actually qualify (C3)
    if h != null:
        state.recent_result_hashes.add(h)

    # 3. CONSEQUENCE TIER
    if is_read_only(tool):
        ret.grounds_assertion = true
        # a completion needs a mutation to have OCCURRED and this read to follow it (C1)
        if state.last_mutation_step > 0 and state.current_step > state.last_mutation_step:
            ret.grounds_completion = true

    return ret
```

## Acceptance cases

| # | Case | Expected |
|---|------|----------|
| T1 | read with NO prior mutation | `{true, false}` ← corrected (C1/C2) |
| T2 | repeated identical read (novelty defeat) | `{false, false}` |
| T3 | unrelated file touched (relevance defeat) | `{false, false}` |
| T4 | mutating call claiming its own effect (consequence defeat) | `{false, false}` |
| T5 | read predating the mutation | `{true, false}` |
| T6 | post-mutation re-read (valid verified) | `{true, true}` |

## Open implementation risks — and their mitigation hooks

- `normalize()` must strip timestamps/rng or novelty never fires on nondeterministic
  tools (a miss fails toward wrong re-acceptance). The reference default covers ISO,
  syslog, RFC822/bare dates, clock times, relative times, and hex/long-digit ids;
  anything noisier registers a per-tool scrubber in `GateState.normalizers` — it runs
  first, and the default floor always applies after. `novelty_exempt` remains a small
  audited allowlist for tools whose output is legitimately never-repeating.
- `extract_identifiers()` must be conservative — over-extraction leaks relevance,
  under-extraction false-rejects cross-cutting work (blocked work, never wrong
  acceptance). Tools whose output lives in a different lexical domain than the claim
  surface (inodes, opaque handles) register a per-tool `GateState.extractors` entry.
