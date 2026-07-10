"""Observation classifier — spec module 2, the hard part.

A completed tool call grounds a claim only if it is
novel AND relevant AND consequence-tier-correct.

Two corrections survived adversarial review of the original draft (the drafting
model had also authored the test that ratified its own bug — acceptance cases
here are authored by the reviewer, never the generator):
  (C1) a completion requires a mutation to have OCCURRED
       (``last_mutation_step > 0``) — otherwise a plain read grounds a
       "I changed X" claim when nothing was ever changed.
  (C3) the novelty hash is recorded only AFTER the relevance gate passes —
       otherwise a novel-but-irrelevant read burns its hash and is wrongly
       denied if it later becomes relevant.
"""

from .state import extract_identifiers, normalize


def classify_observation(tool_name, args, result, state, read_only):
    """Decide whether one completed tool call flips grounding.

    Returns ``{"grounds_assertion": bool, "grounds_completion": bool}``.
    Mutates ``state.recent_result_hashes`` for qualifying novel calls.
    """
    ret = {"grounds_assertion": False, "grounds_completion": False}

    h = None
    if tool_name not in state.novelty_exempt:                      # 1. NOVELTY
        h = hash((tool_name, normalize(args), normalize(result)))
        if h in state.recent_result_hashes:
            return ret

    idents = extract_identifiers(args, result)                     # 2. RELEVANCE
    if not (idents & state.claim_surface):
        return ret

    if h is not None:                                              # (C3)
        state.recent_result_hashes.add(h)

    if read_only:                                                  # 3. CONSEQUENCE
        ret["grounds_assertion"] = True
        if state.last_mutation_step > 0 and state.current_step > state.last_mutation_step:
            ret["grounds_completion"] = True                       # (C1)
    return ret
