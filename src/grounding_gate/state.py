"""Gate state, model-class presets, and the normalization helpers.

Spec modules 1 (state container) and 6 (per-model-class presets): fleet
variance is absorbed as integers, not prose.
"""

import re
from dataclasses import dataclass, field

# Per-model-class presets. Two documented agent failure modes get their own
# tuning: "skipper" models emit confident terminals without observing reality
# (strict grounding), "diverger" models reason in closed context until a
# confident wrong answer ships (small budget, small refill — starves loops).
PRESETS = {
    "skipper":  {"CAP": 5, "REFILL": 2, "strict_g": True},
    "diverger": {"CAP": 4, "REFILL": 1, "strict_g": False},
    "default":  {"CAP": 6, "REFILL": 2, "strict_g": False},
}


@dataclass
class GateState:
    budget: int
    cap: int
    refill: int
    # strict G (skipper preset): even assertions require the verified tier
    strict_g: bool = False
    # novelty
    recent_result_hashes: set = field(default_factory=set)
    novelty_exempt: set = field(default_factory=set)
    # per-tool extension hooks (tool name -> callable); see normalize() and
    # extract_identifiers() for the contracts they extend
    normalizers: dict = field(default_factory=dict)
    extractors: dict = field(default_factory=dict)
    # relevance
    claim_surface: set = field(default_factory=set)
    # consequence
    last_mutation_step: int = 0   # 0 = no mutation has EVER occurred
    current_step: int = 0
    # per-turn latches
    grounded_this_turn: bool = False
    verified_this_turn: bool = False
    # declarative rails
    verified_signals: set = field(default_factory=set)
    goal_predicates: list = field(default_factory=list)
    halted: bool = False

    @classmethod
    def for_model_class(cls, model_class="default", cap=None, refill=None, **kw):
        """Presets are STARTING GUESSES — tune ``cap``/``refill`` per model.

        A model that front-loads reasoning before its first tool call needs a
        higher ``cap`` than the aggressive ``diverger`` default.
        """
        p = PRESETS[model_class]
        kw.setdefault("strict_g", p["strict_g"])
        cap = p["CAP"] if cap is None else cap
        refill = p["REFILL"] if refill is None else refill
        if not 1 <= refill < cap:                # spec module 3 invariant
            raise ValueError(
                "need 1 <= refill < cap (got refill=%r, cap=%r): refill < 1 "
                "breaks one-observation recovery; refill >= cap makes the "
                "budget meaningless" % (refill, cap))
        return cls(budget=cap, cap=cap, refill=refill, **kw)


# Applied in order; the broad hex/long-digit rule runs LAST. Note the hex rule
# also eats pure-digit runs >= 8 (epoch seconds/millis included) — that bias
# is deliberate: over-stripping fails toward novelty-REJECTION (safe), while
# under-stripping fails toward wrong re-acceptance. Anchors (month names,
# AM/PM markers, date shapes) keep the riskier patterns from eating scores,
# ratios, and versions.
_MON = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
_DOW = r"(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)"
_NOISE_PATTERNS = (
    # ISO datetime, seconds optional, tail bounded to timestamp parts only
    # (a greedy \S* would eat payload glued to the timestamp)
    (re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2})?(?:[.,]\d+)?"
                r"(?:Z|[+-]\d{2}:?\d{2})?"), "<TS>"),
    # syslog / ls -l recent: "Jul 10 20:47[:03]" — [ \t] only: \s would eat
    # newlines and swallow meaningful end-of-line content into the token
    (re.compile(r"\b" + _MON + r"[ \t]+\d{1,2}[ \t]+\d{1,2}:\d{2}(?::\d{2})?\b"),
     "<TS>"),
    # RFC822/RFC1123 date, optional day-of-week: "[Thu, ]10 Jul 2026"
    (re.compile(r"\b(?:" + _DOW + r",?[ \t]+)?\d{1,2}[ \t]+" + _MON +
                r"[ \t]+\d{4}\b"), "<DATE>"),
    # ls -l older-than-6-months: "Jul 10  2025"
    (re.compile(r"\b" + _MON + r"[ \t]+\d{1,2}[ \t]+\d{4}\b"), "<DATE>"),
    (re.compile(r"\b\d{4}[-/]\d{2}[-/]\d{2}\b"), "<DATE>"),               # bare date
    # US date, year anchored to 19xx/20xx (else block sizes like 10/12/1024
    # would collapse)
    (re.compile(r"\b\d{1,2}/\d{1,2}/(?:19|20)\d{2}\b"), "<DATE>"),        # US date
    # 12-hour clock, AM/PM marker consumed too (else it leaks across noon)
    (re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?\s*[APap]\.?[Mm]\.?\b"),
     "<TS>"),
    (re.compile(r"\b\d{1,2}:\d{2}:\d{2}(?:[.,]\d+)?\b"), "<TS>"),         # clock time
    (re.compile(r"\b\d+\s*(?:ms|s|secs?|seconds|m|mins?|minutes|h|hrs?|hours|"
                r"d|days?|w|wks?|weeks?|mos?|months?|y|yrs?|years?)\s+ago\b"),
     "<REL>"),                                                            # relative
    (re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
                r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"), "<HEX>"),           # UUID
    # 0x-prefixed addresses (Python reprs, pointers): the generic rule below
    # can't reach them ('x' is a word char, so no \b precedes the hex run).
    # 6+ digits so small constants like 0xFF stay meaningful.
    (re.compile(r"\b0[xX][0-9a-fA-F]{6,}\b"), "<HEX>"),
    (re.compile(r"\b[0-9a-fA-F]{8,}\b"), "<HEX>"),                        # hex/long ids
)


def normalize(text):
    """Strip nondeterminism before hashing: timestamps in the common formats
    (ISO, syslog, RFC822 dates, bare dates/clock times, "N min ago") plus
    hex/long-digit ids.

    No fixed list covers every tool — known residuals include bare 24-hour
    HH:MM times (deliberately left: eating them would collapse scores,
    ratios, and aspect ratios like 16:9), short counters, digit runs glued
    to hex-letter words, and locale-specific formats. For
    anything noisier, register a per-tool scrubber in ``GateState.normalizers``
    (``{tool_name: callable}``). The custom scrubber runs FIRST and these
    default patterns always run after it — so a DETERMINISTIC hook can only
    add scrubbing (a nondeterministic hook can still defeat novelty; don't
    write one). There is deliberately no per-tool opt-OUT of the defaults:
    over-stripping fails toward rejection (safe); tools whose output is
    legitimately never-repeating belong in ``novelty_exempt`` instead. A
    missed pattern fails toward wrong re-acceptance, which is why noisy tools
    deserve their own entry.
    """
    text = str(text)
    for pattern, token in _NOISE_PATTERNS:
        text = pattern.sub(token, text)
    return text


def extract_identifiers(args, result):
    """Conservative token extraction for the relevance check.

    Over-extraction leaks relevance; under-extraction false-rejects
    cross-cutting work. When a tool's output lives in a different lexical
    domain than the claim surface (inode numbers, opaque handles, row ids),
    the default can never intersect — register a per-tool extractor in
    ``GateState.extractors`` (``{tool_name: callable(args, result) -> set}``)
    that maps the tool's output back to surface identifiers. A registered
    extractor REPLACES this default and *is* the relevance gate for that
    tool: it must derive identifiers from what the call actually touched
    (resolve the handle, parse the output) — an unconditional constant set
    turns every call into a wrong acceptance. The failure direction of a
    MISSING extractor is a false REJECTION (blocked work), never a wrong
    acceptance; a careless one is the opposite.
    """
    return set(re.findall(r"[\w.\-/]+", f"{args} {result}"))
