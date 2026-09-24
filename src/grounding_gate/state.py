"""Gate state, model-class presets, and the normalization helpers.

Spec modules 1 (state container) and 6 (per-model-class presets): fleet
variance is absorbed as integers, not prose.
"""

import fnmatch
import posixpath
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
    # last step at which a VERIFIED-tier (post-mutation) observation landed.
    # A MONOTONIC step marker like last_mutation_step (0 = never), maintained
    # by the wiring layer (turn_loop / adapter), NEVER reset per turn — so
    # progress()'s steps_since_last_verification can point across a turn
    # boundary. Kept out of classify_observation, which stays free of telemetry.
    last_verification_step: int = 0
    # TASK-CUMULATIVE count of structural + verifier-downgrade REJECTs at the
    # boundary. Not per-turn — boundary_check increments it on every REJECT.
    rejection_count: int = 0
    # mutation targets not yet re-read since they were changed. A completion
    # verifies only once this is empty, so re-reading ONE of two edited files
    # (or an unchanged neighbour) is not enough. The adapter resets it each
    # user turn (an unverified turn was already reported via
    # exited_unverified); moves and deletes transfer or clear entries, so an
    # entry can always be paid. See note_mutation / cover_pending.
    pending_verification: set = field(default_factory=set)
    # owed entry -> the other path that also pays it (an ambiguous `mv a b`)
    pending_aliases: dict = field(default_factory=dict)
    # per-turn latches
    grounded_this_turn: bool = False
    verified_this_turn: bool = False
    # per-turn qualifying observations retained for the optional verifier tier
    # (verify_with) and progress() telemetry. Reset with the other per-turn
    # latches by the wiring layer (adapter user_prompt_submit); turn_loop is
    # single-turn so it starts empty. Populated in the WIRING layer only, never
    # in classify_observation.
    turn_observations: list = field(default_factory=list)
    # declarative rails
    verified_signals: set = field(default_factory=set)
    goal_predicates: list = field(default_factory=list)
    halted: bool = False
    # granularity gate for the optional verify_with tier: a verifier confidence
    # strictly below this DOWNGRADES a structural ACCEPT to the typed
    # `unverified` path. Literal 0.5 mirrors verifiers.GRANULARITY (NOT imported
    # here — state.py stays free of the optional subpackage so the floor never
    # reaches toward it).
    verify_threshold: float = 0.5

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

    def note_mutation(self, args, cwd="", output="", failed=False, home=None):
        """Record a mutating call at ``current_step``.

        A completion now needs a fresh read AFTER this step, earlier
        verification stops counting, and what the call changed is owed a
        re-read. For a shell command (``{"command": ...}``) the effects are
        applied in the order the command ran them (``shell.effects``): a
        file it writes is owed, a move carries debt to the destination, an
        ``rm`` clears it, and content the command itself showed afterwards
        pays it. A temp file written and then moved or deleted in the same
        command is therefore never left owed. ``failed`` calls may have
        stopped anywhere, so only their reset applies: nothing is cleared,
        moved or paid on their behalf.
        """
        self.last_mutation_step = self.current_step
        self.verified_this_turn = False
        command = args.get("command") if isinstance(args, dict) else None
        if not isinstance(command, str):
            self.pending_verification |= mutation_targets(args, self.claim_surface, cwd)
            self.claim_surface |= mutation_identifiers(args)
            src, dst = (args.get("source"), args.get("destination")) \
                if isinstance(args, dict) else (None, None)
            if isinstance(src, str) and isinstance(dst, str) and not failed:
                self._move(_under(cwd, src), _under(cwd, dst), None)
            return
        if failed:
            return
        from . import shell
        effs = shell.effects(command, cwd, output, home)
        if effs is None:
            return                     # unparseable: freshness still applies
        for eff in effs:
            kind = eff[0]
            if kind == "write":
                self.pending_verification.add(eff[1])
                self.claim_surface.add(eff[1])
            elif kind == "remove":
                for e in _removed({eff[1]}, self.pending_verification, eff[2],
                                  self.pending_aliases):
                    self._drop(e)
            elif kind == "move":
                self._move(*eff[1:])
            elif kind == "read":
                for e in self._owed_hits({eff[1]}):
                    self._drop(e)
        # content the command showed after its last change verifies it, as
        # a separate re-read would (`npm test > log; tail log`)
        last_change = max((k for k, e in enumerate(effs) if e[0] != "read"),
                          default=-1)
        trailing = {e[1] for e in effs[last_change + 1:] if e[0] == "read"}
        if trailing and not self.pending_verification and surface_hits(
                trailing, self.claim_surface):
            self.verified_this_turn = True
            self.last_verification_step = self.current_step

    def _drop(self, entry):
        self.pending_verification.discard(entry)
        self.pending_aliases.pop(entry, None)

    def _owed_hits(self, idents):
        """Owed entries ``idents`` pay: the entry's path, or its alias."""
        hits = surface_hits(idents, self.pending_verification)
        by_alias = {a: e for e, a in self.pending_aliases.items()
                    if e in self.pending_verification}
        hits |= {by_alias[a] for a in surface_hits(idents, set(by_alias))}
        return hits

    def _move(self, src, dst, alt):
        """Carry debt from ``src`` (a file, or a directory holding owed
        files) to ``dst``. ``alt`` is the other reading of an ambiguous
        ``mv a b``: b may be an existing directory, so b/a also pays. An
        entry is found by its own path or by its alias, so a file moved
        into a directory can be moved again."""
        inside = "/" + _canonical(src).rstrip("/") + "/"
        for e in list(self.pending_verification):
            for candidate in (e, self.pending_aliases.get(e)):
                if not candidate:
                    continue
                ce = "/" + _canonical(candidate)
                if surface_hits({src}, {candidate}):
                    new, alias = dst, alt
                elif inside in ce:
                    rest = ce[ce.index(inside) + len(inside):]
                    new = posixpath.join(dst, rest)
                    alias = posixpath.join(alt, rest) if alt else None
                else:
                    continue
                self._drop(e)
                self.pending_verification.add(new)
                self.claim_surface.add(new)
                if alias:
                    self.pending_aliases[new] = alias
                    self.claim_surface.add(alias)
                break

    def observation_identifiers(self, tool_name, args, result):
        """Identifiers one tool call touched (per-tool extractor or default).

        A bare string from an extractor is ONE identifier — set("app.cfg")
        would explode it into characters and silently break relevance.
        """
        ids = self.extractors.get(tool_name, extract_identifiers)(args, result)
        return {ids} if isinstance(ids, str) else set(ids)

    def cover_pending(self, tool_name, args, result, idents=None):
        """Mark the owed targets this verified-tier read covers; return True
        once nothing is owed (the completion is fully verified). ``idents``
        overrides what the read covers (a shell read covers only the files
        whose content it showed, not ones it merely listed)."""
        if idents is None:
            idents = self.observation_identifiers(tool_name, args, result)
        for e in self._owed_hits(idents):
            self._drop(e)
        return not self.pending_verification

    def progress(self):
        """Zero-token telemetry snapshot — a PURE function of GateState.

        Makes NO tool call and NO model call: the structural analog of the
        paper's Claude Code progress monitor, computed only from the integers,
        booleans, and sets the gate already tracks. Safe to call anywhere; it
        mutates nothing (call it twice, get equal dicts).

        Two fields carry semantics a casual reader might misjudge, so they are
        spelled out here:
          * ``rejection_count`` is TASK-CUMULATIVE, not per-turn — every REJECT
            at the boundary (structural or verifier-downgrade) increments it and
            it is never reset. A consumer wanting per-turn counts must diff
            snapshots itself.
          * ``steps_since_last_verification`` is derived from the MONOTONIC
            ``last_verification_step`` marker, which is not reset per turn, so
            it can report a positive delta even while ``verified_this_turn`` is
            already False (it means "N steps since anything was verified",
            spanning turn boundaries). ``None`` means nothing has verified yet.
        ``steps_since_last_mutation`` mirrors that (``None`` = no mutation ever).
        """
        return {
            "budget": self.budget,
            "cap": self.cap,
            "refill": self.refill,
            "budget_headroom": self.cap - self.budget,
            "step": self.current_step,
            "grounded_this_turn": self.grounded_this_turn,
            "verified_this_turn": self.verified_this_turn,
            "rejection_count": self.rejection_count,
            "steps_since_last_mutation": (
                self.current_step - self.last_mutation_step
                if self.last_mutation_step > 0 else None),
            "steps_since_last_verification": (
                self.current_step - self.last_verification_step
                if self.last_verification_step > 0 else None),
            "pending_verification": sorted(self.pending_verification),
            "halted": self.halted,
            "observations_this_turn": len(self.turn_observations),
            "unmet_signals": [s for s in self.goal_predicates
                              if s not in self.verified_signals],
        }


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


def _canonical(ident):
    while ident.startswith("./"):
        ident = ident[2:]
    if "/" in ident:
        # src/../app.cfg is app.cfg (normpath keeps a URL's leading //)
        ident = posixpath.normpath(ident)
    return ident


class Symbol(str):
    """An identifier read from file TEXT (``parse_config``), not a path the
    call touched. It matches only a claim-surface entry that is itself a
    bare name, so a README that mentions the ``Makefile`` can't pay the debt
    owed by ``/app/Makefile``."""


def surface_hits(idents, surface):
    """The claim-surface entries that ``idents`` refer to.

    Paths are compared by trailing components at ``/`` boundaries (after
    dropping a leading ``./``), so ``app.cfg``, ``./app.cfg``,
    ``proj/app.cfg`` and ``/srv/proj/app.cfg`` all name the same file, while
    ``/etc/app.cfg`` and ``/srv/proj/app.cfg`` stay distinct (neither is a
    tail of the other). Non-path identifiers still need an exact match, and
    a ``Symbol`` matches only a bare-name entry.
    """
    symbols = {str(i) for i in idents if isinstance(i, Symbol)}
    idents = {_canonical(i) for i in idents if not isinstance(i, Symbol)}
    # every trailing part of every LOCAL path identifier, computed once so a
    # large surface against a large output stays linear ("//host/app.cfg",
    # from a URL, names a remote file and contributes no tails)
    tails = set()
    for i in idents:
        if "/" in i and not i.startswith("//"):
            parts = i.split("/")
            tails.update("/".join(parts[k:]) for k in range(1, len(parts)))
    hits = set()
    for entry in surface:
        s = _canonical(entry)
        if s in idents or s in tails or ("/" not in entry and entry in symbols):
            hits.add(entry)
            continue
        # an identifier that is a tail of the surface path ("app.cfg" for
        # "/srv/proj/app.cfg")
        parts = s.split("/")
        if any("/".join(parts[k:]) in idents for k in range(1, len(parts))):
            hits.add(entry)
    return hits


# tool_input keys whose VALUES name what was touched
_PATH_KEYS = ("file_path", "path", "notebook_path", "filename", "file")


def path_values(tool_input):
    """Whole path strings from a structured call's path keys. A path is one
    identifier: splitting ``a.cfg~`` or ``my notes.txt`` into word tokens
    would make different files alias each other."""
    if not isinstance(tool_input, dict):
        return set()
    return {str(v) for k, v in tool_input.items()
            if k in _PATH_KEYS and isinstance(v, (str, int, float)) and str(v)}


def mutation_identifiers(tool_input):
    """Identifiers of WHAT a mutation touched — extracted from VALUES only.

    JSON schema keys (``file_path``, ``content``) are shared across every
    file tool; letting them into the claim surface would make a read of ANY
    file pass the relevance gate. Path-key values are used whole when
    present; otherwise all values contribute. (Shell commands are handled by
    ``note_mutation``, which adds only the files a command writes, never
    words from its description.)
    """
    if isinstance(tool_input, dict):
        paths = path_values(tool_input)
        if paths:
            return paths
        out = set()
        for v in tool_input.values():
            out |= mutation_identifiers(v)
        return out
    if isinstance(tool_input, (list, tuple)):
        out = set()
        for v in tool_input:
            out |= mutation_identifiers(v)
        return out
    if tool_input is None:
        return set()
    return extract_identifiers(str(tool_input), "")


def mutation_targets(args, surface, cwd=""):
    """What a structured or free-text mutating call verifiably changed.

    Path-key values of a structured call (``{"file_path": ...}``), or, for
    free-text args, the claim-surface entries they name. Shell commands are
    handled by ``note_mutation`` via ``shell.effects``; anything vaguer
    returns nothing and the completion falls back to the freshness check,
    because a target no read could match would trap the agent.
    """
    if isinstance(args, dict):
        return path_values(args)
    if isinstance(args, str):
        return surface_hits(extract_identifiers(args, ""), surface)
    if isinstance(args, (list, tuple)):
        return {t for a in args for t in mutation_targets(a, surface, cwd)}
    return set()


def _under(cwd, path):
    return path if path.startswith("/") or not cwd else posixpath.join(cwd, path)


def _removed(operands, pending, recursive=True, aliases=None):
    """The owed entries an ``rm`` of ``operands`` deleted: the files
    themselves (path matching as in ``surface_hits``), entries under a
    removed directory when ``recursive``, and glob matches. A glob is
    matched component by component (``tmp/*.txt`` never reaches into
    ``tmp/sub/``, but ``rm -r tmp/*`` removes what is under ``tmp/sub``).
    An entry also counts as removed when its alias path is."""
    aliases = aliases or {}
    hits = set()
    for entry in pending:
        for candidate in (entry, aliases.get(entry)):
            if candidate and _removes_path(operands, candidate, recursive):
                hits.add(entry)
                break
    return hits


def _removes_path(operands, path, recursive):
    cpath = _canonical(path)
    for operand in operands:
        op = _canonical(operand).rstrip("/") or "/"
        if set(op) & set("*?["):
            parts = cpath.strip("/").split("/")
            depth = op.strip("/").count("/") + 1
            if op.startswith("/") and cpath.startswith("/"):
                if len(parts) >= depth and fnmatch.fnmatchcase(
                        "/" + "/".join(parts[:depth]), op) and (
                        len(parts) == depth or recursive):
                    return True
            elif fnmatch.fnmatchcase("/".join(parts[-depth:]), op.lstrip("/")):
                return True
            continue
        if surface_hits({operand}, {path}):
            return True
        if recursive and ("/" + op.strip("/") + "/") in ("/" + cpath):
            return True
    return False
