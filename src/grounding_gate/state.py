"""Gate state, model-class presets, and the normalization helpers.

Spec modules 1 (state container) and 6 (per-model-class presets): fleet
variance is absorbed as integers, not prose.
"""

import fnmatch
import posixpath
import re
import shlex
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

    def note_mutation(self, args, cwd=""):
        """Record a mutating call at ``current_step``.

        A completion now needs a fresh read AFTER this step, earlier
        verification stops counting, the mutated identifiers join the claim
        surface, and the call's targets are owed a re-read (see
        ``mutation_targets``).
        """
        self.last_mutation_step = self.current_step
        self.verified_this_turn = False
        if isinstance(args, dict) and isinstance(args.get("command"), str):
            # a deleted file (or a file under a deleted directory) can never
            # be re-read: drop what it owed
            self.pending_verification -= _removed(
                shell_remove_targets(args["command"], cwd), self.pending_verification)
        for src, dst in move_pairs(args, cwd):
            # a moved file's debt moves with it: re-read it at its new path
            moved = _removed({src}, self.pending_verification)
            if moved:
                self.pending_verification -= moved
                self.pending_verification.add(dst)
                self.claim_surface.add(dst)
        self.pending_verification |= mutation_targets(args, self.claim_surface, cwd)
        self.claim_surface |= mutation_identifiers(args)

    def observation_identifiers(self, tool_name, args, result):
        """Identifiers one tool call touched (per-tool extractor or default).

        A bare string from an extractor is ONE identifier — set("app.cfg")
        would explode it into characters and silently break relevance.
        """
        ids = self.extractors.get(tool_name, extract_identifiers)(args, result)
        return {ids} if isinstance(ids, str) else set(ids)

    def cover_pending(self, tool_name, args, result):
        """Mark the owed targets this verified-tier read covers; return True
        once nothing is owed (the completion is fully verified)."""
        idents = self.observation_identifiers(tool_name, args, result)
        self.pending_verification -= surface_hits(idents, self.pending_verification)
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


def surface_hits(idents, surface):
    """The claim-surface entries that ``idents`` refer to.

    Paths are compared by trailing components at ``/`` boundaries (after
    dropping a leading ``./``), so ``app.cfg``, ``./app.cfg``,
    ``proj/app.cfg`` and ``/srv/proj/app.cfg`` all name the same file, while
    ``/etc/app.cfg`` and ``/srv/proj/app.cfg`` stay distinct (neither is a
    tail of the other). Non-path identifiers still need an exact match.
    """
    idents = {_canonical(i) for i in idents}
    hits = set()
    for entry in surface:
        s = _canonical(entry)
        if s in idents:
            hits.add(entry)
            continue
        # an identifier that is a tail of the surface path ("app.cfg" for
        # "/srv/proj/app.cfg"), or a longer LOCAL path ending in the surface
        # entry ("//host/app.cfg", from a URL, names a remote file)
        parts = s.split("/")
        if any("/".join(parts[k:]) in idents for k in range(1, len(parts))) or \
                any(i.endswith("/" + s) for i in idents
                    if "/" in i and not i.startswith("//")):   # not a URL
            hits.add(entry)
    return hits


# tool_input keys whose VALUES name what was touched
_PATH_KEYS = ("file_path", "path", "notebook_path", "filename", "file")


def mutation_identifiers(tool_input):
    """Identifiers of WHAT a mutation touched — extracted from VALUES only.

    JSON schema keys (``file_path``, ``content``) are shared across every
    file tool; letting them into the claim surface would make a read of ANY
    file pass the relevance gate. Path-like values are preferred when
    present; otherwise all values contribute.
    """
    if isinstance(tool_input, dict):
        path_vals = [v for k, v in tool_input.items()
                     if k in _PATH_KEYS and isinstance(v, (str, int, float))]
        if path_vals:
            out = set()
            for v in path_vals:
                out |= extract_identifiers(str(v), "")
            return out
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
    """What a mutating call verifiably changed, for per-target coverage.

    Path-key values of a structured call (``{"file_path": ...}``), or, for
    free-text args, the claim-surface entries they name. Anything vaguer (a
    shell command with no path key) returns nothing and the completion falls
    back to the freshness check alone, because an unparseable target could
    never be covered and would trap the agent.
    """
    if isinstance(args, dict):
        targets = {i for k, v in args.items()
                   if k in _PATH_KEYS and isinstance(v, (str, int, float))
                   for i in extract_identifiers(str(v), "")}
        if not targets and isinstance(args.get("command"), str):
            targets = shell_write_targets(args["command"], cwd)
        return targets
    if isinstance(args, str):
        return surface_hits(extract_identifiers(args, ""), surface)
    if isinstance(args, (list, tuple)):
        return {t for a in args for t in mutation_targets(a, surface, cwd)}
    return set()


_SHELL_SEP = re.compile(r"\|\||&&|[;|&\n]")
_SHELL_SEP_KEEP = re.compile(r"(\|\||&&|[;|&\n])")
# plain redirects only: `2>err.log` and `>&2` are diagnostics, not the edit
_REDIRECT = re.compile(r"(?<![\d&>])>>?[ \t]*([^\s;|&<>]+)")


def _segments(command, cwd=""):
    """``(segment, argv, resolve, piped)`` per simple command in ``command``;
    ``piped`` is True when its output feeds the next command through ``|``.

    ``argv`` is None when the segment doesn't parse. ``resolve(path)`` maps a
    relative path through ``cwd`` (the session's working directory, when
    known) and any earlier ``cd`` in the same command
    (``cd conf && cat app.cfg`` reads ``conf/app.cfg``); it returns None
    once a ``cd`` goes somewhere unknown (``cd``, ``cd -``, ``cd ~``,
    ``cd $DIR``), so callers skip what they can't place.
    """
    cwd, saved = posixpath.normpath(cwd) if cwd else "", []
    pieces = _SHELL_SEP_KEEP.split(command)
    for k in range(0, len(pieces), 2):
        segment = pieces[k]
        piped = k + 1 < len(pieces) and pieces[k + 1] == "|"
        # `(cd conf && cat x)`: a subshell's cd ends with its closing paren
        body = _HARMLESS_REDIRECTS.sub("", segment).strip()
        opens = len(body) - len(body.lstrip("("))
        closes = len(body) - len(body.rstrip(")"))
        saved += [cwd] * opens
        try:
            argv = shlex.split(body.strip("()"))
        except ValueError:
            argv = None

        def resolve(path, cwd=cwd):
            if path.startswith("/") or cwd == "":
                return path
            return None if cwd is None else posixpath.join(cwd, path)

        yield segment, argv, resolve, piped
        if argv and argv[0] == "cd":
            target = argv[1] if len(argv) > 1 else "~"
            if target == "-" or set(target) & set("~$`"):
                cwd = None
            elif target.startswith("/"):
                cwd = posixpath.normpath(target)
            elif cwd is not None:
                cwd = posixpath.normpath(posixpath.join(cwd, target))
        for _ in range(min(closes, len(saved))):
            cwd = saved.pop()


def shell_write_targets(command, cwd=""):
    """Files a shell command explicitly writes, from a few unambiguous forms:
    ``> FILE``, ``>> FILE``, ``tee [-a] FILE...`` and ``sed -i ... FILE...``.

    Anything else (``pytest x.py``, ``python3.11 s.py``, ``cat app.cfg``)
    yields nothing, so read-only shell use never leaves a re-read owed. Paths
    built from variables or globs are skipped too: a target that can never be
    matched by a later read would trap the agent.
    """
    out = set()
    for segment, argv, resolve, _ in _segments(command, cwd):
        files = _REDIRECT.findall(segment)
        argv = argv or []
        if argv and argv[0] == "tee":
            files += [a for a in argv[1:] if not a.startswith("-")]
        elif argv and argv[0] == "sed" and any(
                a.startswith("-i") or a == "--in-place" for a in argv[1:]):
            rest, has_e, skip = [], False, False
            for a in argv[1:]:
                if skip:
                    skip = False
                elif a in ("-e", "-f", "--expression", "--file"):
                    has_e, skip = True, True
                elif not a.startswith("-"):
                    rest.append(a)
            files += rest if has_e else rest[1:]   # first operand is the script
        out |= {r for r in (resolve(f) for f in files
                            if f != "/dev/null" and not set(f) & set("$*?`{")) if r}
    return _identifiers_of(out)


# programs that only read (no flag of theirs used here writes a file); `sed`,
# `awk`, `sort` and `find` are left out because a script or flag can write
_READ_ONLY_PROGRAMS = frozenset({
    "cat", "head", "tail", "grep", "egrep", "fgrep", "rg", "ls", "wc", "diff",
    "cmp", "stat", "file", "nl", "cut", "tr", "jq", "tree", "du", "df", "pwd",
    "cd", "echo", "printf", "which", "md5sum", "sha256sum", "true", ":"})
_READ_ONLY_GIT = frozenset({"diff", "show", "log", "status", "blame", "ls-files"})
_HARMLESS_REDIRECTS = re.compile(r"\d?>\s*/dev/null|\d?>&\d")


def _removed(operands, pending):
    """The owed entries an ``rm`` of ``operands`` deleted: path matches as in
    ``surface_hits``, entries under a removed directory, and glob operands
    matched against each entry's trailing components (``tmp*.txt`` removes
    ``/srv/proj/tmp1.txt``)."""
    hits = surface_hits(_identifiers_of(operands), pending)
    for operand in operands:
        # `rm -r build` removes everything owed under build/
        inside = "/" + _canonical(operand).rstrip("/") + "/"
        hits |= {e for e in pending if inside in "/" + _canonical(e)}
    for pattern in (o[2:] if o.startswith("./") else o for o in operands):
        if set(pattern) & set("*?["):
            depth = pattern.count("/") + 1
            hits |= {e for e in pending
                     if fnmatch.fnmatchcase("/".join(e.split("/")[-depth:]), pattern)}
    return hits


def _under(cwd, path):
    return path if path.startswith("/") or not cwd else posixpath.join(cwd, path)


def move_pairs(args, cwd=""):
    """``(source, destination)`` for each file a call moves: ``mv`` and
    ``git mv`` in a shell command, or a structured ``source``/``destination``
    call (the MCP filesystem server's ``move_file``). With several sources,
    or a destination ending in ``/``, the destination is a directory."""
    if not isinstance(args, dict):
        return []
    src, dst = args.get("source"), args.get("destination")
    if isinstance(src, str) and isinstance(dst, str):
        return [(_under(cwd, src), _under(cwd, dst))]
    command = args.get("command")
    if not isinstance(command, str):
        return []
    pairs = []
    for _, argv, resolve, _ in _segments(command, cwd):
        if not argv:
            continue
        if argv[0] == "mv":
            operands = argv[1:]
        elif argv[:2] == ["git", "mv"]:
            operands = argv[2:]
        else:
            continue
        operands = [o for o in operands
                    if not o.startswith("-") and not set(o) & set("$`{*?")]
        if len(operands) < 2:
            continue
        *sources, target = operands
        into_dir = len(sources) > 1 or target.endswith("/")
        for s in sources:
            d = posixpath.join(target, posixpath.basename(s)) if into_dir else target
            if resolve(s) and resolve(d):
                pairs.append((resolve(s), resolve(d)))
    return pairs


def shell_remove_targets(command, cwd=""):
    """Path operands of any ``rm`` in a shell command, globs included
    (``tmp*.txt``); operands built from variables are skipped."""
    out = set()
    for _, argv, resolve, _ in _segments(command, cwd):
        if argv and argv[0] == "rm":
            out |= {r for r in (resolve(a) for a in argv[1:]
                                if not a.startswith("-") and not set(a) & set("$`{"))
                    if r}
    return out


# How a read-only program relates to the files it is given: CONTENT programs
# print what the files say (can verify a change), LISTING programs report
# that they exist or their size (an assertion at most), and anything else in
# _READ_ONLY_PROGRAMS (echo, pwd, ...) observes no file at all.
_CONTENT_PROGRAMS = frozenset({"cat", "head", "tail", "nl", "diff", "cmp",
                               "cut", "md5sum", "sha256sum"})
# ...of which these print the lines themselves, not a digest or a verdict
_PASS_THROUGH_PROGRAMS = frozenset({"cat", "head", "tail", "nl", "diff", "cut", "tr",
                                    "json.tool"})
_PATTERN_PROGRAMS = frozenset({"grep", "egrep", "fgrep", "rg", "jq"})
# sed/awk run as pure viewers (see _is_viewer): script first, then files
_VIEWER_PROGRAMS = frozenset({"sed", "awk"})
# sed scripts that only print: "p", "5,9p", "$p", "/re/p", "="
_SED_ADDR = r"(?:\d+|\$|/[^/]*/)"
_SED_PRINT_ONLY = re.compile(
    r"^\s*(?:%s(?:\s*,\s*%s)?)?\s*[p=]?\s*$" % (_SED_ADDR, _SED_ADDR))
_AWK_SIDE_EFFECTS = ("system", "getline", "|", ">", "close", "fflush")


_PYTHON = re.compile(r"^python(?:3(?:\.\d+)?)?$")


def _is_json_tool(argv):
    """``python -m json.tool [FILE]``: pretty-prints only. A second operand
    would be an output file, so that form doesn't qualify."""
    return (len(argv) >= 3 and bool(_PYTHON.match(argv[0]))
            and argv[1:3] == ["-m", "json.tool"]
            and len([a for a in argv[3:] if not a.startswith("-")]) <= 1)


def _is_viewer(argv):
    """``sed`` with no in-place flag and print-only scripts, or ``awk`` whose
    program can't run commands or write files: both only print file content."""
    args = argv[1:]
    if argv[0] == "sed":
        if any(a.startswith("-i") or a.startswith("--in-place") or a in ("-f", "--file")
               for a in args):
            return False
        scripts = [args[k + 1] for k, a in enumerate(args[:-1])
                   if a in ("-e", "--expression")]
        if not scripts:
            operands = [a for a in args if not a.startswith("-")]
            scripts = operands[:1]
        return bool(scripts) and all(_SED_PRINT_ONLY.match(x) for x in scripts)
    if argv[0] == "awk":
        if "-f" in args:
            return False
        operands = [a for a in args if not a.startswith("-")]
        return bool(operands) and not any(t in operands[0] for t in _AWK_SIDE_EFFECTS)
    return False
_LISTING_PROGRAMS = frozenset({"ls", "stat", "file", "wc", "du", "tree"})
_GIT_CONTENT = frozenset({"diff", "show", "blame"})
_GIT_LISTING = frozenset({"status", "log", "ls-files"})
_DIFF_NEW_FILE = re.compile(r"^\+\+\+ b/(\S+)", re.M)
_OUTPUT_PATH = re.compile(r"^([^\s:]+):", re.M)


def shell_read_operands(command, output="", cwd=""):
    """``(content, listed)``: identifiers of the files a read-only shell
    command shows the content of, and of those it only lists.

    Operands only: a grep/rg/jq pattern or an ``echo`` argument names nothing
    that was read. Pattern programs also credit the ``path:`` prefixes of
    their output lines, which is how a recursive search reports whose content
    matched.
    """
    content, listed = set(), set()
    stage_content, stage_listed = set(), set()   # of the current pipeline
    for segment, argv, resolve, piped in _segments(command, cwd):
        if not argv:
            continue
        seen_content, seen_listed = set(), set()
        prog, args = argv[0], argv[1:]
        if prog == "git" and args:
            prog, args = "git " + args[0], args[1:]
        elif _is_json_tool(argv):
            prog, args = "json.tool", argv[3:]
        operands = [a for a in args if not a.startswith("-")]
        if prog in _VIEWER_PROGRAMS:
            # the script comes first unless given with -e / -f
            seen_content.update(operands if any(a in ("-e", "-f") for a in args)
                                else operands[1:])
        elif prog in _PATTERN_PROGRAMS:
            if not any(a in ("-e", "-f") or a.startswith("--regexp") for a in args):
                operands = operands[1:]   # the first operand is the pattern
            seen_content.update(_OUTPUT_PATH.findall(str(output)))
            seen_content.update(operands)
        elif prog.startswith("git ") and prog[4:] in _GIT_CONTENT:
            # REV:path is a committed copy, not the working tree; a diff
            # names the working-tree files it shows in its +++ headers
            seen_content.update(o for o in operands if ":" not in o)
            if prog == "git diff":
                seen_content.update(_DIFF_NEW_FILE.findall(str(output)))
        elif prog in _CONTENT_PROGRAMS or prog == "json.tool":
            seen_content.update(operands)
        elif prog in _LISTING_PROGRAMS or (
                prog.startswith("git ") and prog[4:] in _GIT_LISTING):
            seen_listed.update(operands)
        stage_content |= {r for r in map(resolve, seen_content) if r}
        stage_listed |= {r for r in map(resolve, seen_listed) if r}
        if piped:
            continue
        # the pipeline's LAST stage decides what the agent actually saw:
        # content only if it passes lines through to the agent
        if _shows_content(prog, args) and not _STDOUT_DISCARDED.search(segment):
            content |= stage_content
        else:
            listed |= stage_content
        listed |= stage_listed
        stage_content, stage_listed = set(), set()
    return _identifiers_of(content), _identifiers_of(listed)


# stdout (not just stderr) thrown away: `> /dev/null`, `&> /dev/null`
_STDOUT_DISCARDED = re.compile(r"(?:^|[^\d>&])&?>\s*/dev/null")
_QUIET_FLAGS = frozenset("qclL")   # grep/rg: quiet, count, files-only


def _shows_content(prog, args):
    """Whether a pipeline's last program passes file lines to the agent
    (``cat``, ``head``, ``grep``, ...) rather than a count, a checksum,
    or nothing (``wc``, ``md5sum``, ``grep -q``)."""
    if prog in _PATTERN_PROGRAMS:
        return not any(
            a in ("--quiet", "--silent", "--count", "--files-with-matches",
                  "--files-without-match")
            or (a.startswith("-") and not a.startswith("--")
                and set(a[1:]) & _QUIET_FLAGS)
            for a in args)
    return (prog in _PASS_THROUGH_PROGRAMS or prog in _VIEWER_PROGRAMS
            or (prog.startswith("git ") and prog[4:] in _GIT_CONTENT))


def _identifiers_of(paths):
    return {i for p in paths for i in extract_identifiers(p, "")}


def shell_is_read_only(command):
    """True when every piece of a shell command is a known read-only program
    with no output redirect, ``tee``, or command substitution. Anything this
    can't vouch for is treated as a possible write, as before."""
    if any(t in command for t in ("`", "$(", "<(", ">(")):
        return False
    command = _HARMLESS_REDIRECTS.sub("", command)   # before `&` splits 2>&1
    if ">" in command:
        return False
    saw_program = False
    for _, argv, _, _ in _segments(command):
        if argv is None:
            return False
        if not argv:
            continue
        saw_program = True
        if argv[0] == "git":
            if len(argv) < 2 or argv[1] not in _READ_ONLY_GIT:
                return False
        elif argv[0] in _VIEWER_PROGRAMS:
            if not _is_viewer(argv):
                return False
        elif _is_json_tool(argv):
            pass
        elif argv[0] not in _READ_ONLY_PROGRAMS:
            return False
    return saw_program
