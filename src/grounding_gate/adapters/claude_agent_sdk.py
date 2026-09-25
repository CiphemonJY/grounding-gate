"""Claude Agent SDK adapter — wire the gate into an SDK agent with hooks.

Maps the gate's choke point onto the Agent SDK's hook events:

====================  =======================================================
SDK hook event        Gate role
====================  =======================================================
``PostToolUse``       Observation classifier: every successful tool result is
                      classified (novel ∧ relevant ∧ consequence-tier).
                      Latches update as in ``turn_loop``; budget REFILLS on
                      qualifying observations, but note the asymmetry below.
``PostToolUseFailure``A FAILED mutating call may still have had an effect, so
                      it conservatively records a mutation (verification is
                      demanded) while earning no grounding credit.
``Stop``              THE submit boundary: when the agent tries to finish,
                      ``boundary_check`` runs. On REJECT the hook returns
                      ``{"decision": "block", "reason": ...}`` — the SDK
                      feeds the reason (the gate's legal_next) back to the
                      model and the run continues.
``UserPromptSubmit``  Turn boundary: per-turn latches, the block counter, and
                      the reasoning budget reset.
====================  =======================================================

The SDK has no typed terminals, so the gate's first-class ``unverified`` exit
is realized as an escape valve: after ``max_blocks`` rejected stop attempts —
or when the reasoning budget exhausts, whichever comes first — the stop is
allowed, ``gate.exited_unverified`` is set (the programmatic marker headless
consumers should check), and a ``systemMessage`` warning is returned. Note the
``systemMessage`` is a USER-facing display string per the SDK contract — the
model never sees it, and headless runs only surface it with the SDK's
``include_hook_events`` enabled. The gate never traps an agent — it only
forbids *confident* ungrounded claims (see the design spec).

Subagent events (hook inputs carrying ``agent_id``) are ignored by default so
a subagent's observations can't ground the main agent's claims; pass
``gate_subagents=True`` to include them in the shared state instead.

BUDGET ASYMMETRY (be honest with yourself about this): the SDK has no hook
for a pure reasoning step, so unlike the reference ``turn_loop`` the budget
here never decrements on thinking — only on rejected finishes. The
anti-divergence floor in SDK integrations is therefore enforced primarily by
the ``max_blocks`` counter; the budget is a secondary backstop, not the
per-step rope it is in the reference loop.

Usage::

    from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions
    from grounding_gate.adapters.claude_agent_sdk import GateHooks

    gate = GateHooks(claim_surface={"app.cfg"}, model_class="default")
    options = ClaudeAgentOptions(hooks=gate.as_options_hooks())

This module imports nothing from the SDK at module load — only
``as_options_hooks()`` needs ``claude-agent-sdk`` installed. The hook
callables themselves are plain async functions you can also register by hand.
"""

import json
import os
import posixpath
import re

from ..boundary import ACCEPT, boundary_check
from ..classifier import classify_observation
from .. import shell
from ..state import (_PATH_KEYS, UNPLACED, GateState, Symbol, _trusted,
                     extract_identifiers, path_values, surface_hits)

# Built-in SDK tools by consequence class. Unknown tools (MCP tools other than
# the reference filesystem server's, see _MCP_FILESYSTEM) are treated as
# NEITHER: they earn no grounding credit and record no mutation — maximally
# conservative in both directions. Override per-agent.
DEFAULT_READ_ONLY_TOOLS = frozenset(
    {"Read", "Glob", "Grep", "WebFetch", "WebSearch", "NotebookRead"})
# Read-only tools that fetch REMOTE text. A page or search result that names
# a local file is not a read of that file: they never verify a change, and
# only the symbols in their text (never paths) count toward relevance.
DEFAULT_REMOTE_TOOLS = frozenset({"WebFetch", "WebSearch"})
# Read-only tools that report what EXISTS (names, paths), not what a file now
# says. They can ground an assertion but never verify a change.
DEFAULT_LISTING_TOOLS = frozenset({"Glob"})
# Read-only tools whose output is the content OF the path they were given.
# Their relevance comes from that path, not from names the text mentions (a
# notes file saying "bump app.cfg" is not a read of app.cfg).
DEFAULT_CONTENT_TOOLS = frozenset({"Read", "NotebookRead"})
# Bash is classed as mutating because it CAN mutate. A command built only
# from known read-only programs (cat, grep, git diff, ...; see
# shell.is_read_only) is the exception: it is treated as a read.
DEFAULT_MUTATING_TOOLS = frozenset(
    {"Write", "Edit", "MultiEdit", "NotebookEdit", "Bash"})

# Tools that neither read nor change files (planning, questions, to-dos).
# Strict reads treats every tool outside the known sets as a possible change.
DEFAULT_NEUTRAL_TOOLS = frozenset(
    {"TodoWrite", "TodoRead", "ExitPlanMode", "AskUserQuestion",
     "BashOutput", "KillShell", "KillBash", "TaskOutput", "TaskStop"})
# tools that report on (or stop) a background command started with
# run_in_background, by the key naming it
_BACKGROUND_TOOLS = {"BashOutput": "bash_id", "KillShell": "shell_id",
                     "KillBash": "shell_id", "TaskOutput": "task_id",
                     "TaskStop": "task_id"}
_JOB_REPORTS = ("BashOutput", "TaskOutput")      # the rest stop the job
_JOB_DONE = re.compile(r"\s*<status>(?:completed|failed|killed|exited)</status>")
# read arguments that select part of a file
_PARTIAL_KEYS = ("offset", "limit", "head", "tail", "pages", "cell_id", "cell",
                 "start_line", "end_line", "line", "lines", "view_range", "range")
# MCP servers whose read tools strict reads trusts to show the LOCAL file
# (a server named "docker" or "remote" reads somewhere else)
DEFAULT_TRUSTED_MCP_SERVERS = frozenset({"filesystem", "fs"})

UNVERIFIED_BANNER = (
    "grounding-gate: exiting UNVERIFIED — the agent finished without a "
    "qualifying observation backing its claims.")


def _progress_line(p):
    """Compact one-line formatter for the opt-in emit_progress systemMessage.

    Takes the merged ``GateHooks.progress()`` dict; uses ``.get`` so it is
    robust to either the merged dict or the bare ``state.progress()`` dict.
    """
    return (
        "[progress budget=%s/%s step=%s grounded=%s verified=%s blocks=%s/%s "
        "rejections=%s obs=%s unmet=%s]" % (
            p.get("budget"), p.get("cap"), p.get("step"),
            p.get("grounded_this_turn"), p.get("verified_this_turn"),
            p.get("blocks"), p.get("max_blocks"),
            p.get("rejection_count"), p.get("observations_this_turn"),
            ",".join(p.get("unmet_signals") or []) or "-"))


def _serialize(value):
    """Deterministic text for hashing/identifier extraction."""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(value)


# an identifier that names a file: has a directory part or an extension
_PATH_LIKE = re.compile(r"/|\.[A-Za-z0-9]{1,8}$")


# Tools of the reference MCP filesystem server, by what they can prove. MCP
# tools arrive as mcp__<server>__<tool>, so any server name matches.
_MCP_FILESYSTEM = {
    "read_file": "content", "read_text_file": "content",
    "read_multiple_files": "content",
    "list_directory": "listing", "list_directory_with_sizes": "listing",
    "directory_tree": "listing", "search_files": "listing",
    "get_file_info": "listing",
    "write_file": "mutating", "edit_file": "mutating", "move_file": "mutating",
    # a new, empty directory changes no file's content
    "create_directory": "neutral",
}


def _truncated(response):
    """A Read response that shows only part of the file: Claude Code's
    ``{"file": {"startLine", "numLines", "totalLines"}}`` says so."""
    info = response.get("file") if isinstance(response, dict) else None
    if not isinstance(info, dict):
        return False
    start, num, total = (info.get(k) for k in ("startLine", "numLines", "totalLines"))
    if isinstance(start, int) and start > 1:
        return True
    return isinstance(num, int) and isinstance(total, int) and num < total


def _job_done(response):
    """A BashOutput response whose own status says the job ended (not text
    the job printed)."""
    if isinstance(response, dict):
        task = response.get("task")
        status = task.get("status") if isinstance(task, dict) else response.get("status")
        return status in ("completed", "failed", "killed", "exited", "stopped",
                          "cancelled", "error")
    return isinstance(response, str) and bool(_JOB_DONE.match(response))


def _job_entry(job):
    return UNPLACED + "background command %s still running" % job


def _mcp_server(tool):
    """The server of an ``mcp__<server>__<tool>`` name, or None when the
    name is ambiguous (``mcp__fs__remote__read_file``: server ``fs`` or
    ``fs__remote``?)."""
    parts = tool.split("__")
    return parts[1] if tool.startswith("mcp__") and len(parts) == 3 else None


def _mcp_class(tool):
    """``content`` / ``listing`` / ``mutating`` for a known MCP filesystem
    tool, else None (unknown tools stay neither, as before)."""
    if not tool.startswith("mcp__"):
        return None
    return _MCP_FILESYSTEM.get(tool.rsplit("__", 1)[-1])


_GREP_LISTING_MODES = ("files_with_matches", "count")


def _grep_lists_files(tool, tool_input, response, result):
    """True when a Grep call returned file names (its default
    ``files_with_matches`` mode) or counts rather than matching lines."""
    if tool != "Grep":
        return False
    mode = tool_input.get("output_mode") if isinstance(tool_input, dict) else None
    if isinstance(response, dict):
        mode = response.get("mode", mode)
    if mode is not None:
        return mode in _GREP_LISTING_MODES
    lines = [ln.strip() for ln in str(result).splitlines() if ln.strip()]
    return bool(lines) and all(
        ":" not in ln and not ln.split()[1:] and _PATH_LIKE.search(ln)
        for ln in lines)


def _symbols(text):
    """Non-path identifiers in file content (``parse_config`` in the source
    that defines it), with dotted names also split into their parts so
    ``settings.load_settings`` names ``load_settings``. They come back as
    ``Symbol``s, which match only bare-name surface entries: file NAMES the
    text mentions (``Makefile``, ``app.cfg``) were not read."""
    out = set()
    for i in extract_identifiers("", text):
        if not _PATH_LIKE.search(i):
            out.add(Symbol(i))
            out.update(Symbol(p) for p in i.split(".") if p)
    return out


def _decode(text):
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return None


def _output_text(response):
    """What a tool printed. Claude Code's Bash response is a dict with
    ``stdout``/``stderr``; line-based parsing needs the real newlines, not
    the escaped ones in its JSON serialization."""
    if isinstance(response, str):
        decoded = _decode(response)
        response = decoded if isinstance(decoded, dict) else response
    if isinstance(response, dict) and ("stdout" in response or "stderr" in response):
        return "\n".join(str(response.get(k) or "") for k in ("stdout", "stderr"))
    if isinstance(response, dict) and "content" in response:
        return str(response.get("content") or "")
    return response if isinstance(response, str) else _serialize(response)


def _content_identifiers(args, result):
    """Relevance for content tools: the path that was read (whole), plus the
    symbols in its text."""
    decoded = _decode(args)
    paths = path_values(decoded) if isinstance(decoded, dict) else set()
    return (paths or extract_identifiers(args, "")) | _symbols(_output_text(result))


def _remote_identifiers(args, result):
    """Relevance for web tools: symbols in the fetched text only."""
    return _symbols(_output_text(result))


def _grep_identifiers(args, result, cwd=""):
    """Relevance for the Grep tool: files whose lines it printed (the
    ``path:`` prefixes of content output, or the searched path when it is
    one file and lines carry no prefix), plus symbols in the matched text.
    A file named inside a pattern or a matched line was not searched."""
    decoded = _decode(args)
    searched = decoded.get("path") if isinstance(decoded, dict) else None
    searched = searched if isinstance(searched, str) and searched else None
    text = _output_text(result)
    # a prefix names a file only if it is path-like and, placed against the
    # session cwd, lies INSIDE what was searched. A single-file search
    # prints bare lines, so "a.cfg: ..." there is matched text (it can't be
    # inside /p/CHANGELOG), and "12:" is a line number.
    base = shell.resolve(searched or ".", cwd) or ""
    prefixed = set()
    for p in shell._OUTPUT_PATH.findall(text):
        placed = shell.resolve(p, cwd)
        if ("/" in p or "." in p) and placed and (
                base in ("", ".") and not placed.startswith("/")
                or placed.startswith(base.rstrip("/") + "/")):
            prefixed.add(placed)
    paths = prefixed or ({searched} if searched and text.strip() else set())
    return paths | _symbols(text)


def _absolutize(tool_input, cwd):
    """Resolve relative path-key values (``{"file_path": "app.cfg"}``)
    against the session's working directory, so ``app.cfg`` read from
    /srv/proj can't be taken for /srv/proj/conf/app.cfg."""
    if not cwd or not isinstance(tool_input, dict):
        return tool_input
    def place(v):
        return posixpath.join(cwd, v) if isinstance(v, str) and v \
            and not v.startswith("/") else v
    return {k: ([place(x) for x in v] if isinstance(v, list) else place(v))
            if k in _PATH_KEYS else v
            for k, v in tool_input.items()}


def _shell_identifiers(args, result, cwd="", home=None):
    """Relevance for a read-only shell call: the files it operates on (and
    the ``path:`` prefixes a multi-file search prints), never a grep pattern
    or an echo argument. Only consulted for shell reads: a mutating call
    grounds nothing whatever it touches."""
    decoded = _decode(args)
    command = decoded.get("command") if isinstance(decoded, dict) else None
    if not isinstance(command, str):
        return extract_identifiers(args, result)
    text = _output_text(result)
    content, listed = shell.read_operands(command, text, cwd, home)
    # output is file content only when a content program ran (not for a
    # bare `echo load_settings`)
    return content | listed | (_symbols(text) if content else set())


class GateHooks:
    """Stateful gate for one agent session.

    Create one instance per ``ClaudeSDKClient`` — the instance carries the
    ``GateState`` across hook invocations. (Hooks bind per client: if you
    multiplex several ``session_id`` values over one client they share this
    gate; give each conversation its own client + GateHooks.)

    Args:
        claim_surface: identifiers the agent's answers are about. Identifiers
            touched by MUTATING tool calls are added automatically ("you must
            verify what you changed"), so this can start empty for pure
            do-then-verify tasks.
        model_class: ``"default"`` | ``"skipper"`` | ``"diverger"`` preset.
            Note ``skipper`` (strict G) demands VERIFIED-tier grounding even
            for assertions — a post-mutation observation — so use it only for
            mutate-then-verify tasks; read-only/Q&A agents should use
            ``default``.
        read_only_tools / mutating_tools: override the consequence classes.
        content_tools: read-only tools whose output is the content of the
            path they were given (default ``{"Read", "NotebookRead"}``).
            Their relevance comes from that path only, never from names the
            text happens to mention. An entry in ``extractors`` overrides it.
        listing_tools: read-only tools whose output names things rather than
            showing their content (default ``{"Glob"}``). They can ground an
            assertion but never verify a change.
        max_blocks: rejected stop attempts before the escape valve allows an
            UNVERIFIED exit (the typed-``unverified`` analog). The valve also
            opens if the reasoning budget exhausts first.
        gate_subagents: include subagent hook events (``agent_id`` set) in
            this gate's state. Default False: subagent observations must not
            ground the main agent's claims.
        normalizers: per-tool novelty scrubbers, ``{tool_name: callable}``.
            A noisy tool (nonstandard timestamps, changing counters) gets its
            own scrubber; it runs before the built-in ``normalize()``. The
            callable receives the SERIALIZED tool input/response (JSON text),
            so write substring-safe regex scrubbers, and keep them
            deterministic.
        extractors: per-tool relevance extractors,
            ``{tool_name: callable(args, result) -> set}`` — for tools whose
            output lives in a different lexical domain than the claim
            surface (inodes, opaque handles). Receives serialized text.
        verifier: optional ``verify_with`` verifier (a
            ``grounding_gate.verifiers`` instance, e.g. ``StubVerifier`` or
            ``LLMVerifier``). When set, a finish the FLOOR would accept is
            additionally scored; a confidence below ``state.verify_threshold``
            DOWNGRADES it to the UNVERIFIED escape path (REJECT), which flows
            through the same block/budget/escape machinery — the verifier can
            add strictness but can never bypass the gate or trap the agent.
            Default None (the floor runs alone, zero-LLM).
        strict_reads: when True, only a direct file read can verify a change:
            a content tool (``Read``, ``NotebookRead``, an MCP ``read_file``
            style tool) of each changed file after its last change. The shell
            parser may add obligations but never relax one: every Bash call
            counts as an unknown change that owes a fresh Read of every file
            changed earlier this turn plus whatever it writes; nothing it
            prints counts as a read; its moves and deletes only apply to files
            it created itself. Grep, listings and web tools can ground an
            answer but never verify a change. A parser mistake can therefore
            only block honest work, never let an unbacked claim through. The
            cost: one more Read after running tests or scripts. Default False.
        strict_trusted_programs: with ``strict_reads``, programs whose writes
            the shell parser doesn't model but you know write no files, by
            name or name and leading words (``("pytest", "npm test")``).
            Any other such program ends the turn unverified. Default ().
        emit_progress: when True, the escape-valve ``systemMessage`` is suffixed
            with a compact ``progress()`` summary (a best-effort, user-facing
            event per the SDK contract). Default False. The reliable programmatic
            surface is ``progress()`` / ``exited_unverified``, not this string.

    Attributes:
        exited_unverified: True when the LAST allowed stop went through the
            escape valve rather than a verified/grounded finish. Check this
            programmatically in headless runs; it resets on the next turn.
    """

    def __init__(self, claim_surface=(), model_class="default",
                 read_only_tools=DEFAULT_READ_ONLY_TOOLS,
                 mutating_tools=DEFAULT_MUTATING_TOOLS,
                 listing_tools=DEFAULT_LISTING_TOOLS,
                 content_tools=DEFAULT_CONTENT_TOOLS,
                 remote_tools=DEFAULT_REMOTE_TOOLS,
                 max_blocks=3, gate_subagents=False,
                 normalizers=None, extractors=None,
                 verifier=None, emit_progress=False, home=None,
                 strict_reads=False, neutral_tools=DEFAULT_NEUTRAL_TOOLS,
                 trusted_mcp_servers=DEFAULT_TRUSTED_MCP_SERVERS,
                 strict_trusted_programs=()):
        extractors = dict(extractors or {})
        for tool in content_tools:
            extractors.setdefault(tool, _content_identifiers)
        for tool in remote_tools:
            extractors.setdefault(tool, _remote_identifiers)
        extractors.setdefault("Grep", self._grep_extract)
        for tool in mutating_tools:
            extractors.setdefault(tool, self._shell_extract)
        self.state = GateState.for_model_class(
            model_class, claim_surface=set(claim_surface),
            normalizers=dict(normalizers or {}), extractors=extractors)
        self.read_only_tools = set(read_only_tools)
        self.mutating_tools = set(mutating_tools)
        self.listing_tools = set(listing_tools)
        self.remote_tools = set(remote_tools)
        self.max_blocks = max_blocks
        self.gate_subagents = gate_subagents
        # optional verify_with tier: a verifier can DOWNGRADE a grounded finish
        # to the UNVERIFIED escape path, never bypass the floor (see boundary.py)
        self._verifier = verifier
        # opt-in: append a zero-token progress summary to the escape-valve
        # systemMessage (best-effort event; progress() is the reliable surface)
        self.emit_progress = emit_progress
        self.strict_reads = bool(strict_reads)
        self.content_tools = set(content_tools)
        self.neutral_tools = set(neutral_tools)
        self.trusted_mcp_servers = set(trusted_mcp_servers)
        self.strict_trusted_programs = tuple(strict_trusted_programs)
        # strict reads: background commands still running, by id, with what
        # they ran and where (their writes land whenever they finish)
        self._background = {}
        self._agent_cwds = {}          # a subagent's last reported cwd
        self.exited_unverified = False
        self._blocks = 0
        self._tool_calls_this_turn = 0
        self._cwd = ""   # session working directory, from hook inputs
        # the agent's home directory, for `~/x` in shell commands (the SDK
        # runs the agent on this machine, as this user)
        self._home = home if home is not None else os.path.expanduser("~")
        self._mutated_this_turn = False
        # a subagent (or unclassified tool) changed files this turn: even a
        # turn with no calls of the agent's own is gated
        self._changed_by_others = False

    # ------------------------------------------------------------ hooks

    async def post_tool_use(self, input_data, tool_use_id, context):
        """PostToolUse: classify the successful observation, update state."""
        try:
            return self._post_tool_use(input_data)
        except Exception:                                  # noqa: BLE001
            return self._fail_closed()

    async def post_tool_use_failure(self, input_data, tool_use_id, context):
        """PostToolUseFailure: no grounding credit, but a failed MUTATING
        call may still have had a partial effect — demand verification."""
        try:
            return self._post_tool_use_failure(input_data)
        except Exception:                                  # noqa: BLE001
            return self._fail_closed()

    def _fail_closed(self):
        """A hook that crashed may have missed a change: nothing this turn
        can verify it (the turn ends unverified rather than trusting it)."""
        self.state.note_unknown_change(
            {UNPLACED + "grounding-gate hit an error reading a tool call"})
        self._mutated_this_turn = True
        self._changed_by_others = True
        return {}

    def _post_tool_use(self, input_data):
        if not isinstance(input_data, dict) or \
                input_data.get("hook_event_name") != "PostToolUse":
            return {}
        tool = input_data.get("tool_name")
        tool = tool if isinstance(tool, str) else ""
        subagent = bool(input_data.get("agent_id"))
        if subagent and not self.gate_subagents:
            # a subagent's reads never reach the main agent; in strict mode
            # its changes still count, as unknown changes
            if self.strict_reads and not self._pure_read(tool):
                self._unknown_change(input_data)
            return {}
        prior_cwd = self._cwd
        cwd = self._event_cwd(input_data)
        if not subagent:
            self._note_cwd(input_data)   # a subagent's cwd isn't the agent's
        raw_input = input_data.get("tool_input", "")
        tool_input = _absolutize(raw_input, cwd)
        output = _output_text(input_data.get("tool_response", ""))
        # (mutation-epoch novelty — a fresh mutation re-opening the verifying
        # re-read — lives in classify_observation's hash tuple, not in the
        # text, so custom normalizers can't corrupt it)
        args = _serialize(tool_input)
        result = _serialize(input_data.get("tool_response", ""))

        self._tool_calls_this_turn += 1
        self.state.current_step += 1
        shell_read = self._is_shell_read(tool, tool_input)
        mcp = _mcp_class(tool)
        if self.strict_reads and not self._classified(tool, mcp):
            self._unknown_change(input_data, counted=True)
            return {}
        if mcp == "content":
            self.state.extractors.setdefault(tool, _content_identifiers)
        read_only = (tool in self.read_only_tools or shell_read
                     or mcp in ("content", "listing"))

        obs = classify_observation(tool, args, result, self.state, read_only)
        shown = None
        if shell_read:
            # only files whose content reached the agent can verify; files
            # it merely listed or counted cannot
            shown = shell.read_operands(tool_input["command"], output, cwd,
                                        self._home)[0]
            if not surface_hits(shown, self.state.claim_surface):
                obs["grounds_completion"] = False
        if (tool in self.listing_tools or tool in self.remote_tools
                or mcp == "listing"
                or _grep_lists_files(tool, tool_input,
                                     input_data.get("tool_response"), result)):
            # a listing shows the file exists, not what the change wrote
            obs["grounds_completion"] = False
        if self.strict_reads and (subagent or not self._verifying_read(
                tool, raw_input if mcp else tool_input, mcp,
                input_data.get("tool_response"))):
            # strict: only the agent's own whole, direct, local file read
            # verifies a change
            obs["grounds_completion"] = False
        qualifying = obs["grounds_assertion"] or obs["grounds_completion"]
        self.state.grounded_this_turn |= obs["grounds_assertion"]
        if obs["grounds_completion"] and self.state.cover_pending(
                tool, args, result, idents=shown):
            self.state.verified_this_turn = True
        if qualifying:
            self.state.budget = min(
                self.state.budget + self.state.refill, self.state.cap)
            self.state.halted = False
            # retain the qualifying observation for the verifier tier +
            # telemetry, and advance the monotonic verification marker — mirrors
            # turn_loop; classify_observation (Module 2) stays free of telemetry
            self.state.turn_observations.append(
                {"tool": tool, "args": args, "result": result,
                 "tier": "verified" if obs["grounds_completion"] else "observed"})
            if obs["grounds_completion"]:
                self.state.last_verification_step = self.state.current_step
        if (tool in self.mutating_tools and not shell_read) or mcp == "mutating":
            response = input_data.get("tool_response")
            # a timed-out or interrupted command stopped somewhere unknown
            interrupted = isinstance(response, dict) and response.get("interrupted") is True
            self._mutate(tool, raw_input, tool_input, output, interrupted, cwd,
                         prior_cwd, input_data)
        if self.strict_reads and tool in _BACKGROUND_TOOLS:
            self._check_background(tool, tool_input, input_data.get("tool_response"))
        return {}

    def _post_tool_use_failure(self, input_data):
        if not isinstance(input_data, dict) or \
                input_data.get("hook_event_name") != "PostToolUseFailure":
            return {}
        if input_data.get("agent_id") and not self.gate_subagents:
            if self.strict_reads:
                tool = input_data.get("tool_name")
                if not self._pure_read(tool if isinstance(tool, str) else ""):
                    self._unknown_change(input_data)
            return {}
        self._tool_calls_this_turn += 1
        self.state.current_step += 1
        prior_cwd = self._cwd
        cwd = self._event_cwd(input_data)
        if not input_data.get("agent_id"):
            self._note_cwd(input_data)
        tool = input_data.get("tool_name")
        tool = tool if isinstance(tool, str) else ""
        raw_input = input_data.get("tool_input", "")
        tool_input = _absolutize(raw_input, cwd)
        if ((tool in self.mutating_tools and not self._is_shell_read(tool, tool_input))
                or _mcp_class(tool) == "mutating"):
            self._mutate(tool, raw_input, tool_input, "", True, cwd, prior_cwd,
                         input_data)
        elif self.strict_reads and not self._classified(tool, _mcp_class(tool)):
            self._unknown_change(input_data, counted=True)
        return {}

    async def stop(self, input_data, tool_use_id, context):
        """Stop: the submit boundary. Block ungrounded finishes."""
        try:
            return self._stop(input_data)
        except Exception:                                  # noqa: BLE001
            # never trap the agent, never pass it silently: the typed exit
            self.exited_unverified = True
            return {"systemMessage": UNVERIFIED_BANNER}

    def _stop(self, input_data):
        if not isinstance(input_data, dict) or input_data.get("hook_event_name") != "Stop":
            return {}
        if self._tool_calls_this_turn == 0 and not self._changed_by_others \
                and not self._background:
            return {}   # tool-free turn: conversational, gate exempt

        # a turn that changed nothing makes assertions, whatever earlier
        # turns did
        claim_type = "completion" if self._mutated_this_turn else "assertion"
        # forward the optional verifier: a downgrade returns REJECT and flows
        # through the UNCHANGED block/budget/escape path below, so the agent
        # still reaches the UNVERIFIED valve — the tier adds strictness, never a trap
        verdict = boundary_check(
            {"claim_type": claim_type, "content": ""}, self.state, self._verifier)

        if verdict["verdict"] == ACCEPT:
            self._blocks = 0
            self.exited_unverified = False
            return {}

        self._blocks += 1
        self.state.budget = max(self.state.budget - 1, 0)   # rejected attempts burn rope
        if self._blocks > self.max_blocks or self.state.budget <= 0:
            # escape valve — the typed `unverified` exit. Never trap.
            self._blocks = 0
            self.exited_unverified = True
            self.state.halted = False   # exit clean, like ct=="unverified"
            if self.emit_progress:
                return {"systemMessage":
                        UNVERIFIED_BANNER + " " + _progress_line(self.progress())}
            return {"systemMessage": UNVERIFIED_BANNER}

        return {"decision": "block", "reason": self._reason(claim_type, verdict)}

    async def user_prompt_submit(self, input_data, tool_use_id, context):
        """UserPromptSubmit: a new turn — reset per-turn state and rope."""
        try:
            return self._user_prompt_submit(input_data)
        except Exception:                                  # noqa: BLE001
            return self._fail_closed()

    def _user_prompt_submit(self, input_data):
        if isinstance(input_data, dict):
            self._note_cwd(input_data)     # where the turn's first command starts
        self.state.grounded_this_turn = False
        self.state.verified_this_turn = False
        self.state.halted = False
        self.state.budget = self.state.cap   # fresh rope each turn
        # owed re-reads are per turn: the last turn's unverified edits were
        # already reported (exited_unverified); carrying them over would
        # block every later turn over work the user has moved past
        self.state.pending_verification = set()
        self.state.pending_aliases = {}
        self.state.changed_this_turn = set()
        for job in self._background:
            # still running: it may write during this turn too
            self.state.pending_verification.add(_job_entry(job))
        self._mutated_this_turn = bool(self._background)
        self._changed_by_others = False
        self.state.turn_observations = []    # per-turn; load-bearing (else a long
        #                                      session leaks retained observations).
        # last_verification_step is intentionally NOT reset — it is a monotonic
        # marker (like last_mutation_step) that telemetry reads across turns.
        self.exited_unverified = False
        self._blocks = 0
        self._tool_calls_this_turn = 0
        return {}

    # ------------------------------------------------------------ telemetry

    def progress(self):
        """Zero-token progress snapshot: ``state.progress()`` plus adapter-only
        counters. This is the RELIABLE programmatic surface (like
        ``exited_unverified``); ``emit_progress`` is only an opt-in best-effort
        ``systemMessage`` event. Makes no tool call and no model call.
        """
        p = self.state.progress()
        p.update({
            "blocks": self._blocks,
            "max_blocks": self.max_blocks,
            "tool_calls_this_turn": self._tool_calls_this_turn,
            "exited_unverified": self.exited_unverified,
            "strict_reads": self.strict_reads,
        })
        return p

    # ------------------------------------------------------------ wiring

    def as_options_hooks(self):
        """Build the ``hooks=`` dict for ``ClaudeAgentOptions``.

        Requires ``claude-agent-sdk``. (``Stop`` ignores matchers, and the
        gate must see every tool result, so no matcher patterns are used.)
        """
        try:
            from claude_agent_sdk import HookMatcher
        except ImportError as exc:                       # pragma: no cover
            raise ImportError(
                "as_options_hooks() needs the Claude Agent SDK: "
                "pip install claude-agent-sdk") from exc
        return {
            "PostToolUse": [HookMatcher(hooks=[self.post_tool_use])],
            "PostToolUseFailure": [
                HookMatcher(hooks=[self.post_tool_use_failure])],
            "Stop": [HookMatcher(hooks=[self.stop])],
            "UserPromptSubmit": [HookMatcher(hooks=[self.user_prompt_submit])],
        }

    # ------------------------------------------------------------ internals

    def _is_shell_read(self, tool, tool_input):
        """A mutating-class shell call (``{"command": ...}``) that only runs
        read-only programs, e.g. ``cat app.cfg``: it observes like a Read and
        records no mutation. Never in strict mode, where shell is opaque."""
        if self.strict_reads:
            return False
        command = tool_input.get("command") if isinstance(tool_input, dict) else None
        return (tool in self.mutating_tools and isinstance(command, str)
                and shell.is_read_only(command, self._home))

    def _pure_read(self, tool):
        mcp = _mcp_class(tool)
        return (tool in self.read_only_tools or tool in self.neutral_tools
                or mcp in ("content", "listing", "neutral"))

    def _classified(self, tool, mcp):
        """Known to the gate: a read, a neutral tool, or a mutating tool it
        can parse. Strict reads treats anything else as an unknown change."""
        return (tool in self.read_only_tools or tool in self.mutating_tools
                or tool in self.neutral_tools or mcp is not None)

    def _unknown_change(self, input_data, counted=False):
        """Strict reads, for a change the main agent's own tools didn't
        make visible. A subagent's known mutating call is parsed like the
        agent's own (its Bash commands too). An unclassified tool may have
        changed anything, files no argument names included, so it blocks
        verification for the turn until the tool is declared in
        ``read_only_tools``, ``neutral_tools`` or ``mutating_tools``."""
        if not counted:
            self.state.current_step += 1
            self._changed_by_others = True
        tool = input_data.get("tool_name")
        tool = tool if isinstance(tool, str) else ""
        cwd = self._event_cwd(input_data)      # the subagent's, not the agent's
        agent = str(input_data.get("agent_id") or "")
        prior = self._agent_cwds.get(agent, "") if agent else self._cwd
        if agent and isinstance(input_data.get("cwd"), str) \
                and input_data["cwd"].startswith("/"):
            self._agent_cwds[agent] = input_data["cwd"]
        raw_input = input_data.get("tool_input", "")
        tool_input = _absolutize(raw_input, cwd)
        response = input_data.get("tool_response")
        failed = input_data.get("hook_event_name") == "PostToolUseFailure" or (
            isinstance(response, dict) and response.get("interrupted") is True)
        if tool in self.mutating_tools or _mcp_class(tool) == "mutating":
            self._mutate(tool, raw_input, tool_input, _output_text(response or ""),
                         failed, cwd, prior, input_data)
            return
        targets = {t if t.startswith("/") else UNPLACED + t
                   for t in path_values(tool_input) if isinstance(tool_input, dict)}
        targets.add(UNPLACED + "unclassified tool %s may have changed files "
                    "(declare it in read_only_tools, neutral_tools or "
                    "mutating_tools)" % (tool or "?"))
        self.state.note_unknown_change(targets)
        self._mutated_this_turn = True

    def _verifying_read(self, tool, tool_input, mcp, response=None):
        """Strict reads: a direct read of one whole local file, by an
        absolute path. Partial reads (offset/limit, head/tail), multi-file
        reads, and read tools on MCP servers not trusted to be local don't
        verify."""
        if mcp == "content":
            if not self._trusted_server(tool) or tool.endswith("read_multiple_files"):
                return False
        elif tool not in self.content_tools:
            return False
        if not isinstance(tool_input, dict):
            return False
        if any(tool_input.get(k) not in (None, "", 0) for k in _PARTIAL_KEYS):
            return False
        # one file; an MCP read's path as given (the server resolves a
        # relative path against its own directory, not the session's)
        if _truncated(response):
            return False
        paths = path_values(tool_input)
        return len(paths) == 1 and all(p.startswith("/") for p in paths)

    def _mutate(self, tool, raw_input, tool_input, output, failed, cwd, prior,
                input_data):
        """Record a mutating call. In strict mode a shell command is placed
        from every directory it may have started in (the hook's ``cwd`` may
        be where it ENDED, after its own ``cd``), and an MCP tool's
        relative path stays unplaced (its server resolves it, not us)."""
        command = tool_input.get("command") if isinstance(tool_input, dict) else None
        if not self.strict_reads:
            self._record_mutation(tool_input, output, failed, cwd=cwd, tool=tool)
            return
        if isinstance(command, str):
            places = self._command_starts(command, cwd, prior)
            for place in places:
                if place is None:
                    self.state.note_unknown_change(
                        {UNPLACED + "a command that changed directory, from an "
                         "unknown start: " + command[:60]})
                    self._mutated_this_turn = True
                else:
                    self._record_mutation(tool_input, output, failed, cwd=place,
                                          tool=tool)
            if tool_input.get("run_in_background"):
                # it keeps running (and writing) after this call returns
                self._start_background(input_data, tool_input,
                                       next(iter(p for p in places if p), cwd))
            return
        if tool.startswith("mcp__"):
            self._record_mutation(raw_input, output, failed, cwd="", tool=tool)
            return
        self._record_mutation(tool_input, output, failed, cwd=cwd, tool=tool)

    def _command_starts(self, command, reported, prior):
        """Where a command may have started, given the hook's ``reported``
        cwd and the one tracked before it (``prior``); None: unknown."""
        if prior:
            if reported == prior or reported in shell.final_cwds(
                    command, prior, self._home):
                return [prior]
            return [prior, reported]
        if not reported:
            return [reported]
        # no earlier cwd: only a command that never changes directory is
        # placed with certainty (`cd /abs` ends there from ANY start)
        probe = "/nonexistent-start-of-grounding-gate"
        if shell.final_cwds(command, probe, self._home) == {probe}:
            return [reported]
        return [reported, None]

    def _trusted_server(self, tool):
        return _mcp_server(tool) in self.trusted_mcp_servers

    def _event_cwd(self, input_data):
        cwd = input_data.get("cwd")
        return cwd if isinstance(cwd, str) and cwd.startswith("/") else self._cwd

    def _start_background(self, input_data, tool_input, cwd):
        command = tool_input.get("command")
        effs = shell.effects(command, cwd, "", self._home, True) \
            if isinstance(command, str) else None
        if effs is not None and all(
                e[0] == "read" or e[0] == "opaque" and _trusted(
                    e[1], self.strict_trusted_programs) for e in effs):
            return            # writes nothing: nothing lands later
        response = input_data.get("tool_response")
        job = response.get("backgroundTaskId") if isinstance(response, dict) else None
        job = str(job) if job else "#%d" % (len(self._background) + 1)
        self._background[job] = (dict(tool_input, run_in_background=False), cwd)
        self.state.pending_verification.add(_job_entry(job))

    def _check_background(self, tool, tool_input, response):
        """A background command that finished (or was killed) wrote what it
        wrote by now: owe that as a change made at this point."""
        job = None
        if isinstance(tool_input, dict):
            for key in (_BACKGROUND_TOOLS[tool], "task_id", "bash_id", "shell_id"):
                if tool_input.get(key) is not None:
                    job = tool_input[key]
                    break
        job = str(job) if job is not None else None
        if job not in self._background:
            return
        if tool in _JOB_REPORTS and not _job_done(response):
            return
        command, cwd = self._background.pop(job)
        self.state.pending_verification.discard(_job_entry(job))
        self._record_mutation(command, cwd=cwd)

    def _note_cwd(self, input_data):
        """Track the session's working directory (every SDK hook input
        carries ``cwd``); relative paths resolve against it."""
        cwd = input_data.get("cwd")
        if isinstance(cwd, str) and cwd.startswith("/"):
            self._cwd = cwd

    def _grep_extract(self, args, result):
        return _grep_identifiers(args, result, self._cwd)

    def _shell_extract(self, args, result):
        return _shell_identifiers(args, result, self._cwd, self._home)

    def _record_mutation(self, tool_input, output="", failed=False, cwd=None,
                         tool=""):
        # a NEW mutation invalidates any earlier verification, its targets are
        # owed a re-read, and mutated identifiers join the claim surface so
        # only reads of THOSE count as verification
        self._mutated_this_turn = True
        if self.strict_reads and _mcp_class(tool) == "mutating" \
                and not self._trusted_server(tool):
            # another machine's (a container's) files: no local read pays
            self.state.note_unknown_change(
                {UNPLACED + "files changed by %s (its server isn't trusted as "
                 "local: trusted_mcp_servers)" % tool})
            return
        self.state.note_mutation(tool_input, self._cwd if cwd is None else cwd,
                                 output, failed, self._home,
                                 strict=self.strict_reads,
                                 trusted_programs=self.strict_trusted_programs)

    def _reason(self, claim_type, verdict=None):
        # a verifier DOWNGRADE is a structural ACCEPT the verify_with tier
        # overrode — the structural "missing" list would be EMPTY (the floor was
        # satisfied), so give a tier-specific, actionable reason instead of the
        # empty-parenthetical fallback (never an empty parenthetical).
        if verdict is not None and verdict.get("downgraded_by_verifier"):
            conf = verdict.get("confidence")
            band = ("confidence %.2f < threshold %s" % (conf, self.state.verify_threshold)
                    if conf is not None else "confidence < threshold")
            return (
                "grounding-gate REJECTED this finish (the verify_with tier could "
                "not confirm this claim (" + band + ")). Legal next moves: "
                "re-ground with a stronger observation and finish again, or state "
                "explicitly that your result is UNVERIFIED.")
        missing = []
        owed = sorted(self.state.pending_verification)
        if claim_type == "completion" and owed:
            missing.append(
                "changed but not re-read since: " + ", ".join(owed) +
                " (read each one after its last change)")
        if claim_type == "completion" and self.strict_reads \
                and not self.state.verified_this_turn:
            missing.append(
                "strict reads: only a direct file read (the Read tool) after "
                "your last change verifies it; shell output and search results "
                "don't count, and any Bash command counts as a new change")
        elif claim_type == "completion" and not self.state.verified_this_turn:
            missing.append(
                "no verified-tier observation: re-read what you modified "
                "(a fresh read of the changed files, AFTER the change)")
        if claim_type == "assertion" and not self.state.grounded_this_turn:
            missing.append(
                "no qualifying observation this turn: read the thing your "
                "answer makes claims about (a novel, relevant read)")
        if (claim_type == "assertion" and self.state.strict_g
                and not self.state.verified_this_turn):
            missing.append(
                "strict-G preset: even assertions need verified-tier "
                "grounding (an observation taken after your change)")
        unmet = [s for s in self.state.goal_predicates
                 if s not in self.state.verified_signals]
        if unmet:
            missing.append("declared signals not verified: " + ", ".join(unmet))
        if self.state.budget <= 0:
            missing.append("reasoning budget exhausted")
        if not missing:
            missing.append("the finish did not satisfy the gate's invariants")
        return (
            "grounding-gate REJECTED this finish (" + "; ".join(missing) +
            "). Legal next moves: make the qualifying observation and finish "
            "again, or state explicitly that your result is UNVERIFIED.")
