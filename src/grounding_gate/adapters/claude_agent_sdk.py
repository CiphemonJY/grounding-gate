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

from ..boundary import ACCEPT, boundary_check
from ..classifier import classify_observation
from ..state import GateState, extract_identifiers

# Built-in SDK tools by consequence class. Unknown tools (including MCP tools)
# are treated as NEITHER: they earn no grounding credit and record no
# mutation — maximally conservative in both directions. Override per-agent.
DEFAULT_READ_ONLY_TOOLS = frozenset(
    {"Read", "Glob", "Grep", "WebFetch", "WebSearch", "NotebookRead"})
# Bash is classed as mutating because it CAN mutate; the cost is that a
# harmless bash call also opens the verified tier for later reads. Narrow
# this set if your agent's bash usage is read-only.
DEFAULT_MUTATING_TOOLS = frozenset(
    {"Write", "Edit", "MultiEdit", "NotebookEdit", "Bash"})

# tool_input keys whose VALUES name what was touched
_PATH_KEYS = ("file_path", "path", "notebook_path", "filename", "file")

UNVERIFIED_BANNER = (
    "grounding-gate: exiting UNVERIFIED — the agent finished without a "
    "qualifying observation backing its claims.")


def _serialize(value):
    """Deterministic text for hashing/identifier extraction."""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(value)


def _mutation_identifiers(tool_input):
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
            out |= _mutation_identifiers(v)
        return out
    if isinstance(tool_input, (list, tuple)):
        out = set()
        for v in tool_input:
            out |= _mutation_identifiers(v)
        return out
    if tool_input is None:
        return set()
    return extract_identifiers(str(tool_input), "")


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

    Attributes:
        exited_unverified: True when the LAST allowed stop went through the
            escape valve rather than a verified/grounded finish. Check this
            programmatically in headless runs; it resets on the next turn.
    """

    def __init__(self, claim_surface=(), model_class="default",
                 read_only_tools=DEFAULT_READ_ONLY_TOOLS,
                 mutating_tools=DEFAULT_MUTATING_TOOLS,
                 max_blocks=3, gate_subagents=False,
                 normalizers=None, extractors=None):
        self.state = GateState.for_model_class(
            model_class, claim_surface=set(claim_surface),
            normalizers=dict(normalizers or {}),
            extractors=dict(extractors or {}))
        self.read_only_tools = set(read_only_tools)
        self.mutating_tools = set(mutating_tools)
        self.max_blocks = max_blocks
        self.gate_subagents = gate_subagents
        self.exited_unverified = False
        self._blocks = 0
        self._tool_calls_this_turn = 0

    # ------------------------------------------------------------ hooks

    async def post_tool_use(self, input_data, tool_use_id, context):
        """PostToolUse: classify the successful observation, update state."""
        if input_data.get("hook_event_name") != "PostToolUse":
            return {}
        if input_data.get("agent_id") and not self.gate_subagents:
            return {}
        tool = input_data.get("tool_name", "")
        tool_input = input_data.get("tool_input", "")
        # (mutation-epoch novelty — a fresh mutation re-opening the verifying
        # re-read — lives in classify_observation's hash tuple, not in the
        # text, so custom normalizers can't corrupt it)
        args = _serialize(tool_input)
        result = _serialize(input_data.get("tool_response", ""))

        self._tool_calls_this_turn += 1
        self.state.current_step += 1
        read_only = tool in self.read_only_tools

        obs = classify_observation(tool, args, result, self.state, read_only)
        qualifying = obs["grounds_assertion"] or obs["grounds_completion"]
        self.state.grounded_this_turn |= obs["grounds_assertion"]
        self.state.verified_this_turn |= obs["grounds_completion"]
        if qualifying:
            self.state.budget = min(
                self.state.budget + self.state.refill, self.state.cap)
            self.state.halted = False
        if tool in self.mutating_tools:
            self._record_mutation(tool_input)
        return {}

    async def post_tool_use_failure(self, input_data, tool_use_id, context):
        """PostToolUseFailure: no grounding credit, but a failed MUTATING
        call may still have had a partial effect — demand verification."""
        if input_data.get("hook_event_name") != "PostToolUseFailure":
            return {}
        if input_data.get("agent_id") and not self.gate_subagents:
            return {}
        self._tool_calls_this_turn += 1
        self.state.current_step += 1
        if input_data.get("tool_name", "") in self.mutating_tools:
            self._record_mutation(input_data.get("tool_input", ""))
        return {}

    async def stop(self, input_data, tool_use_id, context):
        """Stop: the submit boundary. Block ungrounded finishes."""
        if input_data.get("hook_event_name") != "Stop":
            return {}
        if self._tool_calls_this_turn == 0:
            return {}   # tool-free turn: conversational, gate exempt

        claim_type = ("completion" if self.state.last_mutation_step > 0
                      else "assertion")
        verdict = boundary_check(
            {"claim_type": claim_type, "content": ""}, self.state)

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
            return {"systemMessage": UNVERIFIED_BANNER}

        return {"decision": "block", "reason": self._reason(claim_type)}

    async def user_prompt_submit(self, input_data, tool_use_id, context):
        """UserPromptSubmit: a new turn — reset per-turn state and rope."""
        self.state.grounded_this_turn = False
        self.state.verified_this_turn = False
        self.state.halted = False
        self.state.budget = self.state.cap   # fresh rope each turn
        self.exited_unverified = False
        self._blocks = 0
        self._tool_calls_this_turn = 0
        return {}

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

    def _record_mutation(self, tool_input):
        self.state.last_mutation_step = self.state.current_step
        # a NEW mutation invalidates any earlier verification: the verifying
        # observation must postdate the LAST mutation
        self.state.verified_this_turn = False
        # you must verify what you changed: mutated identifiers join the
        # claim surface so only reads of THOSE count as verification
        self.state.claim_surface |= _mutation_identifiers(tool_input)

    def _reason(self, claim_type):
        missing = []
        if claim_type == "completion" and not self.state.verified_this_turn:
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
