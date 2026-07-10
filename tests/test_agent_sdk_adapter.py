"""Claude Agent SDK adapter suite — pure hook-dict simulation, no SDK needed.

Input shapes mirror the documented SDK hook contract (PostToolUse carries
tool_name/tool_input/tool_response; Stop returns {"decision": "block",
"reason": ...} to force continuation). Runs under pytest, or bare:
``python tests/test_agent_sdk_adapter.py``.
"""

import asyncio
import sys
from pathlib import Path

try:
    import grounding_gate  # noqa: F401
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grounding_gate.adapters.claude_agent_sdk import UNVERIFIED_BANNER, GateHooks

run = asyncio.run


def ptu(tool, tool_input, response):
    return {"hook_event_name": "PostToolUse", "tool_name": tool,
            "tool_input": tool_input, "tool_response": response}


STOP = {"hook_event_name": "Stop", "stop_hook_active": False}
PROMPT = {"hook_event_name": "UserPromptSubmit", "prompt": "next task"}


def test_completion_blocked_without_post_mutation_read():
    gate = GateHooks()
    run(gate.post_tool_use(ptu("Write", {"file_path": "app.cfg",
                                         "content": "timeout=30"}, "ok"), "t1", None))
    out = run(gate.stop(STOP, None, None))
    assert out["decision"] == "block"
    assert "re-read" in out["reason"]


def test_completion_allowed_after_post_mutation_read():
    gate = GateHooks()
    run(gate.post_tool_use(ptu("Write", {"file_path": "app.cfg",
                                         "content": "timeout=30"}, "ok"), "t1", None))
    run(gate.post_tool_use(ptu("Read", {"file_path": "app.cfg"},
                               "timeout=30"), "t2", None))
    assert run(gate.stop(STOP, None, None)) == {}


def test_mutated_identifiers_join_claim_surface_automatically():
    gate = GateHooks()
    run(gate.post_tool_use(ptu("Write", {"file_path": "app.cfg"}, "ok"), "t1", None))
    assert "app.cfg" in gate.state.claim_surface


def test_grounded_assertion_allowed():
    gate = GateHooks(claim_surface={"svc.cfg"})
    run(gate.post_tool_use(ptu("Read", {"file_path": "svc.cfg"},
                               "retries=5"), "t1", None))
    assert run(gate.stop(STOP, None, None)) == {}


def test_irrelevant_read_does_not_license_assertion():
    gate = GateHooks(claim_surface={"svc.cfg"})
    run(gate.post_tool_use(ptu("Read", {"file_path": "other.txt"},
                               "hello"), "t1", None))
    out = run(gate.stop(STOP, None, None))
    assert out["decision"] == "block"


def test_unknown_tool_neither_grounds_nor_mutates():
    gate = GateHooks(claim_surface={"svc.cfg"})
    run(gate.post_tool_use(
        ptu("mcp__db__query", {"q": "select svc.cfg"}, "svc.cfg: retries=5"),
        "t1", None))
    assert not gate.state.grounded_this_turn
    assert gate.state.last_mutation_step == 0


def test_conversational_turn_exempt():
    gate = GateHooks()
    assert run(gate.stop(STOP, None, None)) == {}


def test_escape_valve_allows_unverified_exit():
    gate = GateHooks(max_blocks=2)
    run(gate.post_tool_use(ptu("Write", {"file_path": "a.txt"}, "ok"), "t1", None))
    first = run(gate.stop(STOP, None, None))
    second = run(gate.stop(STOP, None, None))
    third = run(gate.stop(STOP, None, None))
    assert first["decision"] == "block" and second["decision"] == "block"
    assert third == {"systemMessage": UNVERIFIED_BANNER}


def test_user_prompt_submit_resets_turn_latches():
    gate = GateHooks(claim_surface={"svc.cfg"})
    run(gate.post_tool_use(ptu("Read", {"file_path": "svc.cfg"}, "d"), "t1", None))
    assert gate.state.grounded_this_turn
    run(gate.user_prompt_submit(PROMPT, None, None))
    assert not gate.state.grounded_this_turn
    # a non-qualifying call in the NEW turn: stale grounding must not license it
    run(gate.post_tool_use(ptu("Read", {"file_path": "other.txt"}, "x"), "t2", None))
    out = run(gate.stop(STOP, None, None))
    assert out["decision"] == "block"   # stale grounding does not carry over


def test_repeated_identical_read_is_novelty_defeated():
    gate = GateHooks(claim_surface={"svc.cfg"})
    run(gate.post_tool_use(ptu("Read", {"file_path": "svc.cfg"}, "d"), "t1", None))
    run(gate.user_prompt_submit(PROMPT, None, None))
    # identical re-read in the new turn: hash already recorded, no credit
    run(gate.post_tool_use(ptu("Read", {"file_path": "svc.cfg"}, "d"), "t2", None))
    assert not gate.state.grounded_this_turn


def test_wrong_event_names_are_ignored():
    gate = GateHooks()
    assert run(gate.post_tool_use(STOP, None, None)) == {}
    assert run(gate.stop(ptu("Read", {}, ""), None, None)) == {}


def test_unrelated_read_after_write_is_NOT_verification():
    # the JSON-key-pollution attack: schema keys must never join the surface
    gate = GateHooks()
    run(gate.post_tool_use(ptu("Write", {"file_path": "app.cfg",
                                         "content": "timeout=30"}, "ok"), "t1", None))
    assert "file_path" not in gate.state.claim_surface
    assert "content" not in gate.state.claim_surface
    run(gate.post_tool_use(ptu("Read", {"file_path": "other.txt"},
                               "totally unrelated junk"), "t2", None))
    out = run(gate.stop(STOP, None, None))
    assert out["decision"] == "block"


def test_idempotent_write_can_still_be_verified():
    # read -> identical write -> identical re-read must ground (epoch novelty)
    gate = GateHooks(claim_surface={"app.cfg"})
    run(gate.post_tool_use(ptu("Read", {"file_path": "app.cfg"},
                               "timeout=30"), "t1", None))
    run(gate.post_tool_use(ptu("Write", {"file_path": "app.cfg",
                                         "content": "timeout=30"}, "ok"), "t2", None))
    run(gate.post_tool_use(ptu("Read", {"file_path": "app.cfg"},
                               "timeout=30"), "t3", None))
    assert run(gate.stop(STOP, None, None)) == {}


def test_later_mutation_invalidates_verification():
    gate = GateHooks()
    run(gate.post_tool_use(ptu("Write", {"file_path": "a.cfg"}, "ok"), "t1", None))
    run(gate.post_tool_use(ptu("Read", {"file_path": "a.cfg"}, "v1"), "t2", None))
    run(gate.post_tool_use(ptu("Write", {"file_path": "b.cfg"}, "ok"), "t3", None))
    out = run(gate.stop(STOP, None, None))
    assert out["decision"] == "block"   # b.cfg never re-read


def test_second_conversational_turn_is_exempt():
    gate = GateHooks()
    run(gate.post_tool_use(ptu("Write", {"file_path": "a.cfg"}, "ok"), "t1", None))
    run(gate.post_tool_use(ptu("Read", {"file_path": "a.cfg"}, "v"), "t2", None))
    assert run(gate.stop(STOP, None, None)) == {}
    run(gate.user_prompt_submit(PROMPT, None, None))
    assert run(gate.stop(STOP, None, None)) == {}   # zero tools this turn


def test_failed_mutating_call_demands_verification():
    gate = GateHooks()
    run(gate.post_tool_use_failure(
        {"hook_event_name": "PostToolUseFailure", "tool_name": "Write",
         "tool_input": {"file_path": "a.cfg"}}, "t1", None))
    out = run(gate.stop(STOP, None, None))
    assert out["decision"] == "block"               # partial effect possible
    run(gate.post_tool_use(ptu("Read", {"file_path": "a.cfg"}, "v"), "t2", None))
    assert run(gate.stop(STOP, None, None)) == {}


def test_subagent_events_ignored_by_default():
    gate = GateHooks(claim_surface={"a.cfg"})
    sub = ptu("Read", {"file_path": "a.cfg"}, "v")
    sub["agent_id"] = "agent_123"
    run(gate.post_tool_use(sub, "t1", None))
    assert not gate.state.grounded_this_turn
    assert gate.state.current_step == 0


def test_escape_valve_sets_flag_and_clean_state():
    gate = GateHooks(max_blocks=0)
    run(gate.post_tool_use(ptu("Write", {"file_path": "a.cfg"}, "ok"), "t1", None))
    out = run(gate.stop(STOP, None, None))
    assert out == {"systemMessage": UNVERIFIED_BANNER}
    assert gate.exited_unverified is True
    assert gate.state.halted is False


def test_new_turn_restores_budget():
    gate = GateHooks(max_blocks=99)
    run(gate.post_tool_use(ptu("Write", {"file_path": "a.cfg"}, "ok"), "t1", None))
    for _ in range(3):
        run(gate.stop(STOP, None, None))            # burn rope
    assert gate.state.budget < gate.state.cap
    run(gate.user_prompt_submit(PROMPT, None, None))
    assert gate.state.budget == gate.state.cap
    assert gate.exited_unverified is False


def test_normalizer_passthrough_defeats_noisy_reread():
    import re as _re
    scrub = {"Read": lambda t: _re.sub(r"@up \d+s@", "@up X@", t)}
    noisy1 = "retries=5 @up 4321s@"
    noisy2 = "retries=5 @up 9876s@"   # same content, uptime counter moved
    # without the scrubber the stale re-read wrongly re-grounds
    bare = GateHooks(claim_surface={"svc.cfg"})
    run(bare.post_tool_use(ptu("Read", {"file_path": "svc.cfg"}, noisy1), "t1", None))
    run(bare.user_prompt_submit(PROMPT, None, None))
    run(bare.post_tool_use(ptu("Read", {"file_path": "svc.cfg"}, noisy2), "t2", None))
    assert bare.state.grounded_this_turn        # the hole James pointed at
    # with the scrubber, novelty correctly defeats it
    gated = GateHooks(claim_surface={"svc.cfg"}, normalizers=scrub)
    run(gated.post_tool_use(ptu("Read", {"file_path": "svc.cfg"}, noisy1), "t1", None))
    run(gated.user_prompt_submit(PROMPT, None, None))
    run(gated.post_tool_use(ptu("Read", {"file_path": "svc.cfg"}, noisy2), "t2", None))
    assert not gated.state.grounded_this_turn


def test_digit_stripping_normalizer_cannot_break_epoch_novelty():
    # the mutation epoch lives in the hash tuple, not the text — a scrubber
    # that erases every digit still can't stop the post-write re-read of
    # textually-identical content from verifying
    import re as _re
    gate = GateHooks(normalizers={"Read": lambda t: _re.sub(r"\d", "", t)})
    run(gate.post_tool_use(ptu("Read", {"file_path": "a.cfg"}, "v=1"), "t1", None))
    run(gate.post_tool_use(ptu("Write", {"file_path": "a.cfg",
                                         "content": "v=1"}, "ok"), "t2", None))
    run(gate.post_tool_use(ptu("Read", {"file_path": "a.cfg"}, "v=1"), "t3", None))
    assert run(gate.stop(STOP, None, None)) == {}


def test_extractor_passthrough_bridges_lexical_domains():
    gate = GateHooks(
        claim_surface={"app.cfg"},
        read_only_tools=set(GateHooks().read_only_tools) | {"StatFile"},
        extractors={"StatFile": lambda a, r: {"app.cfg"}})
    run(gate.post_tool_use(ptu("StatFile", {"handle": "h7"}, "ino:8812732"),
                           "t1", None))
    assert run(gate.stop(STOP, None, None)) == {}


def test_strict_g_reason_is_actionable():
    gate = GateHooks(claim_surface={"svc.cfg"}, model_class="skipper")
    run(gate.post_tool_use(ptu("Read", {"file_path": "svc.cfg"}, "d"), "t1", None))
    out = run(gate.stop(STOP, None, None))
    assert out["decision"] == "block"
    assert "strict-G" in out["reason"]              # never an empty parenthetical


# ------------------------------------------------------- bare-python runner

if __name__ == "__main__":
    import inspect
    failures = []
    cases = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)
             and not inspect.signature(f).parameters]
    for name, fn in cases:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError:
            print(f"  FAIL  {name}")
            failures.append(name)
    print(f"\n{'ALL PASS' if not failures else f'FAILED: {failures}'}"
          f" — {len(cases) - len(failures)}/{len(cases)}")
    sys.exit(1 if failures else 0)
