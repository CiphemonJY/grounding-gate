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
from grounding_gate.verifiers import StubVerifier

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


# ------------------------------------------- progress telemetry + verify_with

def test_progress_accessor():
    gate = GateHooks(claim_surface={"svc.cfg"})
    run(gate.post_tool_use(ptu("Read", {"file_path": "svc.cfg"}, "retries=5"), "t1", None))
    p = gate.progress()
    assert p["grounded_this_turn"] is True
    assert p["observations_this_turn"] >= 1
    # state keys AND adapter-only keys are both present
    for k in ("budget", "cap", "step", "rejection_count", "unmet_signals"):
        assert k in p
    for k in ("blocks", "max_blocks", "tool_calls_this_turn", "exited_unverified"):
        assert k in p


def test_emit_progress_appends_banner():
    gate = GateHooks(emit_progress=True, max_blocks=0)
    run(gate.post_tool_use(ptu("Write", {"file_path": "a.txt"}, "ok"), "t1", None))
    out = run(gate.stop(STOP, None, None))
    assert UNVERIFIED_BANNER in out["systemMessage"]
    assert "[progress" in out["systemMessage"]        # a compact summary is appended


def test_verifier_downgrades_grounded_finish():
    # a FLOOR-accepted finish (write then post-mutation read) that the verifier
    # rejects is blocked with a verify-tier reason...
    down = GateHooks(verifier=StubVerifier(0.0))
    run(down.post_tool_use(ptu("Write", {"file_path": "app.cfg",
                                         "content": "timeout=30"}, "ok"), "t1", None))
    run(down.post_tool_use(ptu("Read", {"file_path": "app.cfg"}, "timeout=30"), "t2", None))
    out = run(down.stop(STOP, None, None))
    assert out["decision"] == "block"
    assert "verify_with" in out["reason"]
    # ...while a confident verifier leaves the identical finish accepted
    ok = GateHooks(verifier=StubVerifier(1.0))
    run(ok.post_tool_use(ptu("Write", {"file_path": "app.cfg",
                                       "content": "timeout=30"}, "ok"), "t1", None))
    run(ok.post_tool_use(ptu("Read", {"file_path": "app.cfg"}, "timeout=30"), "t2", None))
    assert run(ok.stop(STOP, None, None)) == {}


def test_verifier_downgrade_reaches_escape_valve():
    # a downgrade feeds the UNCHANGED escape path, so the agent is never trapped
    gate = GateHooks(verifier=StubVerifier(0.0), max_blocks=0)
    run(gate.post_tool_use(ptu("Write", {"file_path": "app.cfg"}, "ok"), "t1", None))
    run(gate.post_tool_use(ptu("Read", {"file_path": "app.cfg"}, "v"), "t2", None))
    out = run(gate.stop(STOP, None, None))
    assert out == {"systemMessage": UNVERIFIED_BANNER}
    assert gate.exited_unverified is True


def test_turn_observations_reset():
    gate = GateHooks(claim_surface={"svc.cfg"})
    run(gate.post_tool_use(ptu("Read", {"file_path": "svc.cfg"}, "d"), "t1", None))
    assert gate.state.turn_observations                 # non-empty after a qualifying read
    run(gate.user_prompt_submit(PROMPT, None, None))
    assert gate.state.turn_observations == []           # emptied by the turn reset


# ------------------------------------------- tool classes, shell, coverage

def _edit_then(*events, surface=()):
    gate = GateHooks(claim_surface=set(surface))
    run(gate.post_tool_use(ptu("Edit", {"file_path": "/p/app.cfg"}, "ok"), "t0", None))
    for i, ev in enumerate(events):
        run(gate.post_tool_use(ev, "t%d" % (i + 1), None))
    return run(gate.stop(STOP, None, None))


def test_glob_listing_cannot_verify_an_edit():
    out = _edit_then(ptu("Glob", {"pattern": "*.cfg"}, "/p/app.cfg"))
    assert out["decision"] == "block"


def test_read_is_relevant_to_its_path_not_text_it_mentions():
    gate = GateHooks(claim_surface={"app.cfg"})
    run(gate.post_tool_use(ptu("Read", {"file_path": "/p/notes.txt"},
                               "TODO: bump app.cfg"), "t1", None))
    assert run(gate.stop(STOP, None, None))["decision"] == "block"


def test_shell_cat_verifies_but_ls_echo_and_grep_pattern_do_not():
    assert _edit_then(ptu("Bash", {"command": "cat app.cfg"}, "retries=5")) == {}
    for cmd, out in (("ls -l app.cfg", "-rw 12 app.cfg"),
                     ("echo updated app.cfg", "updated app.cfg"),
                     ("grep -n app.cfg notes.txt", "3: app.cfg")):
        assert _edit_then(ptu("Bash", {"command": cmd}, out))["decision"] == "block", cmd


def test_shell_read_records_no_mutation():
    gate = GateHooks(claim_surface={"svc.cfg"})
    run(gate.post_tool_use(ptu("Bash", {"command": "cat svc.cfg"}, "a=1"), "t1", None))
    assert gate.state.last_mutation_step == 0
    assert run(gate.stop(STOP, None, None)) == {}   # a plain grounded assertion


def test_every_edited_file_must_be_reread():
    gate = GateHooks()
    run(gate.post_tool_use(ptu("Edit", {"file_path": "/p/a.cfg"}, "ok"), "t1", None))
    run(gate.post_tool_use(ptu("Write", {"file_path": "/p/b.cfg"}, "ok"), "t2", None))
    run(gate.post_tool_use(ptu("Read", {"file_path": "/p/a.cfg"}, "a=1"), "t3", None))
    assert run(gate.stop(STOP, None, None))["decision"] == "block"
    assert gate.progress()["pending_verification"] == ["/p/b.cfg"]
    run(gate.post_tool_use(ptu("Read", {"file_path": "b.cfg"}, "b=2"), "t4", None))
    assert run(gate.stop(STOP, None, None)) == {}


def test_shell_sed_targets_are_owed_a_reread():
    gate = GateHooks()
    run(gate.post_tool_use(ptu("Bash", {"command": "sed -i s/1/2/ a.cfg b.cfg"}, ""),
                           "t1", None))
    run(gate.post_tool_use(ptu("Read", {"file_path": "/p/a.cfg"}, "a=2"), "t2", None))
    assert run(gate.stop(STOP, None, None))["decision"] == "block"


def test_symbols_ground_through_read_and_shell_cat():
    for event in (ptu("Read", {"file_path": "/p/src/cfg.py"}, "def parse_config(p):"),
                  ptu("Read", {"file_path": "/p/src/app.py"}, "x = cfg.parse_config()"),
                  ptu("Bash", {"command": "cat src/cfg.py"}, "def parse_config(p):")):
        gate = GateHooks(claim_surface={"parse_config"})
        run(gate.post_tool_use(event, "t1", None))
        assert run(gate.stop(STOP, None, None)) == {}, event


def test_echo_output_grounds_no_symbol():
    gate = GateHooks(claim_surface={"parse_config"})
    run(gate.post_tool_use(ptu("Bash", {"command": "echo parse_config"},
                               "parse_config"), "t1", None))
    assert run(gate.stop(STOP, None, None))["decision"] == "block"


def test_deleted_scratch_file_is_not_owed():
    for rm in ("rm tmp1.txt", "rm -f tmp*.txt", "rm ./tmp1.txt"):
        gate = GateHooks()
        run(gate.post_tool_use(ptu("Write", {"file_path": "/p/tmp1.txt"}, "ok"), "t1", None))
        run(gate.post_tool_use(ptu("Bash", {"command": rm}, ""), "t2", None))
        assert not gate.state.pending_verification, rm
        run(gate.post_tool_use(ptu("Edit", {"file_path": "/p/app.cfg"}, "ok"), "t3", None))
        run(gate.post_tool_use(ptu("Read", {"file_path": "/p/app.cfg"}, "a=1"), "t4", None))
        assert run(gate.stop(STOP, None, None)) == {}, rm


def test_grep_file_list_cannot_verify_but_content_mode_can():
    files = ptu("Grep", {"pattern": "x", "path": "/p"}, "/p/app.cfg")
    assert _edit_then(files)["decision"] == "block"
    lines = ptu("Grep", {"pattern": "x", "path": "/p", "output_mode": "content"},
                "/p/app.cfg:3:x=1")
    assert _edit_then(lines) == {}


def test_mcp_filesystem_tools_are_classified():
    read = ptu("mcp__fs__read_text_file", {"path": "/p/app.cfg"}, "x=1")
    assert _edit_then(read) == {}
    listing = ptu("mcp__fs__list_directory", {"path": "/p"}, "[FILE] app.cfg")
    assert _edit_then(listing)["decision"] == "block"
    gate = GateHooks()
    run(gate.post_tool_use(ptu("mcp__fs__write_file", {"path": "/p/a.cfg"}, "ok"),
                           "t1", None))
    assert gate.state.pending_verification == {"/p/a.cfg"}


def test_relative_paths_resolve_against_session_cwd():
    def event(tool, tool_input, out="ok"):
        return dict(ptu(tool, tool_input, out), cwd="/srv/proj")
    gate = GateHooks()
    run(gate.post_tool_use(event("Edit", {"file_path": "/srv/proj/conf/app.cfg"}),
                           "t1", None))
    # from /srv/proj, a bare app.cfg is /srv/proj/app.cfg, not conf/app.cfg
    run(gate.post_tool_use(event("Bash", {"command": "cat app.cfg"}, "x=1"), "t2", None))
    assert run(gate.stop(STOP, None, None))["decision"] == "block"
    run(gate.post_tool_use(event("Read", {"file_path": "conf/app.cfg"}, "x=2"),
                           "t3", None))
    assert run(gate.stop(STOP, None, None)) == {}


def _gate_after(*events):
    gate = GateHooks()
    for i, ev in enumerate(events):
        if ev is PROMPT:
            run(gate.user_prompt_submit(PROMPT, None, None))
        else:
            run(gate.post_tool_use(ev, "t%d" % i, None))
    return gate, run(gate.stop(STOP, None, None))


def test_moved_file_is_owed_at_its_new_path():
    edit = ptu("Edit", {"file_path": "/p/a.cfg"}, "ok")
    for move in (ptu("Bash", {"command": "mv /p/a.cfg /p/b.cfg"}, ""),
                 ptu("Bash", {"command": "git mv /p/a.cfg /p/b.cfg"}, ""),
                 ptu("mcp__fs__move_file",
                     {"source": "/p/a.cfg", "destination": "/p/b.cfg"}, "ok")):
        gate, out = _gate_after(edit, move)
        assert out["decision"] == "block" and "/p/b.cfg" in out["reason"], move
        gate, out = _gate_after(edit, move,
                                ptu("Read", {"file_path": "/p/b.cfg"}, "x=1"))
        assert out == {}, move


def test_removed_directory_clears_what_it_owed():
    gate, out = _gate_after(
        ptu("Edit", {"file_path": "/p/build/x.cfg"}, "ok"),
        ptu("Bash", {"command": "rm -rf /p/build"}, ""),
        ptu("Edit", {"file_path": "/p/app.cfg"}, "ok"),
        ptu("Read", {"file_path": "/p/app.cfg"}, "a=1"))
    assert out == {}


def test_owed_rereads_reset_each_turn():
    gate, out = _gate_after(
        ptu("Edit", {"file_path": "/p/a.cfg"}, "ok"), PROMPT,
        ptu("Edit", {"file_path": "/p/b.cfg"}, "ok"),
        ptu("Read", {"file_path": "/p/b.cfg"}, "b=1"))
    assert out == {}
    gate, out = _gate_after(
        ptu("Edit", {"file_path": "/p/a.cfg"}, "ok"),
        ptu("Read", {"file_path": "/p/a.cfg"}, "a=1"), PROMPT,
        ptu("Edit", {"file_path": "/p/b.cfg"}, "ok"))
    assert out["decision"] == "block"


def test_block_reason_names_the_files_still_owed():
    gate, out = _gate_after(ptu("Edit", {"file_path": "/p/a.cfg"}, "ok"),
                            ptu("Edit", {"file_path": "/p/b.cfg"}, "ok"),
                            ptu("Read", {"file_path": "/p/a.cfg"}, "a=1"))
    assert "/p/b.cfg" in out["reason"] and "/p/a.cfg" not in out["reason"]


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
