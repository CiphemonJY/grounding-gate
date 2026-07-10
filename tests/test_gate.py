"""Full acceptance suite: the 17 reference cases ported 1:1, extended with
cases pinning every adversarial-review finding since (the runner prints the
live count).

Expected outcomes were authored by the reviewer, never by the model that
drafted the module under test (see README, "How this was built").

Runs under pytest, or with zero dependencies: ``python tests/test_gate.py``.
"""

import sys
from pathlib import Path

try:
    import grounding_gate  # noqa: F401  (installed)
except ImportError:        # zero-install fallback: run from a raw checkout
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grounding_gate import (
    ACCEPT,
    REJECT,
    GateState,
    boundary_check,
    classify_observation,
    turn_loop,
)


def S(**kw):
    return GateState.for_model_class("default", claim_surface={"file.txt"}, **kw)


# ---------------------------------------------------------- Module 2: classifier

def test_t1_read_no_prior_mutation_grounds_assertion_only():
    st = S()
    st.current_step = 1
    assert classify_observation("read", "file.txt", "data", st, True) == {
        "grounds_assertion": True, "grounds_completion": False}


def test_t2_repeated_identical_read_novelty_defeat():
    st = S()
    st.current_step = 1
    classify_observation("read", "file.txt", "data", st, True)   # records hash
    assert classify_observation("read", "file.txt", "data", st, True) == {
        "grounds_assertion": False, "grounds_completion": False}


def test_t3_unrelated_file_relevance_defeat():
    st = S()
    st.current_step = 1
    assert classify_observation("read", "other.txt", "stuff", st, True) == {
        "grounds_assertion": False, "grounds_completion": False}


def test_t4_mutating_call_never_self_grounds():
    st = S()
    st.current_step = 1
    assert classify_observation("write", "file.txt", "ok", st, False) == {
        "grounds_assertion": False, "grounds_completion": False}


def test_t5_read_predating_mutation_asserts_only():
    st = S()
    st.last_mutation_step = 5
    st.current_step = 3
    assert classify_observation("read", "file.txt", "data", st, True) == {
        "grounds_assertion": True, "grounds_completion": False}


def test_t6_post_mutation_read_verifies():
    st = S()
    st.last_mutation_step = 4
    st.current_step = 5
    assert classify_observation("read", "file.txt", "new_data", st, True) == {
        "grounds_assertion": True, "grounds_completion": True}


def test_t7_syslog_stamped_stale_reread_is_novelty_defeated():
    # identical content, only a syslog timestamp differs -> no re-grounding
    st = S()
    st.current_step = 1
    first = classify_observation(
        "read", "file.txt", "file.txt Jul 10 20:47:03 retries=5", st, True)
    assert first["grounds_assertion"]
    st.current_step = 2
    stale = classify_observation(
        "read", "file.txt", "file.txt Jul 10 20:47:59 retries=5", st, True)
    assert not stale["grounds_assertion"]


def test_t8_normalize_covers_common_timestamp_formats():
    from grounding_gate import normalize
    pairs = [
        ("done 10 Jul 2026 ok", "done 11 Jul 2026 ok"),          # RFC822 date
        ("Thu, 10 Jul 2026 20:47:03", "Fri, 11 Jul 2026 01:02:03"),  # RFC1123 + DOW
        ("at 20:47:03 ready", "at 21:03:59 ready"),              # clock time
        ("built 2026/07/10 fine", "built 2026/07/11 fine"),      # bare date
        ("due 07/10/2026 x", "due 07/11/2026 x"),                # US date
        ("seen 5 min ago", "seen 12 min ago"),                   # relative
        ("seen 2 weeks ago", "seen 3 months ago"),               # long relative
        ("-rw- 512 Jul 10 20:47 app.cfg", "-rw- 512 Jul 10 21:03 app.cfg"),  # ls -l
        ("-rw- 512 Jul 10  2025 app.cfg", "-rw- 512 Aug  3  2024 app.cfg"),  # ls -l old
        ("run at 8:47 PM done", "run at 9:03 AM done"),          # 12h across noon
        ("2026-07-10 20:47 saved", "2026-07-10 21:03 saved"),    # minute ISO
        ("id DEADBEEF01 ok", "id CAFEBABE99 ok"),                # uppercase hex
        ("req 550e8400-e29b-41d4-a716-446655440000",
         "req 6ba7b810-9dad-11d1-80b4-00c04fd430c8"),            # dashed UUID
        ("<Foo object at 0x7f3a2b4c5d60>",
         "<Foo object at 0x7f9e1c8d2a10>"),                      # repr address
    ]
    for a, b in pairs:
        assert normalize(a) == normalize(b), (a, b)
    # bounded ISO tail: payload glued to a timestamp must SURVIVE — genuinely
    # different values may not collapse (wrong-rejection guard)
    assert (normalize("2026-07-10T20:47:03,value=5")
            != normalize("2026-07-10T20:47:03,value=7"))


def test_t14_meaningful_content_survives_normalization():
    # must-NOT-collapse pairs: real value changes and documented residuals
    from grounding_gate import normalize
    pairs = [
        # month rules stop at line breaks: end-of-line counters are content
        ("Total errors: 3\nSep 2026 report", "Total errors: 4\nSep 2026 report"),
        # US-date year anchor: block sizes are not dates
        ("chunk 10/12/1024 ok", "chunk 10/12/2048 ok"),
        # small 0x constants are values, not addresses
        ("flags=0xFF set", "flags=0x1F set"),
        # documented residual: bare 24h HH:MM is ambiguous with scores/ratios
        # and deliberately survives (per-tool normalizers are the fix)
        ("checked at 20:47", "checked at 20:52"),
    ]
    for a, b in pairs:
        assert normalize(a) != normalize(b), (a, b)


def test_t9_per_tool_normalizer_composes_with_default():
    # custom scrubber handles a format the default can't; the default ISO
    # floor still applies after it
    import re as _re
    st = S(normalizers={"read": lambda t: _re.sub(r"@up \d+s@", "@up X@", t)})
    st.current_step = 1
    first = classify_observation(
        "read", "file.txt", "v=1 @up 4321s@ 2026-07-10T20:47:03", st, True)
    assert first["grounds_assertion"]
    st.current_step = 2
    stale = classify_observation(
        "read", "file.txt", "v=1 @up 9876s@ 2026-07-10T21:00:00", st, True)
    assert not stale["grounds_assertion"]


def test_t10_per_tool_extractor_bridges_lexical_domains():
    # tool output (an inode) shares no tokens with the claim surface: the
    # default extractor can never intersect; a registered one can
    st = S()
    st.current_step = 1
    assert not classify_observation(
        "statfile", "app-handle-7", "ino:8812732", st, True)["grounds_assertion"]
    st2 = S(extractors={"statfile": lambda a, r: {"file.txt"}})
    st2.current_step = 1
    assert classify_observation(
        "statfile", "app-handle-7", "ino:8812732", st2, True)["grounds_assertion"]


# ---------------------------------------------------------- Module 4: boundary

def B(ct, **kw):
    st = S()
    for k, v in kw.items():
        setattr(st, k, v)
    return boundary_check({"claim_type": ct, "content": "x"}, st)["verdict"]


def test_r1_ungrounded_assertion_rejected():
    assert B("assertion", budget=3) == REJECT


def test_r2_completion_without_verified_rejected():
    assert B("completion", budget=3, grounded_this_turn=True) == REJECT


def test_r3_grounded_but_budget_zero_rejected():
    assert B("assertion", budget=0, grounded_this_turn=True) == REJECT


def test_r5_declared_signal_missing_rejected():
    assert B("completion", budget=3, verified_this_turn=True,
             goal_predicates=["tests_passed"]) == REJECT


def test_a1_unverified_always_legal():
    assert B("unverified", budget=0) == ACCEPT


def test_a2_grounded_assertion_accepted():
    assert B("assertion", budget=2, grounded_this_turn=True) == ACCEPT


def test_a3_verified_completion_with_signals_accepted():
    st = S()
    st.budget = 1
    st.verified_this_turn = True
    st.goal_predicates = ["tests_passed"]
    st.verified_signals = {"tests_passed"}
    assert boundary_check(
        {"claim_type": "completion", "content": "x"}, st)["verdict"] == ACCEPT


def test_s1_strict_g_rejects_observed_tier_assertion():
    # skipper preset: a merely-observed (read-only) grounding must NOT
    # license an assertion — verified tier is required (spec module 6)
    st = GateState.for_model_class("skipper", claim_surface={"file.txt"})
    st.grounded_this_turn = True
    assert boundary_check(
        {"claim_type": "assertion", "content": "x"}, st)["verdict"] == REJECT


def test_s2_strict_g_accepts_verified_tier_assertion():
    st = GateState.for_model_class("skipper", claim_surface={"file.txt"})
    st.grounded_this_turn = True
    st.verified_this_turn = True
    assert boundary_check(
        {"claim_type": "assertion", "content": "x"}, st)["verdict"] == ACCEPT


# ---------------------------------------------- Integration: turn_loop leaks

def test_l1_verified_latch_survives_later_noop_call():
    # edit -> re-read (verified) -> repeat read (novelty-defeated) -> completion
    out, _ = turn_loop([
        {"type": "tool_call", "tool": "write", "args": "file.txt", "result": "ok",
         "mutating": True},
        {"type": "tool_call", "tool": "read", "args": "file.txt", "result": "new_data"},
        {"type": "tool_call", "tool": "read", "args": "file.txt", "result": "new_data"},
        {"type": "terminal", "attempt": {"claim_type": "completion", "content": "done"}},
    ], S())
    assert out == "done"


def test_l2_non_qualifying_call_leaves_halt_sticky():
    # halted -> novelty-defeated read does NOT clear halt -> reasoning refused
    _, tr = turn_loop([
        {"type": "tool_call", "tool": "read", "args": "file.txt", "result": "d"},
        {"type": "terminal", "attempt": {"claim_type": "completion", "content": "no"}},
        {"type": "tool_call", "tool": "read", "args": "file.txt", "result": "d"},
        {"type": "reasoning"},
    ], S())
    assert tr[-1][0] == "refused_reasoning"


def _l3_script():
    return (
        [{"type": "terminal", "attempt": {"claim_type": "assertion", "content": "no"}}]
        + [{"type": "reasoning"}] * 3
        + [{"type": "terminal", "attempt": {"claim_type": "assertion", "content": "no"}},
           {"type": "terminal", "attempt": {"claim_type": "unverified",
                                            "content": "unverified"}}]
    )


def test_l3_starved_diverger_exits_via_unverified():
    st = S()
    st.budget = 2
    out, _ = turn_loop(_l3_script(), st)
    assert out == "unverified"


def test_r4_every_halted_reasoning_step_refused():
    st = S()
    st.budget = 2
    _, tr = turn_loop(_l3_script(), st)
    assert all(t[0] == "refused_reasoning" for t in tr[1:4])


def test_t11_date_only_change_is_verifiable_in_turn_loop():
    # the observed content differs only by a normalized-away date: the
    # mutation epoch in the novelty hash must still let the post-mutation
    # re-read verify (before the epoch moved into the hash, this task could
    # NEVER complete through the reference loop)
    out, tr = turn_loop([
        {"type": "tool_call", "tool": "read", "args": "file.txt",
         "result": "expires: 2026-07-10"},
        {"type": "tool_call", "tool": "write", "args": "file.txt", "result": "ok",
         "mutating": True},
        {"type": "tool_call", "tool": "read", "args": "file.txt",
         "result": "expires: 2026-07-11"},   # normalizes identical to read #1
        {"type": "terminal", "attempt": {"claim_type": "completion", "content": "done"}},
    ], S())
    assert out == "done"


def test_t13_extractor_may_return_any_iterable_or_bare_string():
    st = S(extractors={"statfile": lambda a, r: ["file.txt"]})   # list, not set
    st.current_step = 1
    assert classify_observation(
        "statfile", "h7", "ino:8812732", st, True)["grounds_assertion"]
    # a bare string is ONE identifier, not a character set
    st2 = S(extractors={"statfile": lambda a, r: "file.txt"})
    st2.current_step = 1
    assert classify_observation(
        "statfile", "h7", "ino:8812732", st2, True)["grounds_assertion"]


def test_b3_refill_cap_invariant_is_enforced():
    for bad in ({"refill": 0}, {"cap": 2, "refill": 2}, {"cap": 0}):
        try:
            GateState.for_model_class("default", **bad)
            assert False, "expected ValueError for %r" % bad
        except ValueError:
            pass


def test_b1_budget_floors_at_zero_and_recovers_in_one_observation():
    # entry-heavy reasoning must not dig an unrecoverable pit: after the
    # floor, ONE qualifying observation restores assert-ability
    st = GateState.for_model_class("diverger", claim_surface={"file.txt"})
    out, tr = turn_loop(
        [{"type": "reasoning"}] * 6
        + [{"type": "tool_call", "tool": "read", "args": "file.txt", "result": "d"},
           {"type": "terminal", "attempt": {"claim_type": "assertion", "content": "ok"}}],
        st)
    assert st.budget >= 0
    assert out == "ok"


def test_b2_for_model_class_accepts_cap_and_refill_overrides():
    st = GateState.for_model_class("diverger", cap=10, refill=3)
    assert (st.budget, st.cap, st.refill, st.strict_g) == (10, 10, 3, False)


def test_m1_new_mutation_invalidates_prior_verification():
    # write -> verifying read -> ANOTHER write -> completion must REJECT
    # (the verification no longer postdates the last mutation)
    out, tr = turn_loop([
        {"type": "tool_call", "tool": "write", "args": "file.txt", "result": "ok",
         "mutating": True},
        {"type": "tool_call", "tool": "read", "args": "file.txt", "result": "v1"},
        {"type": "tool_call", "tool": "write", "args": "file.txt", "result": "ok2",
         "mutating": True},
        {"type": "terminal", "attempt": {"claim_type": "completion", "content": "done"}},
        {"type": "tool_call", "tool": "read", "args": "file.txt", "result": "v2"},
        {"type": "terminal", "attempt": {"claim_type": "completion", "content": "done"}},
    ], S())
    verdicts = [t[1] for t in tr if t[0] == "terminal"]
    assert verdicts == ["REJECT", "ACCEPT"]    # re-verified after 2nd write
    assert out == "done"


# ------------------------------------------------------- bare-python runner

if __name__ == "__main__":
    failures = []
    cases = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
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
