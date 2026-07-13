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
    LEGAL_NEXT,
    REJECT,
    GateState,
    boundary_check,
    classify_observation,
    turn_loop,
)
from grounding_gate.verifiers import StubVerifier


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


# ------------------------------------------- Module 5: verify_with escalation
# The optional verifier tier can ONLY downgrade a structural ACCEPT to the typed
# `unverified` path; it can NEVER upgrade a structural REJECT. verifier=None is a
# byte-identical no-op (the whole 30-case floor suite above runs with it).

def test_v1_verifier_none_is_noop():
    # a grounded assertion with no verifier accepts exactly as it does today
    st = S(); st.grounded_this_turn = True
    v = boundary_check({"claim_type": "assertion", "content": "x"}, st, None)
    assert v == {"verdict": ACCEPT, "legal_next": []}
    assert st.halted is False


def test_v2_downgrade_of_structural_accept():
    # floor ACCEPTs (grounded), verifier confidence 0.0 < 0.5 -> DOWNGRADE
    st = S(); st.grounded_this_turn = True
    v = boundary_check({"claim_type": "assertion", "content": "x"}, st, StubVerifier(0.0))
    assert v["verdict"] == REJECT
    assert v["downgraded_by_verifier"] is True
    assert v["confidence"] == 0.0
    assert v["legal_next"] == LEGAL_NEXT   # the typed-unverified path
    assert st.halted is True


def test_v3_verifier_cannot_upgrade_reject():
    # UNgrounded assertion is a structural REJECT — the verifier is NEVER
    # consulted (no downgraded flag), proving floor-first / never-upgrade
    st = S()   # grounded_this_turn stays False
    v = boundary_check({"claim_type": "assertion", "content": "x"}, st, StubVerifier(1.0))
    assert v["verdict"] == REJECT
    assert "downgraded_by_verifier" not in v


def test_v4_unverified_immune_to_verifier():
    # the typed escape hatch is never escalated, even by a 0.0 verifier
    st = S()
    v = boundary_check({"claim_type": "unverified", "content": "x"}, st, StubVerifier(0.0))
    assert v["verdict"] == ACCEPT


def test_v5_none_immune_to_verifier():
    # a non-claim terminal is exempt and never escalated
    st = S()
    v = boundary_check({"claim_type": "none", "content": "x"}, st, StubVerifier(0.0))
    assert v["verdict"] == ACCEPT


def test_v6_abstain_leaves_accept():
    # a verifier that returns None ABSTAINS -> the floor's ACCEPT stands
    st = S(); st.grounded_this_turn = True
    v = boundary_check({"claim_type": "assertion", "content": "x"}, st,
                       StubVerifier(rule=lambda *a: None))
    assert v == {"verdict": ACCEPT, "legal_next": []}


def test_v7_turn_loop_routes_to_unverified():
    # a grounded-completion script the FLOOR would ship, downgraded by the
    # verifier: with only the completion terminal, nothing ships (out is None);
    # a following typed-unverified terminal is accepted WITHOUT the verifier.
    base = [
        {"type": "tool_call", "tool": "write", "args": "file.txt", "result": "ok",
         "mutating": True},
        {"type": "tool_call", "tool": "read", "args": "file.txt", "result": "new_data"},
        {"type": "terminal", "attempt": {"claim_type": "completion", "content": "done"}},
    ]
    out, tr = turn_loop(base, S(), StubVerifier(0.0))
    assert out is None
    assert [t[1] for t in tr if t[0] == "terminal"] == ["REJECT"]

    out2, tr2 = turn_loop(base + [
        {"type": "terminal", "attempt": {"claim_type": "unverified",
                                         "content": "unverified: could not confirm"}},
    ], S(), StubVerifier(0.0))
    assert out2 == "unverified: could not confirm"
    assert [t[1] for t in tr2 if t[0] == "terminal"] == ["REJECT", "ACCEPT"]


def test_v8_criteria_reflect_strict_g():
    # skipper (strict_g) routes an ACCEPTed assertion to COMPLETION-tier criteria
    seen = {}

    def rule(claim, observations, criteria):
        seen["criteria"] = criteria
        return 1.0

    st = GateState.for_model_class("skipper", claim_surface={"file.txt"})
    st.verified_this_turn = True   # strict-G needs verified tier to reach ACCEPT
    boundary_check({"claim_type": "assertion", "content": "x"}, st, StubVerifier(rule=rule))
    assert [n for n, q in seen["criteria"]] == ["effect_shown", "no_overreach"]


# ------------------------------------------------- Module 5: progress telemetry

def test_progress_shape_zero_token():
    st = S()
    p1 = st.progress()
    assert set(p1) == {
        "budget", "cap", "refill", "budget_headroom", "step",
        "grounded_this_turn", "verified_this_turn", "rejection_count",
        "steps_since_last_mutation", "steps_since_last_verification",
        "halted", "observations_this_turn", "unmet_signals"}
    assert p1["budget"] == 6 and p1["cap"] == 6 and p1["budget_headroom"] == 0
    assert p1["steps_since_last_mutation"] is None       # never mutated
    assert p1["steps_since_last_verification"] is None   # never verified
    assert p1["rejection_count"] == 0
    assert p1["observations_this_turn"] == 0
    assert p1["unmet_signals"] == []
    # pure function: calling again yields an equal dict, no side effects
    assert st.progress() == p1


def test_progress_counts_rejections():
    st = S()   # ungrounded -> each claim-bearing boundary_check REJECTs
    boundary_check({"claim_type": "assertion", "content": "x"}, st)
    boundary_check({"claim_type": "assertion", "content": "x"}, st)
    assert st.progress()["rejection_count"] == 2


def test_progress_steps_since():
    st = S()
    st.current_step = 10
    st.last_mutation_step = 4
    st.last_verification_step = 7
    p = st.progress()
    assert p["steps_since_last_mutation"] == 6
    assert p["steps_since_last_verification"] == 3
    # a zero marker reads as None (not a bogus delta)
    st.last_verification_step = 0
    assert st.progress()["steps_since_last_verification"] is None


def test_progress_after_mutate_verify_complete():
    st = S()
    _, _ = turn_loop([
        {"type": "tool_call", "tool": "write", "args": "file.txt", "result": "ok",
         "mutating": True},
        {"type": "tool_call", "tool": "read", "args": "file.txt", "result": "new_data"},
        {"type": "terminal", "attempt": {"claim_type": "completion", "content": "done"}},
    ], st)
    p = st.progress()
    assert p["verified_this_turn"] is True
    assert p["observations_this_turn"] >= 1
    assert p["steps_since_last_verification"] == 0   # verified at the current step


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
