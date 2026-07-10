"""Full acceptance suite — 17 cases, ported 1:1 from the verified reference.

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
