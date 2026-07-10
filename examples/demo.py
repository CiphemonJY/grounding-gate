"""Same agent, with and without the gate. Zero LLM calls, fully deterministic.

Two scripted agents exhibit the two canonical failure modes:
  1. The SKIPPER edits a config file and immediately claims completion —
     without ever reading the file back.
  2. The DIVERGER reasons in circles and then ships a confident assertion —
     without a single observation.

An ungated loop emits whatever the agent submits. The gated loop rejects the
terminal, tells the agent its only legal moves, and the claim ships only after
a qualifying observation — or exits as a typed `unverified`, never a confident
hallucination.

Run: ``python examples/demo.py`` (asserts its own expected outcomes; exit 0).
"""

import sys
from pathlib import Path

try:
    import grounding_gate  # noqa: F401
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grounding_gate import GateState, turn_loop


def ungated_loop(script):
    """What most agent harnesses do: emit the first terminal the model offers."""
    observations = 0
    for step in script:
        if step["type"] == "tool_call":
            observations += 1
        if step["type"] == "terminal":
            return step["attempt"]["content"], observations
    return None, observations


def show(title, lines):
    print(f"\n=== {title}")
    for line in lines:
        print(f"    {line}")


# ---------------------------------------------------------------- Scenario 1
# Task: set timeout=30 in app.cfg, then confirm it's fixed.
SKIPPER_NAIVE = [
    {"type": "tool_call", "tool": "write", "args": "app.cfg timeout=30",
     "result": "ok", "mutating": True},
    {"type": "terminal",
     "attempt": {"claim_type": "completion", "content": "Fixed: timeout is now 30."}},
]
# What a gated model does after seeing REJECT + legal_next: it grounds.
SKIPPER_COACHED = SKIPPER_NAIVE + [
    {"type": "tool_call", "tool": "read", "args": "app.cfg", "result": "timeout=30"},
    {"type": "terminal",
     "attempt": {"claim_type": "completion", "content": "Fixed: timeout is now 30."}},
]

print("SCENARIO 1 — the skipper: edit, then claim completion with no post-check")

out, obs = ungated_loop(SKIPPER_NAIVE)
show("WITHOUT gate", [
    f"shipped: {out!r}",
    f"observations of app.cfg after the write: 0 (write result {'ok'!r} is self-report)",
    "the 'completion' is unverified — if the write silently failed, this is a lie",
])
assert out == "Fixed: timeout is now 30."

state = GateState.for_model_class("skipper", claim_surface={"app.cfg"})
out, trace = turn_loop(SKIPPER_COACHED, state)
verdicts = [t for t in trace if t[0] == "terminal"]
show("WITH gate", [
    f"first completion attempt: {verdicts[0][1]} (mutating call never self-grounds)",
    "legal next moves surfaced: qualifying_tool_call | unverified_terminal",
    "agent re-reads app.cfg -> post-mutation observation -> verified tier",
    f"second attempt: {verdicts[1][1]} -> shipped: {out!r}",
])
assert [v[1] for v in verdicts] == ["REJECT", "ACCEPT"]
assert out == "Fixed: timeout is now 30."

# ---------------------------------------------------------------- Scenario 2
# Task: report the value of retries in svc.cfg. The model never looks.
DIVERGER = (
    [{"type": "reasoning"}] * 3
    + [{"type": "terminal",
        "attempt": {"claim_type": "assertion",
                    "content": "retries is definitely 5 (I am confident)."}}]
    + [{"type": "reasoning"}] * 2
    + [{"type": "terminal",
        "attempt": {"claim_type": "assertion",
                    "content": "retries is definitely 5 (I am confident)."}},
       {"type": "terminal",
        "attempt": {"claim_type": "unverified",
                    "content": "unverified: could not confirm retries; needs a read of svc.cfg"}}]
)

print("\nSCENARIO 2 — the diverger: reason in circles, ship a confident guess")

out, obs = ungated_loop(DIVERGER)
show("WITHOUT gate", [
    f"shipped: {out!r}",
    f"tool calls made before shipping: {obs}",
    "a confident factual claim, grounded in nothing",
])
assert obs == 0

state = GateState.for_model_class("diverger", claim_surface={"svc.cfg"})
out, trace = turn_loop(DIVERGER, state)
refused = sum(1 for t in trace if t[0] == "refused_reasoning")
verdicts = [t[1] for t in trace if t[0] == "terminal"]
show("WITH gate", [
    f"assertion attempts: {verdicts[:-1]} (ungrounded; budget starves the loop)",
    f"reasoning steps refused while halted: {refused}",
    f"exit: {verdicts[-1]} -> shipped: {out!r}",
    "the honest answer ships as a typed `unverified`, not a confident hallucination",
])
assert verdicts == ["REJECT", "REJECT", "ACCEPT"]
assert out.startswith("unverified")

print("\nAll demo assertions passed.")
