"""Reproducible preset-tuning harness — sweep CAP/REFILL/strict_g on seeded runs.

This DOGFOODS the sibling ``lcb-gate`` package: it ranks candidate presets by a
paired-seed win-rate lower confidence bound (``lcb_gate.compare``), using common
random numbers (the SAME seeds for every candidate) so the comparison has power.
It tunes the STRUCTURAL floor only — no LLM verifier is involved, so every run is
hermetic and offline given ``lcb-gate`` installed.

Honesty contract (mirrors the design): NO fabricated "tuned" numbers are
committed. ``main`` PRINTS a ranked table each run and the module writes nothing;
the only durable claim is this docstring. ``lcb-gate`` is an OPTIONAL example
dependency (examples are not shipped in the wheel) — it is imported lazily and,
if absent, the script prints an install hint and exits 0.

Run: ``python examples/tune_presets.py``      (self-checks, then exits 0)
     ``python examples/tune_presets.py --profile diverger --n 300``   (ranked table)

Self-check (like examples/demo.py): asserts invariants that need no fabricated
tuning data — ``score`` is deterministic, ``outcome_ok`` never rewards a
confident ungrounded claim, and (only if ``lcb-gate`` is installed) a preset
compared against ITSELF yields win_rate == 0.5 and is not ``.better``.
"""

import random
import sys
from pathlib import Path

try:
    import grounding_gate  # noqa: F401
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grounding_gate import ACCEPT, GateState, turn_loop

CLAIM_SURFACE = {"app.cfg"}
DEFAULT_PRESET = {"cap": 6, "refill": 2, "strict_g": False}   # the shipped default


def synth_transcript(rng, profile):
    """Deterministic ``turn_loop`` script for one seed.

    Common random numbers: the script depends ONLY on ``rng`` (seeded from
    ``profile`` + seed in ``score``) and ``profile`` — never on the preset — so
    the SAME seed yields the identical script for every candidate preset, which
    is where the paired comparison gets its power. ``profile`` shapes the
    ground/reason/skip mix:

      * ``skipper``  — mutate then claim completion, often WITHOUT re-reading.
      * ``diverger`` — reason a lot, then assert, often with NO observation.
      * ``mixed``    — a middle blend.

    Roughly half the scripts also offer a typed ``unverified`` fallback, so a
    coached agent has an honest exit; the rest can get structurally stuck under a
    mismatched preset (the signal the sweep is measuring).
    """
    if profile == "diverger":
        reason_n = rng.randint(3, 8)
        grounds = rng.random() < 0.5
    elif profile == "skipper":
        reason_n = rng.randint(0, 2)
        grounds = rng.random() < 0.6
    else:  # mixed
        reason_n = rng.randint(1, 5)
        grounds = rng.random() < 0.7

    steps = [{"type": "reasoning"} for _ in range(reason_n)]

    if profile == "skipper":
        steps.append({"type": "tool_call", "tool": "write", "args": "app.cfg",
                      "result": "ok", "mutating": True})
        if grounds:   # the post-mutation re-read that verifies the completion
            steps.append({"type": "tool_call", "tool": "read", "args": "app.cfg",
                          "result": "v=%d" % rng.randint(0, 1_000_000)})
        steps.append({"type": "terminal",
                      "attempt": {"claim_type": "completion", "content": "done"}})
    else:
        if grounds:   # an observed-tier read (verified-tier under strict_g? no)
            steps.append({"type": "tool_call", "tool": "read", "args": "app.cfg",
                          "result": "v=%d" % rng.randint(0, 1_000_000)})
        steps.append({"type": "terminal",
                      "attempt": {"claim_type": "assertion", "content": "answer"}})

    if rng.random() < 0.5:
        steps.append({"type": "terminal",
                      "attempt": {"claim_type": "unverified",
                                  "content": "unverified: could not confirm"}})
    return steps


def outcome_ok(out, trace):
    """1.0 iff the gate produced the RIGHT structural outcome, else 0.0.

    RIGHT = a genuinely grounded claim shipped, OR the honest typed ``unverified``
    exit. WRONG = the agent got structurally stuck (no terminal shipped) OR — and
    this is the load-bearing guard — a NON-unverified claim shipped that the
    trace does not back with both a gate ACCEPT and a qualifying observation.
    A confident ungrounded claim can never be rewarded: it is checked against the
    trace, not merely by "something shipped".
    """
    if out is None:
        return 0.0
    if str(out).startswith("unverified"):
        return 1.0
    accepted = any(t[0] == "terminal" and t[1] == ACCEPT for t in trace)
    grounded = any(t[0] == "tool_call"
                   and (t[1]["grounds_assertion"] or t[1]["grounds_completion"])
                   for t in trace)
    return 1.0 if (accepted and grounded) else 0.0


def score(preset, seed, profile):
    """Structural-outcome score in [0,1] for one (preset, seed, profile).

    Seeds a fresh RNG from ``profile`` + ``seed`` (so it is reproducible and
    preset-independent), builds the scripted transcript, and drives it through
    the zero-LLM ``turn_loop`` on a fresh state built from ``preset``. Pure and
    deterministic: the same arguments always return the same value.
    """
    rng = random.Random("%s:%s" % (profile, seed))
    script = synth_transcript(rng, profile)
    state = GateState.for_model_class(
        "default", claim_surface=set(CLAIM_SURFACE),
        cap=preset["cap"], refill=preset["refill"], strict_g=preset["strict_g"])
    out, trace = turn_loop(script, state)
    return outcome_ok(out, trace)


def sweep(seeds, profile, grid, champion=None):
    """Rank ``grid`` presets against ``champion`` by paired win-rate LCB.

    Lazily imports ``lcb_gate.compare`` (optional example dep). Each candidate is
    compared on the SAME ``seeds`` as the champion (common random numbers), and
    the results are sorted by ``CompareResult.lcb`` descending — the honest
    "better than the champion" bar is ``lcb > 0.5``. Returns the ranked list of
    ``(preset, CompareResult)`` pairs, or ``None`` if ``lcb-gate`` is absent.
    """
    try:
        from lcb_gate import compare
    except ImportError:
        print("preset tuning needs the sibling package: pip install lcb-gate")
        return None
    if champion is None:
        champion = dict(DEFAULT_PRESET)
    seeds = list(seeds)
    ranked = []
    for cand in grid:
        res = compare(lambda s: score(cand, s, profile),
                      lambda s: score(champion, s, profile),
                      seeds=seeds)
        ranked.append((cand, res))
    ranked.sort(key=lambda pair: pair[1].lcb, reverse=True)
    return ranked


def _default_grid():
    grid = []
    for cap in (4, 6, 8):
        for refill in (1, 2, 3):
            if not 1 <= refill < cap:           # the enforced state.py invariant
                continue
            for strict_g in (False, True):
                grid.append({"cap": cap, "refill": refill, "strict_g": strict_g})
    return grid


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", default="mixed",
                        choices=["skipper", "diverger", "mixed"])
    parser.add_argument("--n", type=int, default=200, help="number of seeds")
    parser.add_argument("--grid", default="default",
                        help="preset grid to sweep (only 'default' is built in)")
    args = parser.parse_args(argv)

    grid = _default_grid()
    ranked = sweep(range(args.n), args.profile, grid)
    if ranked is None:
        return 0   # lcb-gate absent; the hint was already printed
    print("Ranked presets for profile=%r over %d seeds (champion=%s):"
          % (args.profile, args.n, DEFAULT_PRESET))
    for cand, res in ranked:
        print("  %-42s %s" % (cand, res))   # str(CompareResult); nothing is saved
    return 0


# --------------------------------------------------------------- self-check

def _check_no_lcb_needed():
    """Invariants that need no lcb-gate and no fabricated tuning data."""
    for profile in ("skipper", "diverger", "mixed"):
        for seed in range(30):
            a = score(DEFAULT_PRESET, seed, profile)
            b = score(DEFAULT_PRESET, seed, profile)
            assert a == b, ("score not deterministic", profile, seed, a, b)
            assert a in (0.0, 1.0)
    # outcome_ok never rewards a confident ungrounded claim...
    assert outcome_ok("retries is definitely 5", [("terminal", ACCEPT)]) == 0.0
    assert outcome_ok("confident guess", []) == 0.0
    assert outcome_ok(None, []) == 0.0
    # ...but does reward the honest exit and a genuinely grounded finish
    assert outcome_ok("unverified: could not confirm", []) == 1.0
    grounded = [("tool_call", {"grounds_assertion": True, "grounds_completion": False}),
                ("terminal", ACCEPT)]
    assert outcome_ok("answer", grounded) == 1.0


def _check_with_lcb(compare):
    """A preset compared against ITSELF is a tie: win_rate 0.5, not better."""
    seeds = list(range(60))
    res = compare(lambda s: score(DEFAULT_PRESET, s, "mixed"),
                  lambda s: score(DEFAULT_PRESET, s, "mixed"),
                  seeds=seeds)
    assert not res.better
    assert abs(res.win_rate - 0.5) < 1e-9, res.win_rate


if __name__ == "__main__":
    _check_no_lcb_needed()
    try:
        from lcb_gate import compare
    except ImportError:
        print("tune_presets: core self-check passed; lcb-gate not installed — "
              "skipping the compare() self-check (pip install lcb-gate to run the "
              "full harness). Exiting 0.")
        sys.exit(0)
    _check_with_lcb(compare)
    print("tune_presets: all self-checks passed (including lcb-gate compare()).")
    sys.exit(0)
