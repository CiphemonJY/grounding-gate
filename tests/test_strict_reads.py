"""Strict-reads mode: ``GateHooks(strict_reads=True)``.

The promise is one-sided and absolute: a claim of change is accepted only
after a direct file read (``Read``-style tool) of every file changed this
turn, taken after its last change and after the last shell command. The
shell parser may add obligations but never relax one, so a parser mistake
can block honest work but never let an unbacked claim through.

Three layers check it: the rules themselves; every leak reproduction from
the three adversarial reviews (all must be rejected); and a property test
that replays random sessions against an independent oracle of the rule.
Runs under pytest, or bare: ``python tests/test_strict_reads.py``.
"""

import random
import sys
from pathlib import Path

try:
    import grounding_gate  # noqa: F401
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from grounding_gate.adapters.claude_agent_sdk import GateHooks  # noqa: E402

import test_review_findings as review  # noqa: E402

STOP = {"hook_event_name": "Stop"}
PROMPT = {"hook_event_name": "UserPromptSubmit"}
drive = review.drive


def ptu(tool, tool_input, response="ok", cwd="/p"):
    return {"hook_event_name": "PostToolUse", "tool_name": tool,
            "tool_input": tool_input, "tool_response": response, "cwd": cwd}


def strict(*steps, cwd="/p", surface=()):
    gate = GateHooks(claim_surface=set(surface), home="/home/u", strict_reads=True)
    for step in steps:
        if step == "PROMPT":
            drive(gate.user_prompt_submit(PROMPT, None, None))
        elif step[0] == "__fail__":
            drive(gate.post_tool_use_failure(
                {"hook_event_name": "PostToolUseFailure", "tool_name": "Bash",
                 "tool_input": step[1], "error": "exit 1", "cwd": cwd}, None, None))
        elif step[0] == "__fail_edit__":
            drive(gate.post_tool_use_failure(
                {"hook_event_name": "PostToolUseFailure", "tool_name": step[1],
                 "tool_input": step[2], "error": "no such file", "cwd": cwd}, None, None))
        else:
            drive(gate.post_tool_use(ptu(*step, cwd=cwd), None, None))
    out = drive(gate.stop(STOP, None, None))
    return ("ACCEPT" if out == {} else "REJECT"), gate, out


def E(p):
    return ("Edit", {"file_path": p})


def R(p, text="x=1"):
    return ("Read", {"file_path": p}, text)


def B(cmd, out="ok"):
    return ("Bash", {"command": cmd},
            {"stdout": out, "stderr": "", "interrupted": False, "isImage": False})


A, B_ = "/p/a.cfg", "/p/b.cfg"


# ------------------------------------------------------------------ rules

def test_a_direct_read_verifies():
    assert strict(E(A), R(A))[0] == "ACCEPT"


def test_shell_output_never_verifies():
    for cmd in ("cat a.cfg", "head -n 5 a.cfg", "git diff a.cfg", "sed -n 1,5p a.cfg"):
        assert strict(E(A), B(cmd, "+x=1"))[0] == "REJECT", cmd


def test_search_and_listing_never_verify():
    grep = ("Grep", {"pattern": "x", "path": A, "output_mode": "content"}, "x=1")
    glob = ("Glob", {"pattern": "*.cfg"}, A)
    for step in (grep, glob):
        assert strict(E(A), step)[0] == "REJECT"


def test_any_shell_command_reowes_this_turns_changes():
    verdict, gate, out = strict(E(A), R(A), B("pytest -q", "3 passed"))
    assert verdict == "REJECT" and A in gate.state.pending_verification
    assert "strict reads" in out["reason"]
    assert strict(E(A), R(A), B("pytest -q", "3 passed"), R(A, "x=2"))[0] == "ACCEPT"
    # even an apparently read-only command: strict mode trusts no parse
    assert strict(E(A), R(A), B("git status", " M a.cfg"))[0] == "REJECT"


def test_shell_writes_are_owed():
    verdict, gate, _ = strict(B("sed -i s/1/2/ a.cfg"), B("sed -i s/1/2/ b.cfg"), R(A))
    assert verdict == "REJECT" and B_ in gate.state.pending_verification
    assert strict(B("sed -i s/1/2/ a.cfg"), R(A, "x=2"))[0] == "ACCEPT"


def test_a_temp_file_the_command_created_can_move_or_go():
    assert strict(B("jq '.x=1' p.json > tmp.json && mv tmp.json p.json"),
                  R("/p/p.json", '{"x":1}'))[0] == "ACCEPT"
    assert strict(B("echo '{}' > req.json && curl -d @req.json localhost && rm req.json"),
                  )[1].state.pending_verification == set()


def test_parsed_moves_and_deletes_never_clear_earlier_debt():
    # strict mode's price: after moving or deleting a changed file, the turn
    # ends unverified rather than trusting the parse of `mv`/`rm`
    for cmd in ("mv a.cfg b.cfg", "rm a.cfg", "rm -rf /p", "git mv a.cfg x.cfg"):
        verdict, gate, _ = strict(E(A), B(cmd), R(B_))
        assert verdict == "REJECT" and A in gate.state.pending_verification, cmd


def test_failed_shell_command_still_owes():
    verdict, gate, _ = strict(E(B_), R(B_), ("__fail__", {"command": "sed -i s/1/2/ a.cfg"}),
                              R(B_, "b=2"))
    assert verdict == "REJECT" and A in gate.state.pending_verification


def test_each_turn_starts_clean():
    assert strict(E(A), "PROMPT", E(B_), R(B_))[0] == "ACCEPT"
    assert strict(E(A), R(A), "PROMPT", B("npm test"), R(B_), surface={B_})[0] == "ACCEPT"


def test_progress_reports_the_mode():
    _, gate, _ = strict(E(A), R(A))
    assert gate.progress()["strict_reads"] is True
    assert GateHooks().progress()["strict_reads"] is False


# ------------------------------------------- every review leak, in strict

def test_every_review_leak_is_rejected_in_strict_mode():
    cases = dict(review.LEAKS)
    cases.update(review.LEAKS_2)
    cases.update(review.LEAKS_3)
    cases.update({k: v for k, (want, v) in review.HOME_TRAPS.items() if want == "REJECT"})
    for name, steps in cases.items():
        assert strict(*steps)[0] == "REJECT", name


# ------------------------------------------ strict review findings

def _ev(tool, tool_input, response="ok", cwd="/p", **extra):
    event = ptu(tool, tool_input, response, cwd)
    event.update(extra)
    return event


def strict_events(*events, surface=()):
    gate = GateHooks(claim_surface=set(surface), home="/home/u", strict_reads=True)
    for e in events:
        if e["hook_event_name"] == "PostToolUse":
            drive(gate.post_tool_use(e, None, None))
        else:
            drive(gate.post_tool_use_failure(e, None, None))
    return "ACCEPT" if drive(gate.stop(STOP, None, None)) == {} else "REJECT"


def _bash(cmd, cwd="/p", **extra):
    return _ev("Bash", dict({"command": cmd}, **extra),
               {"stdout": "", "stderr": "", "interrupted": False, "isImage": False}, cwd)


def _E(p):
    return _ev("Edit", {"file_path": p})


def _R(p, text="c", **extra):
    return _ev("Read", dict({"file_path": p}, **extra), text)


STRICT_LEAKS = {
    "L1 glob write target": (_E("/p/src/x.py"), _bash("sed -i s/a/b/ src/*.py"),
                             _R("/p/src/x.py")),
    "L2 conditional rm of a changed file": (
        _E("/p/out.json"), _R("/p/out.json", "v1"),
        _bash("python gen.py > out.json || rm -f out.json"), _E("/p/x.py"), _R("/p/x.py")),
    "L2b mv -n may not move": (
        _E("/p/x.py"), _bash("sed -i s/a/b/ a.cfg && mv -n a.cfg b.cfg"),
        _R("/p/b.cfg"), _R("/p/x.py")),
    "L3 unparseable command": (_E("/p/x.py"), _bash("cat > notes.md <<EOF\nrun `make`\nEOF"),
                               _R("/p/x.py")),
    "L4 subagent edit after verification": (
        _E("/p/a.cfg"), _R("/p/a.cfg"), _ev("Task", {"prompt": "tidy"}, "done"),
        _ev("Edit", {"file_path": "/p/a.cfg"}, agent_id="sub1")),
    "L5 MCP move destination": (
        _E("/p/a.cfg"), _R("/p/a.cfg"),
        _ev("mcp__fs__move_file", {"source": "/p/c.cfg", "destination": "/p/d.cfg"}),
        _R("/p/a.cfg", "c2")),
    "L6 unknown tool after verification": (
        _E("/p/a.cfg"), _R("/p/a.cfg"), _ev("ApplyPatch", {"patch": "*** Update File: a"})),
    "L6b unknown path key": (
        _E("/p/x.py"), _ev("mcp__x__write_file", {"filepath": "/p/z.cfg", "content": "h"}),
        _R("/p/x.py")),
    "L7 mv onto an existing file": (_E("/p/x.py"), _bash("mv a.cfg.orig a.cfg"), _R("/p/x.py")),
    "L7b git checkout -- f": (_E("/p/x.py"), _bash("git checkout -- a.cfg"), _R("/p/x.py")),
    "L7c /usr/bin/sed": (_E("/p/x.py"), _bash("/usr/bin/sed -i s/a/b/ a.cfg"), _R("/p/x.py")),
    "L7d curl -o": (_E("/p/x.py"), _bash("curl -o a.json https://x/y"), _R("/p/x.py")),
    "L8 cd to an unknown place": (
        _E("/p/x.py"),
        _bash('cd "$(git rev-parse --show-toplevel)"; sed -i s/a/b/ README.md', "/p/sub"),
        _R("/p/sub/README.md"), _R("/p/x.py")),
    "L9 stderr to a file": (_E("/p/x.py"), _bash("python gen.py 2> err.log"), _R("/p/x.py")),
    "L10 background command": (
        _E("/p/a.cfg"), _bash("sleep 60; sed -i s/a/b/ a.cfg", run_in_background=True),
        _R("/p/a.cfg")),
    "L11 partial read": (_E("/p/a.cfg"), _R("/p/a.cfg", "  900\t}", offset=900, limit=1)),
    "L12 multi-file MCP read": (
        _E("/home/u/b.cfg"),
        _ev("mcp__fs__read_multiple_files", {"paths": ["/home/u/b.cfg"]},
            "/home/u/b.cfg: Error - Access denied")),
    "L13 untrusted MCP server": (_E("/p/a.cfg"), _ev("mcp__docker__read_file", {"path": "a.cfg"}, "x")),
    "L14 no cwd": (_ev("Bash", {"command": "sed -i s/1/2/ a.cfg"}, "", cwd=None),
                   _R("/q/a.cfg")),
}

STRICT_TRAPS = {
    "T1 plain cp": (_bash("cp b.cfg a.cfg"), _R("/p/a.cfg")),
    "T2 temp file moved into a directory": (
        _bash("jq . a.json > t.json && mv t.json out"), _R("/p/out/t.json")),
    "T3 created temp file deleted": (
        _bash("echo '{}' > req.json && curl -d @req.json localhost && rm req.json"),
        _E("/p/x.py"), _R("/p/x.py")),
}


def test_strict_review_leaks_are_rejected():
    for name, events in STRICT_LEAKS.items():
        assert strict_events(*events) == "REJECT", name


def test_strict_review_traps_are_accepted():
    for name, events in STRICT_TRAPS.items():
        assert strict_events(*events) == "ACCEPT", name


def test_create_directory_owes_nothing():
    gate = GateHooks(home="/home/u", strict_reads=True)
    drive(gate.post_tool_use(_ev("mcp__fs__create_directory", {"path": "/p/nd"}), None, None))
    assert gate.state.pending_verification == set()


# ------------------------------------------------------- property test

FILES = ["a.cfg", "b.cfg", "c.cfg"]
# (template, files it truly writes). `{f}`/`{g}` are files; `py` commands
# write unknown things (the oracle then owes every file changed so far).
SHELL = [
    ("sed -i s/1/2/ {f}", "{f}"), ("sed -ni '/k/p' {f}", "{f}"),
    ("sed -Ei 's/a/b/' {f}", "{f}"), ("echo x > {f}", "{f}"),
    ("echo x >> {f}", "{f}"), ("printf y | tee {f}", "{f}"),
    ("sort -o {f} {f}", "{f}"), ("cp {g} {f}", "{f}"), ("cat {f}", ""),
    ("cat {f} | wc -l", ""), ("git diff {f}", ""), ("grep -n x {f} || true", ""),
    ("jq . {f} > /tmp/o && mv /tmp/o {f}", "{f}"), ("out=$(sed -i s/1/2/ {f})", "{f}"),
    ("cat <<EOF > {f}\nx\nEOF", "{f}"), ("(cd /p && echo z > {f})", "{f}"),
    ("git commit -am \"$(cat <<'EOF'\nUse \"->\"\nEOF\n)\"", ""),
    ("python fix.py", "?"), ("pytest -q", "?"), ("make", "?"),
]


def _session(rng):
    """Random steps plus the ground truth the oracle needs."""
    steps, truth = [], []
    for _ in range(rng.randint(1, 9)):
        f, g = rng.choice(FILES), rng.choice(FILES)
        kind = rng.random()
        if kind < 0.25:
            tool = rng.choice(["Edit", "Write"])
            steps.append((tool, {"file_path": "/p/" + f}))
            truth.append(("change", {f}))
        elif kind < 0.55:
            steps.append(R("/p/" + f, "v%d" % rng.randint(0, 10 ** 6)))
            truth.append(("read", f))
        elif kind < 0.62:
            steps.append(("Grep", {"pattern": "x", "path": "/p/" + f,
                                   "output_mode": "content"}, "x=%d" % rng.randint(0, 99)))
            truth.append(("none", None))
        else:
            template, writes = rng.choice(SHELL)
            cmd = template.format(f=f, g=g)
            out = "v%d" % rng.randint(0, 10 ** 6)
            if rng.random() < 0.15:
                steps.append(("__fail__", {"command": cmd}))
            else:
                steps.append(B(cmd, out))
            written = set() if writes in ("", "?") else {writes.format(f=f, g=g)}
            truth.append(("shell", written))
    return steps, truth


def _oracle_accepts(truth):
    """Independent statement of the strict rule: every file changed this
    turn was read after its last change and after the last shell command."""
    last_change, last_shell, changed = {}, -1, set()
    for k, (kind, arg) in enumerate(truth):
        if kind == "change":
            for f in arg:
                last_change[f] = k
                changed.add(f)
        elif kind == "shell":
            last_shell = k
            for f in arg:
                last_change[f] = k
                changed.add(f)
    if not changed and last_shell < 0:
        return True           # nothing changed: not a completion claim
    for f in changed:
        after = max(last_change[f], last_shell)
        if not any(kind == "read" and arg == f and k > after
                   for k, (kind, arg) in enumerate(truth)):
            return False
    return True


def test_strict_never_accepts_what_the_oracle_rejects():
    rng = random.Random(20260925)
    checked = 0
    for _ in range(4000):
        steps, truth = _session(rng)
        changed = any(kind in ("change", "shell") for kind, _ in truth)
        if not changed:
            continue
        verdict = strict(*steps)[0]
        checked += 1
        assert not (verdict == "ACCEPT" and not _oracle_accepts(truth)), steps
    assert checked > 2000


if __name__ == "__main__":
    failures = []
    cases = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    for name, fn in cases:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as exc:
            print(f"  FAIL  {name}: {str(exc)[:300]}")
            failures.append(name)
    print(f"\n{'ALL PASS' if not failures else f'FAILED: {failures}'}"
          f" — {len(cases) - len(failures)}/{len(cases)}")
    sys.exit(1 if failures else 0)
