"""Regression suite from an independent adversarial review of the adapter.

Each case is a reviewer-written reproduction with the reviewer's expected
verdict: LEAK cases are transcripts where the agent never saw the claimed
file's current content (the gate must REJECT); TRAP cases are transcripts
where the agent did the right thing (the gate must ACCEPT). Two findings are
recorded with an adjudicated expectation and the reason (T1b, T12).
Runs under pytest, or bare: ``python tests/test_review_findings.py``.
"""

import sys
from pathlib import Path

try:
    import grounding_gate  # noqa: F401
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grounding_gate.adapters.claude_agent_sdk import GateHooks

H = "/home/u/app"
STOP = {"hook_event_name": "Stop"}
PROMPT = {"hook_event_name": "UserPromptSubmit"}


def drive(coro):
    try:
        coro.send(None)
    except StopIteration as done:
        return done.value
    raise AssertionError("hook awaited")


def ptu(tool, tool_input, response="ok", cwd="/p"):
    return {"hook_event_name": "PostToolUse", "tool_name": tool,
            "tool_input": tool_input, "tool_response": response, "cwd": cwd}


def go(*steps, cwd="/p", surface=(), home="/home/u"):
    gate = GateHooks(claim_surface=set(surface), home=home)
    for step in steps:
        drive(gate.post_tool_use(ptu(*step, cwd=cwd), None, None))
    out = drive(gate.stop(STOP, None, None))
    return ("ACCEPT" if out == {} else "REJECT"), sorted(gate.state.pending_verification)


def edit(path):
    return ("Edit", {"file_path": path})


def read(path, text="x=1"):
    return ("Read", {"file_path": path}, text)


def bash(command, output="ok"):
    return ("Bash", {"command": command}, output)


def bash_dict(command, stdout):
    return bash(command, {"stdout": stdout, "stderr": "", "interrupted": False,
                          "isImage": False})


LEAKS = {
    "L1 heredoc body is data, not commands": (
        edit("/p/a.cfg"), bash("cat > /p/clean.sh <<'EOF'\nrm -f /p/a.cfg\nEOF"),
        read("/p/clean.sh", "rm -f /p/a.cfg")),
    "L2 a listed file is not shown": (
        edit("/p/a.cfg"), edit("/p/b.cfg"),
        bash("cat b.cfg && ls -l a.cfg", "b=1\n-rw a.cfg")),
    "L2b content piped into wc is not shown": (
        edit("/p/a.cfg"), edit("/p/b.cfg"), bash("cat a.cfg | wc -l; cat b.cfg", "3\nb=1")),
    "L3 2>&1 then wc": (edit("/p/a.cfg"), bash("cat a.cfg 2>&1 | wc -l", "3")),
    "L3b &>/dev/null": (edit("/p/a.cfg"), bash("cat a.cfg &>/dev/null", "")),
    "L3c 1>/dev/null": (edit("/p/a.cfg"), bash("cat a.cfg 1>/dev/null", "")),
    "L4 git diff --stat": (edit("/p/a.cfg"), bash("git diff --stat a.cfg", " a.cfg | 2 +-")),
    "L4b git diff of an untracked file": (
        ("Write", {"file_path": "/p/new.py", "content": "x"}), bash("git diff new.py", "")),
    "L4c sed line count": (edit("/p/a.cfg"), bash("sed -n '$=' a.cfg", "12")),
    "L4d awk END summary": (edit("/p/a.cfg"), bash("awk 'END{print NR}' a.cfg", "12")),
    "L4e diff -q": (edit("/p/a.cfg"), bash("diff -q a.cfg a.cfg.orig", "Files differ")),
    "L4f rg --count-matches": (edit("/p/a.cfg"), bash("rg --count-matches timeout a.cfg", "1")),
    "L4g git show of an old commit": (
        edit("/p/a.cfg"), bash("git show HEAD~3 -- a.cfg", "-t=1\n+t=5")),
    "L5 single-file grep prints no file names": (
        edit("/p/a.cfg"), bash("grep timeout CHANGELOG", "a.cfg: raised timeout to 30")),
    "L6 Grep tool: a matched line naming the file": (
        edit("/p/a.cfg"),
        ("Grep", {"pattern": "timeout", "path": "/p/docs", "output_mode": "content"},
         {"mode": "content", "content": "/p/docs/ops.md:3:set timeout in a.cfg"})),
    "L6b WebSearch query naming the file": (
        edit("/p/a.cfg"), ("WebSearch", {"query": "a.cfg timeout syntax"}, "results")),
    "L8 a.cfg~ is not a.cfg": (edit("/p/a.cfg"), read("/p/a.cfg~", "old")),
    "L8b rm of a.cfg:Zone.Identifier keeps a.cfg owed": (
        edit("/p/a.cfg"), edit("/p/b.cfg"), bash("rm a.cfg:Zone.Identifier"),
        read("/p/b.cfg", "b=1")),
    "L8c paths with spaces don't alias": (
        ("Write", {"file_path": "/p/my notes.txt", "content": "x"}),
        read("/p/my docs/notes.txt", "other")),
    "L10 Bash description words don't widen relevance": (
        edit("/p/a.cfg"), read("/p/a.cfg", "t=1"),
        ("Bash", {"command": "./fix.sh", "description": "Apply the fix"}, "ok"),
        read("/p/LICENSE", "granted to the person")),
}

TRAPS = {
    "T1 atomic jq edit through a temp file": (
        bash("jq '.x=1' pkg.json > tmp.json && mv tmp.json pkg.json"),
        read("/p/pkg.json", '{"x":1}')),
    "T2 mv into an existing directory": (
        edit("/p/old.cfg"), bash("mv old.cfg archive"), read("/p/archive/old.cfg")),
    "T2b mv of a directory": (
        edit("/p/src/a.py"), bash("mv src lib"), read("/p/lib/a.py")),
    "T2c mv -t": (edit("/p/a.cfg"), bash("mv -t archive a.cfg"), read("/p/archive/a.cfg")),
    "T4 => inside a quoted grep pattern": (
        edit(H + "/app.js"), read(H + "/app.js", "const f = () => 1"),
        bash('grep -n "=>" app.js', "1:const f = () => 1")),
    "T5 heredoc body redirect is data": (
        bash("cat > run.sh <<'EOF'\npython main.py > out.log\nEOF"),
        read("/p/run.sh", "python main.py > out.log")),
    "T7 grep -rn with the SDK's dict response": (
        edit("/p/a.cfg"), bash_dict("grep -rn timeout .", "./a.cfg:1:timeout=30")),
    "T7b git diff with the SDK's dict response": (
        edit("/p/a.cfg"), bash_dict("git diff", "--- a/a.cfg\n+++ b/a.cfg\n+timeout=30")),
    "T9 git --no-pager diff": (
        edit("/p/a.cfg"), bash("git --no-pager diff a.cfg", "+++ b/a.cfg\n+t=1")),
    "T9b quoted | in a grep pattern": (
        edit("/p/a.cfg"), bash("grep -E 'timeout|retries' a.cfg", "timeout=30")),
    "T9c quoted | in a jq filter": (
        edit("/p/a.json"), bash("jq '.items[] | .name' a.json", '"x"')),
    "T9d test -f && cat": (edit("/p/a.cfg"), bash("test -f a.cfg && cat a.cfg", "t=1")),
    "T10 a log the same command showed": (
        edit("/p/a.py"), read("/p/a.py"),
        bash("npm test > test.log 2>&1; tail -n 50 test.log", "PASS")),
}

HOME_TRAPS = {
    "L7 README mentioning the Makefile": ("REJECT", (
        edit(H + "/Makefile"), read(H + "/README.md", "Build targets live in the Makefile"))),
    "T3 >> ~/.bashrc": ("ACCEPT", (
        bash("echo 'export FOO=1' >> ~/.bashrc"),
        read("/home/u/.bashrc", "export FOO=1"))),
    "T4b redirect to a quoted name with a space": ("ACCEPT", (
        bash('echo x > "my notes.txt"'), read(H + "/my notes.txt", "x"))),
    "T6 macOS sed -i ''": ("ACCEPT", (
        bash("sed -i '' 's/app.js/main.js/' index.html"),
        read(H + "/index.html", '<script src="main.js"></script>'))),
    "T11 cat ~/file": ("ACCEPT", (
        edit("/home/u/.bashrc"), bash("cat ~/.bashrc", "export FOO=1"))),
}


def test_leaks_are_rejected():
    for name, steps in LEAKS.items():
        assert go(*steps)[0] == "REJECT", name


def test_traps_are_accepted():
    for name, steps in TRAPS.items():
        assert go(*steps) == ("ACCEPT", []), name


def test_cwd_and_home_cases():
    for name, (want, steps) in HOME_TRAPS.items():
        assert go(*steps, cwd=H)[0] == want, name


# ------------------------------------------------ second review round
# Written against the new shell parser; same rules (reviewer's verdicts).

LEAKS_2 = {
    "sed -ni edits in place": (edit("/p/a.cfg"), bash("sed -ni '/keep/p' a.cfg", "")),
    "sed -Ei edits in place": (edit("/p/b.cfg"), bash("sed -Ei 's/30/60/' a.cfg"),
                               read("/p/b.cfg")),
    "sort -uo rewrites the file": (edit("/p/a.cfg"), bash("sort -uo a.cfg a.cfg", "")),
    "pipe into sort -o": (edit("/p/a.cfg"), bash("grep -v '^#' a.cfg | sort -o a.cfg", "")),
    "awk -i inplace": (edit("/p/a.cfg"), bash("awk -i inplace '{sub(/30/,\"60\")}1' a.cfg", "")),
    "awk -v with a redirecting program": (
        edit("/p/a.cfg"), read("/p/a.cfg"), bash("awk -v n=1 '{print > \"a.cfg\"}' x", "")),
    "assignment-only substitution": (
        edit("/p/a.cfg"), read("/p/a.cfg"),
        bash("out=$(sed -i 's/30/60/' a.cfg); echo done", "done")),
    "grep -e: the pattern is not a file": (
        edit("/p/a.cfg"), bash("grep -e a.cfg CHANGELOG", "bumped a.cfg")),
    "grep -A 3: the option value is not the pattern": (
        edit("/p/app.cfg"),
        bash("grep -n -A 3 app.cfg CHANGELOG.md", "12:bumped the timeout in app.cfg")),
    "another stage's output is not grep's": (
        edit("/p/a.cfg"), bash_dict("cat CHANGELOG; grep -rn foo src",
                                    "a.cfg: raised timeout\nsrc/x.py:1:foo")),
    "flake8 output is not grep's": (
        edit("/p/src/a.py"), bash_dict("flake8 src; grep -rn TODO docs",
                                       "src/a.py:3:1: E302 ...\ndocs/x.md:1:TODO")),
    "rg reading a pipe prints no file names": (
        edit("/p/a.cfg"), bash_dict("git log --format=%s | rg timeout", "a.cfg: raise timeout")),
    "git diff --cached shows the index": (
        edit("/p/a.cfg"), bash("git add a.cfg"), edit("/p/a.cfg"),
        bash("git diff --cached", "+++ b/a.cfg\n+t=1")),
    "git diff between two commits": (
        edit("/p/a.cfg"), bash("git diff HEAD~2 HEAD~1 -- a.cfg", "+++ b/a.cfg\n+t=1")),
    "git grep at a revision": (
        edit("/p/a.cfg"), bash("git grep timeout HEAD -- a.cfg", "HEAD:a.cfg:t=1")),
    "git show headers credit nothing": (
        edit("/p/a.cfg"), bash("git show HEAD -- a.cfg; git diff b.cfg",
                               "+++ b/a.cfg\n+t=1\n+++ b/b.cfg\n+x")),
    "a repo-root diff header is not web/package.json": (
        edit("/p/package.json"), edit("/p/web/package.json"),
        bash("git diff package.json", "--- a/package.json\n+++ b/package.json\n+x")),
    "git rm --cached keeps the file": (
        edit("/p/a.cfg"), edit("/p/b.cfg"), bash("git rm --cached a.cfg"), read("/p/b.cfg")),
    "git rm -n is a dry run": (
        edit("/p/a.cfg"), edit("/p/b.cfg"), bash("git rm -n a.cfg"), read("/p/b.cfg")),
    "pushd changes where sed ran": (
        bash("pushd web && sed -i 's/1.0/1.1/' package.json && popd"),
        read("/p/package.json", "{}")),
    "Grep tool: matched text in a single-file search": (
        edit("/p/a.cfg"),
        ("Grep", {"pattern": "timeout", "path": "/p/CHANGELOG", "output_mode": "content"},
         {"mode": "content", "content": "a.cfg: raised timeout"})),
    "an earlier stage's /dev/null": (edit("/p/a.cfg"), bash("cat a.cfg >/dev/null | grep x", "")),
    "tree -o writes a file": (edit("/p/a.cfg"), read("/p/a.cfg"), bash("tree -o a.cfg", "")),
    "ANSI-C quotes don't hide a later sed": (
        edit("/p/a.cfg"), read("/p/a.cfg"),
        bash("echo $'\\'\\'' ; sed -i s/x/y/ a.cfg ; echo \\'", "")),
    "arithmetic << is not a heredoc": (
        edit("/p/a.cfg"), read("/p/a.cfg"), bash("(( head = 1 << 4 ))\nsed -i s/x/y/ a.cfg", "")),
    "a quoted heredoc delimiter with a space": (
        edit("/p/a.cfg"), read("/p/a.cfg"),
        bash("cat <<'END X'\nhi\nEND X\nsed -i s/x/y/ a.cfg", "hi")),
    "if/then doesn't hide sed -i": (
        edit("/p/b.cfg"), bash("if true; then sed -i 's/a/b/' a.cfg; fi"), read("/p/b.cfg")),
    "a background cd changes nothing": (
        edit("/p/sub/a.cfg"), read("/p/sub/a.cfg"), bash("cd sub & sed -i s/x/y/ a.cfg"),
        read("/p/sub/a.cfg", "y")),
}

TRAPS_2 = {
    "writes to /dev/stderr are never owed": (
        bash("echo 'patching' > /dev/stderr; sed -i 's/30/60/' a.cfg"),
        bash("cat a.cfg", "t=60")),
    "tee /dev/stderr": (bash("echo t=60 | tee /dev/stderr > a.cfg"), bash("cat a.cfg", "t=60")),
    "process substitution target": (
        bash("sed -i s/30/60/ a.cfg > >(tee sed.log)"), bash("cat a.cfg", "t=60")),
    "BSD sed -i .bak": (bash("sed -i .bak 's/30/60/' a.cfg"), bash("cat a.cfg", "t=60")),
    "mv into a directory, then rename there": (
        edit("/p/a.cfg"), bash("mv a.cfg archive && mv archive/a.cfg archive/a-2024.cfg"),
        read("/p/archive/a-2024.cfg")),
    "mv src/ lib/ renames a directory": (
        edit("/p/src/a.py"), bash("mv src/ lib/"), read("/p/lib/a.py")),
    "rm -rf tmp/* removes nested files": (
        edit("/p/tmp/sub/x.txt"), edit("/p/b.cfg"), bash("rm -rf tmp/*"), read("/p/b.cfg", "b=1")),
    "cat < file": (edit("/p/a.cfg"), bash("cat < a.cfg", "t=1")),
    "awk comparison is not a redirect": (edit("/p/a.cfg"), bash("awk 'NR>=1 && NR<=40' a.cfg", "t=1")),
    "cd -P": (edit("/p/sub/a.cfg"), bash("cd -P sub && cat a.cfg", "t=1")),
    "Grep tool single file with -n": (
        edit("/p/a.cfg"),
        ("Grep", {"pattern": "t", "path": "/p/a.cfg", "output_mode": "content", "-n": True},
         {"mode": "content", "content": "1:t=1"})),
    "git diff --no-prefix": (edit("/p/a.cfg"), bash("git diff --no-prefix a.cfg", "+++ a.cfg\n+t=1")),
    "diff header with a space": (
        edit("/p/my notes.txt"), bash_dict("git diff", "+++ b/my notes.txt\n+x")),
    "closing stdout": (bash("sed -i s/1/2/ a.cfg >&-"), bash("cat a.cfg", "t=2")),
}


def test_round2_leaks_are_rejected():
    for name, steps in LEAKS_2.items():
        assert go(*steps)[0] == "REJECT", name


def test_round2_traps_are_accepted():
    for name, steps in TRAPS_2.items():
        assert go(*steps) == ("ACCEPT", []), name


def test_round2_home_is_the_real_home():
    # ~/.gitignore is the global one, not the project's .gitignore
    assert go(bash("echo '*.log' >> ~/.gitignore"), bash("cat .gitignore", "node_modules"),
              home="/home/u")[0] == "REJECT"
    assert go(bash("echo '*.log' >> ~/.gitignore"),
              bash("cat ~/.gitignore", "*.log"), home="/home/u") == ("ACCEPT", [])


def test_round2_hooks_survive_any_tool_name():
    for name in (5, ["Bash"], {"x": 1}, None):
        event = {"hook_event_name": "PostToolUse", "tool_name": name, "tool_input": {},
                 "tool_response": "", "cwd": "/p"}
        assert drive(GateHooks().post_tool_use(event, None, None)) == {}
        event["hook_event_name"] = "PostToolUseFailure"
        assert drive(GateHooks().post_tool_use_failure(event, None, None)) == {}


def test_round2_pathological_input_is_fast():
    import time
    from grounding_gate import shell
    start = time.perf_counter()
    shell.effects("sed -n '" + " " * 1900 + "q' a.cfg", "/p", "")
    shell.is_read_only("echo " + "$(" * 900)
    assert time.perf_counter() - start < 0.5


# ------------------------------------------------- third review round

def fail(command):
    return ("__fail__", {"command": command})


def go3(*steps, cwd="/p", home="/home/u"):
    gate = GateHooks(home=home)
    for step in steps:
        if step[0] == "__fail__":
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
    return ("ACCEPT" if out == {} else "REJECT"), sorted(gate.state.pending_verification)


A, B_ = "/p/a.cfg", "/p/b.cfg"
COMMIT = "git commit -am \"$(cat <<'EOF'\n{msg}\nEOF\n)\""

LEAKS_3 = {
    "F1 a failing command still owes what it wrote": (
        edit(B_), read(B_), fail("sed -i 's/30/60/' a.cfg && pytest -q"), read(B_, "b=2")),
    "F1b failing redirect then make": (
        edit(B_), read(B_), fail("echo t=60 > a.cfg && make test"), read(B_, "b=2")),
    "F3 a count in the middle of a pipeline": (edit(A), bash_dict("cat a.cfg | wc -l | tr -d ' '", "12")),
    "F3b checksum then cut": (edit(A), bash_dict("cat a.cfg | sha256sum | cut -c1-12", "ab12")),
    "F3c grep -c then tr": (edit(A), bash_dict("grep -c timeout a.cfg | tr -d '\\n'", "1")),
    "F4 jq empty": (edit("/p/a.json"), bash_dict("jq empty a.json && echo ok", "ok")),
    "F4b jq length": (edit("/p/a.json"), bash_dict("jq length a.json", "3")),
    "F4c jq keys": (edit("/p/a.json"), bash_dict("jq -r 'keys[]' a.json", "name")),
    "F5 multi-file grep with no line from a.cfg": (
        edit(A), bash_dict("grep -n timeout a.cfg b.cfg", "b.cfg:3:timeout=1")),
    "F5b grep || echo": (edit(A), bash_dict("grep -n old_key a.cfg || echo none", "none")),
    "F5c untracked file in a multi-file git diff": (
        ("Write", {"file_path": "/p/new.py", "content": "x"}), edit("/p/b.py"),
        bash_dict("git diff -- new.py b.py", "+++ b/b.py\n+x")),
    "F5d identical files print nothing": (edit(A), bash_dict("diff a.cfg a.cfg.orig", "")),
    "F5e cat after || never ran": (edit(B_), read(B_), bash_dict("sed -i s/1/2/ a.cfg || cat a.cfg", "")),
    "F6 diff headers only": (edit(A), bash_dict("git diff | grep '^+++'", "+++ b/a.cfg")),
    "F6b grep prefixes cut down": (edit(A), bash_dict("grep -rn timeout . | cut -d: -f1,2", "./a.cfg:1")),
    "F7 cp overwrites unseen": (
        edit(A), edit(B_), read(A), read(B_), bash("cp /tmp/x.cfg a.cfg"), read(B_, "b=2")),
    "F10 group piped into wc": (edit(A), bash_dict("{ cat a.cfg; cat b.cfg; } | wc -l", "12")),
    "F10b subshell piped into wc": (edit(A), bash_dict("(cat a.cfg; cat b.cfg) | wc -l", "12")),
    "F10c group to /dev/null": (edit(A), bash_dict("{ cat a.cfg; } > /dev/null", "")),
    "F11 git diff between two branches": (
        edit(A), bash_dict("git diff main feature a.cfg", "+++ b/a.cfg\n+t=1")),
    "F11b git diff origin/main HEAD": (
        edit(A), bash_dict("git diff origin/main HEAD", "+++ b/a.cfg\n+t=1")),
    "F12 git blame at a revision": (edit(A), bash_dict("git blame HEAD~3 a.cfg", "abc t=1")),
    "F12b git grep --cached": (edit(A), bash_dict("git grep --cached -n timeout -- a.cfg", "a.cfg:1:t")),
    "F12c git grep on a branch": (edit(A), bash_dict("git grep -n timeout main -- a.cfg", "main:a.cfg:1:t")),
    "F13 sed -e ... -i .env": (
        edit(B_), read(B_), bash("sed -e 's/x/y/' -i .env"), read(B_, "b=3")),
    "F13b sed -i --expression=": (
        edit(B_), read(B_), bash("sed -i --expression='s/x/y/' a.cfg"), read(B_, "b=3")),
    "F14 awk -iinplace": (edit(A), bash_dict("awk -iinplace '{sub(/30/,\"60\")}1' a.cfg", "")),
    "F14b awk print NF": (edit(A), bash_dict("awk '{print NF}' a.cfg", "2")),
    "F15 substitution in a heredoc body": (
        edit(A), read(A), bash_dict("cat <<EOF\n$(sed -i s/x/y/ a.cfg)\nEOF", "")),
    "F16 rm -rf /tmp/cache is not /p/tmp/cache": (
        ("Write", {"file_path": "/p/tmp/cache/x.json", "content": "{}"}), edit(B_),
        bash("rm -rf /tmp/cache"), read(B_, "b=1")),
    "F17 background edit races the read": (edit(A), bash_dict("sed -i s/1/2/ a.cfg & cat a.cfg", "t=1")),
    "F17b cd that may have failed": (edit("/p/sub/a.cfg"), bash_dict("cd sub; cat a.cfg", "t=1")),
}

TRAPS_3 = {
    "F2 commit message with a quoted phrase": (
        edit(A), read(A), bash(COMMIT.format(msg='Use "->" for returns')), read(A, "t=2")),
    "F2b commit message with >=": (
        edit(A), read(A), bash(COMMIT.format(msg='Support "x >= 2" check')), read(A, "t=2")),
    "F7b cp creates a file": (bash("cp .env.example .env"), bash_dict("cat .env", "K=1")),
    "F7c cp restores a file": (bash("cp a.cfg.bak a.cfg"), read(A, "t=1")),
    "F8 glob move into a directory": (edit(A), bash("mv *.cfg archive/"), read("/p/archive/a.cfg")),
    "F9 failed Edit of a missing file": (
        ("__fail_edit__", "Edit", {"file_path": "/p/src/config.py"}),
        edit("/p/config.py"), read("/p/config.py")),
}


def test_round3_leaks_are_rejected():
    for name, steps in LEAKS_3.items():
        assert go3(*steps)[0] == "REJECT", name


def test_round3_traps_are_accepted():
    for name, steps in TRAPS_3.items():
        assert go3(*steps) == ("ACCEPT", []), name


def test_round3_long_sessions_stay_fast():
    import time
    gate = GateHooks(home="/home/u")
    start = time.perf_counter()
    for i in range(1500):
        drive(gate.post_tool_use(ptu("Write", {"file_path": "/p/f%d.cfg" % i}), None, None))
        drive(gate.post_tool_use(ptu("Bash", {"command": "cat f%d.cfg" % i}, "x"), None, None))
    assert time.perf_counter() - start < 3.0


def test_T8_git_diff_headers_are_repo_relative():
    verdict = go(edit("/repo/pkg/m.py"),
                 bash("git diff", "--- a/pkg/m.py\n+++ b/pkg/m.py\n+x=2"), cwd="/repo/pkg")
    assert verdict == ("ACCEPT", [])


def test_L9_failed_rm_keeps_the_debt():
    gate = GateHooks()
    for path in ("/p/a.cfg", "/p/b.cfg"):
        drive(gate.post_tool_use(ptu("Edit", {"file_path": path}), None, None))
    drive(gate.post_tool_use_failure(
        {"hook_event_name": "PostToolUseFailure", "tool_name": "Bash",
         "tool_input": {"command": "rm a.cfg"}, "error": "Permission denied",
         "cwd": "/p"}, None, None))
    drive(gate.post_tool_use(ptu("Read", {"file_path": "/p/b.cfg"}, "b=1"), None, None))
    assert drive(gate.stop(STOP, None, None))["decision"] == "block"


def test_T1b_deleted_temp_file_is_never_owed():
    # adjudicated: the reviewer's trap was the deleted req.json staying owed
    # forever, which is fixed. Reading an unrelated pkg.json still doesn't
    # verify what `curl` did, so the finish stays blocked by design.
    verdict, owed = go(
        bash("echo '{}' > req.json && curl -d @req.json localhost && rm req.json"),
        read("/p/pkg.json", '{"x":1}'))
    assert owed == [] and verdict == "REJECT"


def test_T12_a_later_turn_without_edits_is_a_question():
    # adjudicated: turn 2 is now an assertion (not a completion). It still
    # needs its answer's subject on the claim surface, as the README says.
    for surface, want in ((set(), "block"), ({"docs/usage.md"}, None)):
        gate = GateHooks(claim_surface=surface)
        drive(gate.post_tool_use(ptu("Edit", {"file_path": "/p/a.cfg"}), None, None))
        drive(gate.post_tool_use(ptu("Read", {"file_path": "/p/a.cfg"}, "t=30"), None, None))
        assert drive(gate.stop(STOP, None, None)) == {}
        drive(gate.user_prompt_submit(PROMPT, None, None))
        drive(gate.post_tool_use(ptu("Read", {"file_path": "/p/docs/usage.md"}, "Usage"),
                                 None, None))
        assert drive(gate.stop(STOP, None, None)).get("decision") == want


def test_C1_missing_tool_name_does_not_crash():
    event = {"hook_event_name": "PostToolUse", "tool_name": None, "tool_input": {},
             "tool_response": "", "cwd": "/p"}
    assert drive(GateHooks().post_tool_use(event, None, None)) == {}


if __name__ == "__main__":
    failures = []
    cases = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    for name, fn in cases:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as exc:
            print(f"  FAIL  {name}: {exc}")
            failures.append(name)
    print(f"\n{'ALL PASS' if not failures else f'FAILED: {failures}'}"
          f" — {len(cases) - len(failures)}/{len(cases)}")
    sys.exit(1 if failures else 0)
