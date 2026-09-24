"""What a shell command reads, shows, writes, moves and removes.

The gate never runs a command; it reads the command text the agent ran and
the output the agent saw. This module turns that text into stages (one per
simple command) with a small quote- and heredoc-aware lexer, then answers
the questions the gate asks:

* ``is_read_only(command)``: can this command only look, never change?
* ``read_operands(command, output, cwd)``: which files did the agent see the
  content of, and which did it only see listed?
* ``effects(command, cwd)``: in order, what the command wrote, moved,
  removed, and showed (for mutating commands).

Everything here is conservative in the same direction as the rest of the
gate: a command it can't vouch for is a possible write, and a file it can't
place gets no credit. Paths come back as whole strings (never split into
word tokens), resolved against ``cwd`` and any ``cd`` in the command.
``~/x`` resolves to the root-less ``x`` (the home directory is unknown), which
the gate's path matching treats like any other relative tail.
"""

import posixpath
import re

# ------------------------------------------------------------------ lexer

_OPERATORS = ("&&", "||", "|&", ";;", "|", ";", "&", "(", ")")
_REDIRECTS = ("&>>", "&>", ">>", ">|", ">&", "<<<", "<<-", "<<", "<&", "<>", ">", "<")


class _Word(str):
    """A shell word with quotes removed. ``subst`` marks a command or
    process substitution inside it (its output is unknown)."""
    subst = False


def _read_balanced(text, i):
    """Index just past the ``)`` that closes the ``(`` at ``text[i-1]``."""
    depth, quote = 1, None
    while i < len(text):
        c = text[i]
        if quote:
            if c == quote:
                quote = None
            elif c == "\\" and quote == '"':
                i += 1
        elif c in "'\"":
            quote = c
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise ValueError("unbalanced substitution")


def tokenize(command):
    """``[("word", _Word) | ("op", str) | ("redir", (fd, op))]``.

    Quotes are removed from words; heredoc bodies are skipped (they are data
    fed to a command, not commands); ``#`` comments are dropped; newlines are
    ``;``. Raises ValueError on an unterminated quote.
    """
    tokens, word, in_word, subst = [], [], False, False
    heredocs = []          # (delimiter, strip_tabs) awaiting the next newline
    i, n = 0, len(command)

    def end_word():
        nonlocal word, in_word, subst
        if in_word:
            w = _Word("".join(word))
            w.subst = subst
            tokens.append(("word", w))
        word, in_word, subst = [], False, False

    while i < n:
        c = command[i]
        if c == "\n":
            end_word()
            tokens.append(("op", ";"))
            i += 1
            while heredocs:
                delim, strip_tabs = heredocs.pop(0)
                while i < n:
                    j = command.find("\n", i)
                    line = command[i:] if j < 0 else command[i:j]
                    i = n if j < 0 else j + 1
                    if (line.lstrip("\t") if strip_tabs else line) == delim:
                        break
            continue
        if c in " \t":
            end_word()
            i += 1
            continue
        if c == "#" and not in_word:
            while i < n and command[i] != "\n":
                i += 1
            continue
        if c == "\\":
            if i + 1 < n and command[i + 1] == "\n":
                i += 2
                continue
            word.append(command[i + 1] if i + 1 < n else "")
            in_word = True
            i += 2
            continue
        if c == "'":
            j = command.find("'", i + 1)
            if j < 0:
                raise ValueError("unterminated single quote")
            word.append(command[i + 1:j])
            in_word = True
            i = j + 1
            continue
        if c == '"':
            i += 1
            while True:
                if i >= n:
                    raise ValueError("unterminated double quote")
                d = command[i]
                if d == '"':
                    i += 1
                    break
                if d == "\\" and i + 1 < n and command[i + 1] in '"\\$`\n':
                    word.append(command[i + 1])
                    i += 2
                    continue
                if d == "`" or command.startswith("$(", i):
                    subst = True
                word.append(d)
                i += 1
            in_word = True
            continue
        if c == "`":
            j = command.find("`", i + 1)
            if j < 0:
                raise ValueError("unterminated backtick")
            word.append(command[i:j + 1])
            in_word, subst = True, True
            i = j + 1
            continue
        if command.startswith("$(", i):
            j = _read_balanced(command, i + 2)
            word.append(command[i:j])
            in_word, subst = True, True
            i = j
            continue
        if command.startswith("<(", i) or command.startswith(">(", i):
            end_word()
            j = _read_balanced(command, i + 2)
            w = _Word(command[i:j])
            w.subst = True
            tokens.append(("word", w))
            i = j
            continue
        redir = next((r for r in _REDIRECTS if command.startswith(r, i)), None)
        if redir:
            fd = ""
            if in_word and "".join(word).isdigit() and redir[0] in "<>":
                fd, word, in_word = "".join(word), [], False
            end_word()
            tokens.append(("redir", (fd, redir)))
            i += len(redir)
            if redir in ("<<", "<<-"):
                # the delimiter word follows; its body starts at the next newline
                while i < n and command[i] in " \t":
                    i += 1
                j = i
                while j < n and command[j] not in " \t\n;&|<>()":
                    j += 1
                delim = command[i:j].replace("'", "").replace('"', "").replace("\\", "")
                heredocs.append((delim, redir == "<<-"))
                tokens.append(("word", _Word(delim)))
                i = j
            continue
        op = next((o for o in _OPERATORS if command.startswith(o, i)), None)
        if op:
            end_word()
            tokens.append(("op", op))
            i += len(op)
            continue
        word.append(c)
        in_word = True
        i += 1
    end_word()
    return tokens


# ----------------------------------------------------------------- stages

class Stage:
    """One simple command: ``argv`` (env assignments and ``sudo``/``time``
    style prefixes removed), its redirects as ``(fd, op, target)``, whether
    anything in it is a substitution, whether its stdout is piped into the
    next stage, and the directory relative paths resolve against."""

    def __init__(self, argv, redirects, subst, piped, cwd):
        self.argv, self.redirects = argv, redirects
        self.subst, self.piped, self.cwd = subst, piped, cwd

    def resolve(self, path):
        return resolve(path, self.cwd)

    @property
    def stdout_redirected(self):
        """Stdout goes to a file or /dev/null, not to the agent."""
        return any(op in (">", ">>", ">|", "&>", "&>>") and fd in ("", "1")
                   or (op == ">&" and fd in ("", "1") and not target.isdigit())
                   for fd, op, target in self.redirects)


_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_PREFIXES = frozenset({"sudo", "time", "command", "nohup", "env", "builtin"})


def resolve(path, cwd):
    """Place ``path`` relative to ``cwd``. ``""`` means an unknown root
    (paths stay relative, matched by their tail); ``None`` means unknown."""
    if not path or set(path) & set("$`") or path.startswith("~") and not (
            path == "~" or path.startswith("~/")):
        return None
    if path.startswith("~"):
        rest = path[2:]            # home is unknown: keep the relative tail
        return posixpath.normpath(rest) if rest else None
    if path.startswith("/"):
        return posixpath.normpath(path)
    if cwd is None:
        return None
    return posixpath.normpath(posixpath.join(cwd, path) if cwd else path)


def parse(command, cwd=""):
    """``[Stage]`` for ``command``, or None when it can't be tokenized.

    ``cd`` updates the base for later stages (``None`` once it goes somewhere
    unknown, such as ``cd -`` or ``cd $DIR``); a subshell's ``cd`` ends with
    its closing parenthesis.
    """
    try:
        tokens = tokenize(command)
    except ValueError:
        return None
    base = posixpath.normpath(cwd) if cwd else ""
    stages, saved = [], []
    argv, redirects, subst = [], [], False
    expect_target = None

    def flush(piped):
        nonlocal argv, redirects, subst, base
        words = argv
        while words and (_ASSIGNMENT.match(words[0]) or words[0] in _PREFIXES
                         or words[0] in ("{", "}")):
            words = words[1:]
        words = [w for w in words if w != "}"]
        if words or redirects:
            stages.append(Stage(list(words), redirects, subst, piped, base))
            if words and words[0] == "cd" and not piped:
                target = words[1] if len(words) > 1 else "~"
                base = None if target in ("-", "~") or target.startswith("~") \
                    else resolve(target, base)
        argv, redirects, subst = [], [], False

    for kind, value in tokens:
        if expect_target is not None:
            fd, op = expect_target
            expect_target = None
            if kind == "word":
                redirects.append((fd, op, str(value)))
                subst = subst or value.subst
                continue
        if kind == "word":
            argv.append(value)
            subst = subst or value.subst
        elif kind == "redir":
            expect_target = value
        elif value in ("|", "|&"):
            flush(piped=True)
        elif value == "(":
            flush(piped=False)
            saved.append(base)
        elif value == ")":
            flush(piped=False)
            if saved:
                base = saved.pop()
        else:                                    # ; && || & ;;
            flush(piped=False)
    flush(piped=False)
    return stages


# ------------------------------------------------------- program knowledge

_READ_ONLY = frozenset({
    "cat", "head", "tail", "grep", "egrep", "fgrep", "rg", "ls", "wc", "diff",
    "cmp", "stat", "file", "nl", "cut", "tr", "jq", "tree", "du", "df", "pwd",
    "cd", "echo", "printf", "which", "md5sum", "sha256sum", "true", ":",
    "test", "[", "basename", "dirname", "realpath", "readlink", "sort"})
# print file lines through to their output
_PASS_THROUGH = frozenset({"cat", "head", "tail", "nl", "cut", "tr", "sort",
                           "grep", "egrep", "fgrep", "rg", "jq", "sed", "awk",
                           "json.tool"})
_CONTENT = frozenset({"cat", "head", "tail", "nl", "cut", "sort", "diff",
                      "json.tool", "sed", "awk", "git diff", "git blame"})
_PATTERN = frozenset({"grep", "egrep", "fgrep", "rg", "jq", "git grep"})
_LISTING = frozenset({"ls", "stat", "file", "wc", "du", "tree", "cmp", "md5sum",
                      "sha256sum", "test", "[", "git status", "git log",
                      "git ls-files"})
_READ_ONLY_GIT = frozenset({"diff", "show", "log", "status", "blame", "ls-files",
                            "grep", "rev-parse"})
_GIT_GLOBAL_WITH_VALUE = frozenset({"-C", "-c", "--git-dir", "--work-tree",
                                    "--namespace"})
# grep/rg flags whose output is a count, a file list, or nothing
_QUIET_LONG = frozenset({"--quiet", "--silent", "--count", "--count-matches",
                         "--files-with-matches", "--files-without-match",
                         "--files", "--json"})
_QUIET_SHORT = frozenset("qclL")
_DIFF_SUMMARY = frozenset({"--stat", "--numstat", "--shortstat", "--name-only",
                           "--name-status", "--summary", "--quiet",
                           "--exit-code", "--dirstat", "-q", "--brief", "-s",
                           "--report-identical-files", "--no-patch"})
_SED_ADDR = r"(?:\d+|\$|/[^/]*/)"
_SED_PRINT_ONLY = re.compile(
    r"^\s*(?:%s(?:\s*,\s*%s)?)?\s*!?\s*p\s*$" % (_SED_ADDR, _SED_ADDR))
_AWK_UNSAFE = ("system", "getline", "|", ">", "close", "fflush", "BEGIN", "END")
_PYTHON = re.compile(r"^python(?:3(?:\.\d+)?)?$")


def _program(stage):
    """``(name, args, cwd)``; git subcommands are named ``git diff`` etc.,
    with global options skipped (``-C DIR`` moves the base)."""
    argv, cwd = stage.argv, stage.cwd
    if not argv:
        return "", [], cwd
    if argv[0] == "git":
        k = 1
        while k < len(argv) and argv[k].startswith("-"):
            opt = argv[k].split("=", 1)[0]
            if argv[k] == "-C" and k + 1 < len(argv):
                cwd = resolve(argv[k + 1], cwd)
            if opt in _GIT_GLOBAL_WITH_VALUE and "=" not in argv[k]:
                k += 1
            k += 1
        if k < len(argv):
            return "git " + argv[k], argv[k + 1:], cwd
        return "git", [], cwd
    if (len(argv) >= 3 and _PYTHON.match(argv[0])
            and argv[1:3] == ["-m", "json.tool"]):
        return "json.tool", argv[3:], cwd
    return argv[0], argv[1:], cwd


def _operands(args):
    return [a for a in args if not a.startswith("-") or a == "-"]


def _sed_parts(args):
    """``(in_place, scripts, files)`` for a sed argument list."""
    in_place, scripts, files, has_e, k = False, [], [], False, 0
    while k < len(args):
        a = args[k]
        if a in ("-e", "--expression"):
            has_e = True
            if k + 1 < len(args):
                scripts.append(args[k + 1])
            k += 2
            continue
        if a in ("-f", "--file"):
            scripts.append(None)          # a script file: unknown commands
            has_e = True
            k += 2
            continue
        if a == "-i" or a.startswith("-i") or a.startswith("--in-place"):
            in_place = True
            if a == "-i" and k + 1 < len(args) and args[k + 1] == "":
                k += 1                    # BSD/macOS `sed -i '' ...`
            k += 1
            continue
        if a.startswith("-") and a != "-":
            k += 1
            continue
        files.append(a)
        k += 1
    if not has_e and files:
        scripts.append(files.pop(0))
    return in_place, scripts, files


def _is_viewer(name, args):
    """sed/awk/json.tool used only to print file content."""
    if name == "sed":
        in_place, scripts, _ = _sed_parts(args)
        return (not in_place and bool(scripts)
                and all(s is not None and _SED_PRINT_ONLY.match(s) for s in scripts))
    if name == "awk":
        if "-f" in args:
            return False
        ops = _operands(args)
        return bool(ops) and not any(t in ops[0] for t in _AWK_UNSAFE)
    if name == "json.tool":
        return len(_operands(args)) <= 1
    return False


def _shows_content(name, args):
    """Whether the program prints file lines (not a count, list or verdict)."""
    if name in _PATTERN:
        return not any(a in _QUIET_LONG or (
            a.startswith("-") and not a.startswith("--")
            and set(a[1:]) & _QUIET_SHORT) for a in args)
    if name in ("diff", "git diff"):
        return not any(a.split("=", 1)[0] in _DIFF_SUMMARY for a in args)
    if name in ("sed", "awk", "json.tool"):
        return _is_viewer(name, args)
    return name in _PASS_THROUGH or name == "git blame"


def _stage_read_only(stage):
    if stage.subst:
        return False
    for fd, op, target in stage.redirects:
        if op in ("<", "<<", "<<-", "<<<", "<&"):
            continue
        if op in (">&", "<&") and target.isdigit():
            continue
        if target == "/dev/null":
            continue
        return False
    name, args, _ = _program(stage)
    if not name:
        return True
    if name.startswith("git "):
        return (name[4:] in _READ_ONLY_GIT
                and not any(a.startswith("--output") for a in args))
    if name in ("sed", "awk", "json.tool"):
        return _is_viewer(name, args)
    if name == "sort":
        return not any(a in ("-o", "--output") or a.startswith("--output=")
                       for a in args)
    return name in _READ_ONLY


def is_read_only(command):
    """True when every stage is a known read-only program with no output
    redirect to a file and no substitution. Anything else may write."""
    stages = parse(command)
    return bool(stages) and all(_stage_read_only(s) for s in stages)


# ------------------------------------------------------------------ reads

_DIFF_NEW_FILE = re.compile(r"^\+\+\+ b/(\S+)", re.M)
_OUTPUT_PATH = re.compile(r"^([^\s:]+):", re.M)


def _prints_filenames(name, args):
    """grep/rg prefix each output line with its file (recursive search,
    several files, or -H); a single-file or stdin search does not."""
    if any(a in ("-h", "--no-filename") for a in args):
        return False
    if name == "git grep" or any(a in ("-H", "--with-filename") for a in args):
        return True
    recursive = any(a in ("-r", "-R", "--recursive") or (
        a.startswith("-") and not a.startswith("--") and set(a[1:]) & set("rR"))
        for a in args)
    ops = _operands(args)
    files = ops[1:] if not any(a in ("-e", "-f", "--regexp") or
                               a.startswith("--regexp=") for a in args) else ops
    if name == "rg":
        return len(files) != 1 or recursive
    return recursive or len(files) > 1


def _stage_reads(stage, output):
    """``(content, listed)`` paths one read-only stage looked at."""
    name, args, cwd = _program(stage)
    content, listed = [], []
    ops = _operands(args)
    if name in _PATTERN:
        if not any(a in ("-e", "-f", "--regexp") or a.startswith("--regexp=")
                   for a in args):
            ops = ops[1:]                      # the first operand is the pattern
        content = ops
        if _prints_filenames(name, args):
            content = content + _OUTPUT_PATH.findall(output or "")
    elif name == "sed":
        content = _sed_parts(args)[2]
    elif name == "awk":
        content = ops[1:]
    elif name == "git diff":
        # an untracked or unchanged file prints nothing: nothing was seen
        if (output or "").strip():
            content = [o for o in ops if o != "--"]
            # headers are repo-relative: keep them root-less (tail-matched)
            content += ["\0" + p for p in _DIFF_NEW_FILE.findall(output)]
    elif name == "git show":
        content = []                           # committed copies, never the tree
    elif name in _CONTENT:
        content = [o for o in ops if o != "-"]
    elif name in _LISTING:
        listed = [o for o in ops if o != "-"]
    placed = []
    for group in (content, listed):
        out = set()
        for p in group:
            if p.startswith("\0"):
                out.add(posixpath.normpath(p[1:]))
            elif ":" not in p or p.startswith("/") or p.startswith("."):
                r = resolve(p, cwd)
                if r:
                    out.add(r)
            else:                              # REV:path, host:path
                continue
        placed.append(out)
    return placed[0], placed[1]


def read_operands(command, output="", cwd=""):
    """``(content, listed)``: paths a read-only command showed the content
    of, and paths it only listed or summarized.

    A pipeline counts as showing content only if its last stage passes lines
    through to the agent (``cat a | grep x`` does; ``cat a | wc -l``,
    ``grep -q``, ``git diff --stat`` and ``> /dev/null`` do not); what it
    read is then demoted to listed.
    """
    stages = parse(command, cwd) or []
    content, listed = set(), set()
    pipe_content, pipe_listed = set(), set()
    for stage in stages:
        c, l_ = _stage_reads(stage, output)
        pipe_content |= c
        pipe_listed |= l_
        if stage.piped:
            continue
        name, args, _ = _program(stage)
        if _shows_content(name, args) and not stage.stdout_redirected:
            content |= pipe_content
        else:
            listed |= pipe_content
        listed |= pipe_listed
        pipe_content, pipe_listed = set(), set()
    return content, listed


# ---------------------------------------------------------------- effects

def _writes(stage):
    """Files a stage writes: stdout/both redirects (stderr logs are not the
    edit), ``tee`` operands, ``sed -i`` files."""
    name, args, cwd = _program(stage)
    out = []
    for fd, op, target in stage.redirects:
        if op in (">", ">>", ">|", "&>", "&>>") and fd in ("", "1"):
            out.append(target)
        elif op == ">&" and fd in ("", "1") and not target.isdigit():
            out.append(target)
    if name == "tee":
        out += _operands(args)
    elif name == "sed":
        in_place, _, files = _sed_parts(args)
        if in_place:
            out += files
    placed = []
    for p in out:
        if p != "/dev/null" and not set(p) & set("*?["):
            r = resolve(p, cwd)
            if r:
                placed.append(r)
    return placed


def _moves(stage):
    """``(src, dst, alt)`` per moved file; ``alt`` is the other reading of
    an ambiguous ``mv a b`` (b may be an existing directory)."""
    name, args, cwd = _program(stage)
    if name == "mv":
        args = list(args)
    elif name == "git mv":
        pass
    else:
        return []
    target_dir, ops, k = None, [], 0
    while k < len(args):
        a = args[k]
        if a in ("-t", "--target-directory") and k + 1 < len(args):
            target_dir = args[k + 1]
            k += 2
            continue
        if a.startswith("--target-directory="):
            target_dir = a.split("=", 1)[1]
        elif not a.startswith("-"):
            ops.append(a)
        k += 1
    if target_dir is None:
        if len(ops) < 2:
            return []
        *sources, target = ops
    else:
        sources, target = ops, target_dir
    into_dir = target_dir is not None or len(sources) > 1 or target.endswith("/")
    moves = []
    for s in sources:
        if set(s) & set("*?["):
            continue
        base = posixpath.basename(s.rstrip("/"))
        inside = posixpath.join(target, base)
        dst, alt = (inside, None) if into_dir else (target, inside)
        src, dst = resolve(s, cwd), resolve(dst, cwd)
        alt = resolve(alt, cwd) if alt else None
        if src and dst:
            moves.append((src, dst, alt))
    return moves


def _removes(stage):
    name, args, cwd = _program(stage)
    if name not in ("rm", "git rm", "unlink", "rmdir"):
        return []
    out = []
    for o in _operands(args):
        r = resolve(o, cwd)
        if r:
            out.append(r)
    return out


def effects(command, cwd="", output=""):
    """Ordered effects of a (possibly mutating) command:
    ``("write", path)``, ``("move", src, dst, alt)``, ``("remove", path)``,
    ``("read", path)`` for content a stage showed the agent, and
    ``("opaque",)`` for a stage whose writes can't be known (a script, a
    build, a substitution). None when the command can't be parsed."""
    stages = parse(command, cwd)
    if stages is None:
        return None
    out, pipe_reads = [], []
    for stage in stages:
        name, args, _ = _program(stage)
        known = _stage_read_only(stage) or name in (
            "tee", "sed", "mv", "git mv", "rm", "git rm", "unlink", "rmdir",
            "mkdir", "touch", "cp")
        if stage.subst or not known:
            out.append(("opaque",))
        out += [("write", p) for p in _writes(stage)]
        out += [("move",) + m for m in _moves(stage)]
        out += [("remove", p) for p in _removes(stage)]
        if _stage_read_only(stage):
            pipe_reads += sorted(_stage_reads(stage, output)[0])
        if stage.piped:
            continue
        if _shows_content(name, args) and not stage.stdout_redirected:
            out += [("read", p) for p in pipe_reads]
        pipe_reads = []
    return out
