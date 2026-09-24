"""What a shell command reads, shows, writes, moves and removes.

The gate never runs a command; it reads the command text the agent ran and
the output the agent saw. This module turns that text into stages (one per
simple command) with a small quote- and heredoc-aware lexer, then answers
the questions the gate asks:

* ``is_read_only(command)``: can this command only look, never change?
* ``read_operands(command, output, cwd, home)``: which files did the agent
  see the content of, and which did it only see listed?
* ``effects(command, cwd, output, home)``: in order, what the command wrote,
  moved, removed, and showed (for mutating commands).

The rule throughout is to fail toward "unknown", never toward a guess:
a command it can't fully account for is a possible write, a file it can't
place gets no credit, output it can't attribute to one producer credits
nothing, and a write target it can't place is never owed (the completion
then falls back to the plain freshness check). Paths come back whole,
resolved against ``cwd``, ``home``, and any ``cd``/``pushd`` in the command.
"""

import fnmatch  # noqa: F401  (re-exported for state's glob matching)
import posixpath
import re

# ------------------------------------------------------------------ lexer

_OPERATORS = ("&&", "||", "|&", ";;", "|", ";", "&", "(", ")")
_REDIRECTS = ("&>>", "&>", ">>", ">|", ">&", "<<<", "<<-", "<<", "<&", "<>", ">", "<")
_MAX_COMMAND = 100_000          # longer than this: not analysed (unknown)


class _Word(str):
    """A shell word with quotes removed. ``subst`` marks a command, process
    or arithmetic substitution inside it (its effects are unknown)."""
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
    raise ValueError("unbalanced parenthesis")


def _read_word(text, i):
    """``(word, end, quoted)`` for a heredoc delimiter starting at ``i``:
    quotes are honoured (``'END X'`` is one word) and removed."""
    out, quoted = [], False
    n = len(text)
    while i < n and text[i] not in " \t\n;&|<>()":
        c = text[i]
        if c in "'\"":
            j = text.find(c, i + 1)
            if j < 0:
                raise ValueError("unterminated quote in heredoc delimiter")
            out.append(text[i + 1:j])
            quoted = True
            i = j + 1
        elif c == "\\" and i + 1 < n:
            out.append(text[i + 1])
            quoted = True
            i += 2
        else:
            out.append(c)
            i += 1
    return "".join(out), i, quoted


def tokenize(command):
    """``[("word", _Word) | ("op", str) | ("redir", (fd, op))]``.

    Quotes are removed from words (``'…'``, ``"…"``, ``$'…'``); heredoc
    bodies are skipped (they are data fed to a command, not commands);
    ``#`` comments are dropped; newlines are ``;``; ``((…))`` arithmetic is
    one opaque word. Raises ValueError on anything unterminated.
    """
    if len(command) > _MAX_COMMAND:
        raise ValueError("command too long to analyse")
    tokens, word, in_word, subst = [], [], False, False
    heredocs = []          # (delimiter, strip_tabs, quoted) awaiting a newline
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
                delim, strip_tabs, quoted = heredocs.pop(0)
                while True:
                    if i >= n:
                        raise ValueError("unterminated heredoc")
                    j = command.find("\n", i)
                    line = command[i:] if j < 0 else command[i:j]
                    i = n if j < 0 else j + 1
                    if (line.lstrip("\t") if strip_tabs else line) == delim:
                        break
                    if not quoted and ("$(" in line or "`" in line):
                        # an unquoted heredoc body runs its substitutions
                        raise ValueError("substitution in heredoc body")
            continue
        if c in " \t\r":
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
        if command.startswith("$'", i):
            i += 2
            while True:
                if i >= n:
                    raise ValueError("unterminated $'' quote")
                if command[i] == "\\" and i + 1 < n:
                    word.append(command[i:i + 2])
                    i += 2
                    continue
                if command[i] == "'":
                    i += 1
                    break
                word.append(command[i])
                i += 1
            in_word = True
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
                if command.startswith("$(", i):
                    # quotes inside $(...) don't end the outer string
                    j = _read_balanced(command, i + 2)
                    word.append(command[i:j])
                    subst = True
                    i = j
                    continue
                if d == "`":
                    j = command.find("`", i + 1)
                    if j < 0:
                        raise ValueError("unterminated backtick")
                    word.append(command[i:j + 1])
                    subst = True
                    i = j + 1
                    continue
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
        if not in_word and command.startswith("((", i):
            end_word()
            j = command.find("))", i + 2)
            if j < 0:
                raise ValueError("unterminated arithmetic")
            w = _Word(command[i:j + 2])
            w.subst = True
            tokens.append(("word", w))
            i = j + 2
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
                while i < n and command[i] in " \t":
                    i += 1
                delim, i, quoted = _read_word(command, i)
                heredocs.append((delim, redir == "<<-", quoted))
                tokens.append(("word", _Word(delim)))
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
    if heredocs:
        raise ValueError("unterminated heredoc")
    return tokens


# ----------------------------------------------------------------- stages

class Stages(list):
    """The stages of a command. ``grouped_output`` is True when a group's
    output (``{ ...; }``, ``( ... )``, ``if ... fi``) is piped or redirected
    as a whole: what its stages printed never reached the agent directly."""
    grouped_output = False


class Stage:
    """One simple command: ``argv`` (env assignments, keywords and
    ``sudo``/``time`` style prefixes removed), its redirects as
    ``(fd, op, target)``, whether anything in it is a substitution, whether
    its stdout is piped into the next stage / its stdin piped from the
    previous one, and the directory and home relative paths resolve
    against."""

    def __init__(self, argv, redirects, subst, piped, piped_in, cwd, home, sep=";"):
        self.argv, self.redirects, self.subst = argv, redirects, subst
        self.piped, self.piped_in = piped, piped_in
        self.cwd, self.home, self.sep = cwd, home, sep

    def resolve(self, path):
        return resolve(path, self.cwd, self.home)

    @property
    def stdout_redirected(self):
        """Stdout goes to a file, /dev/null, or is closed: not to the agent."""
        for fd, op, target in self.redirects:
            if fd not in ("", "1"):
                continue
            if op in (">", ">>", ">|", "&>", "&>>"):
                return True
            if op == ">&" and not target.isdigit():
                return True
        return False


_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_PREFIXES = frozenset({"sudo", "time", "command", "nohup", "env", "builtin",
                       "if", "then", "else", "elif", "do", "while", "until",
                       "!", "{"})
_CLOSERS = frozenset({"}", "fi", "done", "esac"})


def resolve(path, cwd, home=None):
    """Place ``path``. ``cwd`` ``""`` means an unknown root (paths stay
    relative, matched by their tail); ``None`` means unknown (no answer).
    ``~/x`` needs ``home``; without it the path can't be placed."""
    if not path or set(path) & set("$`") or path.startswith("-"):
        return None
    if path.startswith("<(") or path.startswith(">("):
        return None
    if path == "~" or path.startswith("~/"):
        if not home:
            return None
        return posixpath.normpath(posixpath.join(home, path[2:]))
    if path.startswith("~"):
        return None                            # ~user: someone else's home
    if path.startswith("/"):
        return posixpath.normpath(path)
    if cwd is None:
        return None
    return posixpath.normpath(posixpath.join(cwd, path) if cwd else path)


def _cd_target(words):
    """The directory a ``cd``/``pushd`` argument list names (options such
    as ``-P`` skipped), or ``None`` when it goes somewhere unknowable."""
    args = [w for w in words[1:] if not (w.startswith("-") and w != "-")]
    if not args:
        return "~"
    return args[0]


def parse(command, cwd="", home=None):
    """``[Stage]`` for ``command``, or None when it can't be tokenized.

    ``cd`` and ``pushd``/``popd`` update the base for later stages (``None``
    once it goes somewhere unknown, such as ``cd -`` or ``cd $DIR``); a
    subshell's ``cd`` ends with its closing parenthesis, and a ``cd`` in a
    pipeline or in the background (``cd x &``) changes nothing.
    """
    try:
        tokens = tokenize(command)
    except ValueError:
        return None
    base = posixpath.normpath(cwd) if cwd else ("" if cwd == "" else None)
    stages, saved, dirstack = Stages(), [], []
    argv, redirects, subst = [], [], False
    expect_target, piped_in, just_closed = None, False, False

    def flush(sep):
        nonlocal argv, redirects, subst, base, piped_in
        piped = sep in ("|", "|&")
        if argv and argv[0] in _CLOSERS and (piped or redirects):
            stages.grouped_output = True
        words = argv
        while words and (_ASSIGNMENT.match(words[0]) or words[0] in _PREFIXES):
            words = words[1:]
        words = [w for w in words if w not in _CLOSERS]
        if words or redirects or subst:
            stages.append(Stage(list(words), redirects, subst, piped, piped_in,
                                base, home, sep))
            name = words[0] if words else ""
            if name in ("cd", "pushd", "popd") and not piped and sep != "&" \
                    and not piped_in:
                if name == "popd":
                    base = dirstack.pop() if dirstack else None
                else:
                    target = _cd_target(words)
                    if name == "pushd":
                        dirstack.append(base)
                    # after `cd x;` the next command runs even if cd failed,
                    # so only `cd x && ...` pins where it runs
                    base = None if target == "-" or sep != "&&" \
                        else resolve(target, base, home)
        piped_in = piped
        argv, redirects, subst = [], [], False

    for kind, value in tokens:
        if just_closed and (kind == "redir" or value in ("|", "|&")):
            stages.grouped_output = True
        just_closed = False
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
        elif value == "(":
            flush(";")
            saved.append(base)
        elif value == ")":
            flush(";")
            if saved:
                base = saved.pop()
            just_closed = True
        else:                                    # | |& ; && || & ;;
            flush(value)
    flush(";")
    return stages


# ------------------------------------------------------- program knowledge

_READ_ONLY = frozenset({
    "cat", "head", "tail", "grep", "egrep", "fgrep", "rg", "ls", "wc", "diff",
    "cmp", "stat", "file", "nl", "cut", "tr", "jq", "tree", "du", "df", "pwd",
    "cd", "pushd", "popd", "echo", "printf", "which", "md5sum", "sha256sum",
    "true", ":", "test", "[", "basename", "dirname", "realpath", "readlink",
    "sort"})
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
# options that take a separate value, per program (so the value is never
# mistaken for a pattern or a file)
_VALUE_OPTS = {
    "grep": {"-e", "-f", "-A", "-B", "-C", "-m", "-d", "-D", "--regexp",
             "--file", "--after-context", "--before-context", "--context",
             "--max-count", "--directories", "--devices", "--label",
             "--include", "--exclude", "--exclude-dir", "--exclude-from"},
    "rg": {"-e", "-f", "-g", "-t", "-T", "-A", "-B", "-C", "-m", "-M", "-j",
           "-r", "-E", "--regexp", "--file", "--glob", "--iglob", "--type",
           "--type-not", "--type-add", "--after-context", "--before-context",
           "--context", "--max-count", "--max-columns", "--threads",
           "--replace", "--encoding", "--pre", "--sort", "--sortr",
           "--max-depth", "--max-filesize", "--context-separator"},
    "jq": {"--arg", "--argjson", "--slurpfile", "--rawfile", "--indent",
           "-L", "--args", "--jsonargs"},
    "awk": {"-F", "-v", "-f", "-i", "-l", "-E"},
    "head": {"-n", "-c"}, "tail": {"-n", "-c"}, "cut": {"-d", "-f", "-c", "-b"},
    "sort": {"-k", "-t", "-S", "-T", "-o", "--key", "--field-separator",
             "--buffer-size", "--temporary-directory", "--output"},
    "diff": {"-U", "-C", "-L", "--label", "-F"},
    "git diff": {"-U", "--unified", "-M", "-C", "-S", "-G", "-O",
                 "--diff-filter", "--relative"},
    "git grep": {"-e", "-f", "-A", "-B", "-C", "-m", "--max-depth"},
    "git blame": {"-L", "-S", "--contents"},
}
_VALUE_OPTS["egrep"] = _VALUE_OPTS["fgrep"] = _VALUE_OPTS["grep"]
# grep/rg flags whose output is a count, a file list, or nothing
_QUIET_LONG = frozenset({"--quiet", "--silent", "--count", "--count-matches",
                         "--files-with-matches", "--files-without-match",
                         "--files", "--json", "--stats"})
_QUIET_SHORT = frozenset("qclL")
_DIFF_SUMMARY = frozenset({"--stat", "--numstat", "--shortstat", "--name-only",
                           "--name-status", "--summary", "--quiet",
                           "--exit-code", "--dirstat", "-q", "--brief", "-s",
                           "--report-identical-files", "--no-patch",
                           "--cached", "--staged"})
_SED_ADDR = r"(?:\d+|\$|/[^/]*/)"
_SED_PRINT_ONLY = re.compile(r"^(?:%s(?:,%s)?)?!?p$" % (_SED_ADDR, _SED_ADDR))
_AWK_UNSAFE = re.compile(
    r"system|getline|close|fflush|BEGIN|END|\||"
    r"\bprintf?\b[^;}]*?(?<![<>=!])>(?!=)")
_PYTHON = re.compile(r"^python(?:3(?:\.\d+)?)?$")
_REVISION = re.compile(r"^(?:HEAD|FETCH_HEAD|ORIG_HEAD|MERGE_HEAD|@)(?:[~^].*)?$|"
                       r"\.\.|[~^]\d*$|^[0-9a-f]{7,40}$|@\{")


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
                cwd = resolve(argv[k + 1], cwd, stage.home)
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


def _split_args(name, args):
    """``(options, operands)``: option values (``-A 3``, ``-e PAT``) stay
    with their option, so they are never taken for a pattern or a file.
    Everything after ``--`` is an operand."""
    takes_value = _VALUE_OPTS.get(name, set())
    opts, ops, k = [], [], 0
    while k < len(args):
        a = args[k]
        if a == "--":
            ops += args[k + 1:]
            break
        if a.startswith("-") and a != "-":
            if a in takes_value and k + 1 < len(args):
                opts.append((a, args[k + 1]))
                k += 2
                continue
            opts.append((a, None))
        else:
            ops.append(a)
        k += 1
    return opts, ops


def _short_flags(opts):
    """All single-letter flags, bundles included (``-ni`` → n, i)."""
    out = set()
    for flag, _ in opts:
        if flag.startswith("-") and not flag.startswith("--"):
            out |= set(flag[1:])
    return out


def _sed_parts(args):
    """``(in_place, scripts, files)`` for a sed argument list. Any short
    bundle with ``i`` (``-ni``, ``-Ei``) edits in place; BSD ``-i ''`` and
    ``-i .bak`` take a separate backup suffix, recognised only when a
    script and a file still follow it (``-e s/x/y/ -i .env`` edits .env)."""
    in_place, scripts, files, has_e, k = False, [], [], False, 0
    has_e = any(a in ("-e", "--expression", "-f", "--file")
                or a.startswith("--expression=") or a.startswith("--file=")
                for a in args)
    while k < len(args):
        a = args[k]
        if a in ("-e", "--expression"):
            if k + 1 < len(args):
                scripts.append(args[k + 1])
            k += 2
            continue
        if a.startswith("--expression="):
            scripts.append(a.split("=", 1)[1])
            k += 1
            continue
        if a in ("-f", "--file") or a.startswith("--file="):
            scripts.append(None)          # a script file: unknown commands
            k += 1 if "=" in a else 2
            continue
        if a.startswith("--in-place"):
            in_place = True
            k += 1
            continue
        if a.startswith("-") and not a.startswith("--") and "i" in a[1:]:
            in_place = True
            if a == "-i" and k + 1 < len(args):
                nxt = args[k + 1]
                following = [x for x in args[k + 2:] if not x.startswith("-")]
                if nxt == "" or (re.match(r"^\.[\w.-]{1,16}$", nxt)
                                 and len(following) >= (1 if has_e else 2)):
                    k += 1                # BSD/macOS backup suffix
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


def _awk_parts(args):
    """``(safe, files, shows_lines)``: awk whose program can only print (no
    ``-f`` script, no in-place extension, no redirection/pipe/system/getline
    and no BEGIN/END summary), the files it reads, and whether what it
    prints is file text (fields, or whole lines) rather than a count."""
    opts, ops = _split_args("awk", args)
    if any(flag.startswith("-i") or flag.startswith("-f") or flag.startswith("-E")
           or flag.startswith("--") for flag, _ in opts):
        return False, [], False
    if not ops:
        return False, [], False
    program = ops[0]
    safe = len(program) <= 2000 and not _AWK_UNSAFE.search(program)
    prints = re.findall(r"\bprintf?\b([^;}]*)", program)
    return safe, ops[1:], all("$" in p for p in prints)


def _sed_prints_only(script):
    if script is None or len(script) > 500:
        return False
    return bool(_SED_PRINT_ONLY.match(re.sub(r"\s+", "", script)))


_JQ_SELECTOR = re.compile(r"^\s*\.[\w\[\]\.\"'-]*(?:\s*\|\s*\.[\w\[\]\.\"'-]*)*\s*$")


def _jq_shows_content(args):
    """jq prints file text only for plain selectors (``.``, ``.a.b``,
    ``.items[] | .name``); ``length``, ``keys``, ``type``, ``empty`` and
    other functions print a derived value."""
    opts, ops = _split_args("jq", args)
    if any(f in ("-f", "--from-file", "-e", "--exit-status", "-n", "--null-input")
           for f, _ in opts) or not ops:
        return False
    return bool(_JQ_SELECTOR.match(ops[0]))


def _is_viewer(name, args):
    """sed/awk/json.tool used only to print file content."""
    if name == "sed":
        in_place, scripts, _ = _sed_parts(args)
        return not in_place and bool(scripts) and all(map(_sed_prints_only, scripts))
    if name == "awk":
        return _awk_parts(args)[0]
    if name == "json.tool":
        return len(_split_args(name, args)[1]) <= 1
    return False


def _writes_via_option(name, args):
    """Read-only-looking programs that write a file through an option."""
    opts, _ = _split_args(name, args)
    if name == "sort":
        return any(f in ("-o", "--output") or f.startswith("--output=") for f, _ in opts) \
            or "o" in _short_flags(opts)
    if name == "tree":
        return "o" in _short_flags(opts) or any(f.startswith("--output")
                                                for f, _ in opts)
    if name.startswith("git "):
        return any(f.startswith("--output") for f, _ in opts)
    return False


def _shows_content(name, args):
    """Whether the program prints file lines (not a count, list or verdict)."""
    if _writes_via_option(name, args):
        return False
    opts, _ = _split_args(name, args)
    flags = {f.split("=", 1)[0] for f, _ in opts}
    if name == "jq":
        return _jq_shows_content(args)
    if name in _PATTERN:
        return not (flags & _QUIET_LONG or _short_flags(opts) & _QUIET_SHORT
                    or "o" in _short_flags(opts) or "--only-matching" in flags)
    if name in ("diff", "git diff"):
        return not flags & _DIFF_SUMMARY
    if name == "sort":
        return not _short_flags(opts) & set("cC") and "--check" not in flags
    if name == "awk":
        safe, _, lines = _awk_parts(args)
        return safe and lines
    if name in ("sed", "json.tool"):
        return _is_viewer(name, args)
    return name in _PASS_THROUGH or name == "git blame"


# a later pipeline stage that passes the earlier stage's lines on intact
# (possibly fewer of them); cut, tr, wc, jq and the like transform them
_PRESERVES = frozenset({"cat", "head", "tail", "grep", "egrep", "fgrep", "rg",
                        "sort", "nl", "less", "more", "sed", "json.tool"})
def _harmless_target(op, target):
    """Redirect targets that write no file: /dev/*, fd dups, closing."""
    return (target.startswith("/dev/") or target == "-"
            or (op in (">&", "<&") and target.isdigit())
            or target.startswith(">(") or target.startswith("<("))


def _stage_read_only(stage):
    if stage.subst:
        return False
    for fd, op, target in stage.redirects:
        if op in ("<", "<<", "<<-", "<<<", "<&"):
            continue
        if _harmless_target(op, target) and not target.startswith(">("):
            continue
        return False
    name, args, _ = _program(stage)
    if not name:
        return True
    if _writes_via_option(name, args):
        return False
    if name.startswith("git "):
        return name[4:] in _READ_ONLY_GIT
    if name in ("sed", "awk", "json.tool"):
        return _is_viewer(name, args)
    return name in _READ_ONLY


def is_read_only(command, home=None):
    """True when every stage is a known read-only program with no output
    redirect to a file and no substitution. Anything else may write."""
    stages = parse(command, "", home)
    return bool(stages) and all(_stage_read_only(s) for s in stages)


# ------------------------------------------------------------------ reads

_DIFF_NEW_FILE = re.compile(r"^\+\+\+ (?:[bwi]/)?(.+?)\t?$", re.M)
_OUTPUT_PATH = re.compile(r"^([^\s:]+):", re.M)


def _pattern_parts(name, args):
    """``(files, prints_filenames, not_worktree)`` for grep/rg/jq/git grep.
    ``not_worktree``: a git grep of the index or of a revision."""
    opts, ops = _split_args(name, args)
    flags = {f.split("=", 1)[0] for f, _ in opts}
    if not any(f in ("-e", "-f", "--regexp", "--file") for f in flags):
        ops = ops[1:]                              # the first operand is the pattern
    if name == "git grep":
        # anything between the pattern and `--` is a revision
        before = args[:args.index("--")] if "--" in args else args
        revs = _split_args(name, before)[1][1:]
        paths = args[args.index("--") + 1:] if "--" in args else []
        return paths, True, bool(revs) or "--cached" in flags
    if name == "jq":
        return ops, False, False
    shorts = _short_flags(opts)
    if "h" in shorts or "--no-filename" in flags:
        return ops, False, False
    if "H" in shorts or "--with-filename" in flags:
        return ops, True, False
    recursive = bool(shorts & set("rR")) or "--recursive" in flags
    if name == "rg":
        return ops, (len(ops) != 1 or recursive), False
    return ops, recursive or len(ops) > 1, False


def _within(path, roots):
    """``path`` is one of ``roots`` or inside one of them."""
    return any(path == r or path.startswith(r.rstrip("/") + "/") or r in ("", ".")
               for r in roots)


def _git_revs_and_paths(args):
    """``(revisions, paths)`` for git diff/blame arguments; without ``--``
    two or more operands are ambiguous (branch names look like paths), so
    they come back as ``None``."""
    opts, ops = _split_args("git diff", args)
    if "--" in args:
        before = args[:args.index("--")]
        return _split_args("git diff", before)[1], args[args.index("--") + 1:]
    if len(ops) >= 2:
        return None, None
    revs = [o for o in ops if _REVISION.search(o)]
    return revs, [o for o in ops if o not in revs]


def _stage_reads(stage, output, attribute):
    """``(content, listed)`` paths one read-only stage looked at.

    ``attribute`` says whether this stage is the command's only producer of
    output, so the output (its emptiness, ``path:`` prefixes, diff headers)
    can be trusted to be its own. Programs that print only when something
    matched (grep, diff, git diff) earn content credit only then.
    """
    name, args, cwd = _program(stage)
    content, listed = [], []
    ops = _split_args(name, args)[1]
    printed = bool((output or "").strip())
    if stage.stdout_redirected:
        return set(), set()
    if name in _PATTERN:
        files, prefixes, not_worktree = _pattern_parts(name, args)
        if not_worktree:
            return set(), set()
        if prefixes:
            # several files or a recursive search: only files whose lines
            # were printed, and only when this stage printed everything
            if attribute and not stage.piped_in:
                roots = [resolve(f, cwd, stage.home) for f in files] or [cwd or ""]
                for p in _OUTPUT_PATH.findall(output or ""):
                    r = resolve(p, cwd, stage.home)
                    if r and _within(r, [x for x in roots if x is not None]) \
                            and ("/" in p or "." in p):
                        content.append(p)
        elif attribute and printed:
            content = [f for f in files if f != "-"]
    elif name == "sed":
        content = _sed_parts(args)[2]
    elif name == "awk":
        content = _awk_parts(args)[1]
    elif name in ("git diff", "git blame"):
        revs, paths = _git_revs_and_paths(args)
        worktree = paths is not None and not (
            {f.split("=", 1)[0] for f, _ in _split_args(name, args)[0]}
            & {"--cached", "--staged"})
        if name == "git diff":
            worktree = worktree and len(revs) <= 1
        else:
            worktree = worktree and not revs
        if worktree and attribute and printed:
            headers = [h for h in _DIFF_NEW_FILE.findall(output) if h != "/dev/null"]
            if headers and name == "git diff":
                content = ["\0" + h for h in headers]   # only files it showed
            elif len(paths) == 1:
                content = list(paths)
    elif name == "git show":
        content = []                           # committed copies, never the tree
    elif name == "diff":
        if attribute and printed:
            content = [o for o in ops if o != "-"]
    elif name in _CONTENT:
        content = [o for o in ops if o != "-"]
    elif name in _LISTING:
        listed = [o for o in ops if o != "-"]
    if name in _PASS_THROUGH or name in _CONTENT:
        content += [t for fd, op, t in stage.redirects if op == "<" and fd in ("", "0")]
    placed = []
    for group in (content, listed):
        out = set()
        for p in group:
            if p.startswith("\0"):
                out |= _repo_candidates(p[1:], cwd)
            elif ":" not in p or p.startswith("/") or p.startswith("."):
                r = resolve(p, cwd, stage.home)
                if r:
                    out.add(r)
        placed.append(out)
    return placed[0], placed[1]


def _repo_candidates(rel, cwd):
    """Where a repo-relative diff header can live: under ``cwd`` or one of
    its ancestors (the repo root). Without a known cwd it stays root-less."""
    rel = posixpath.normpath(rel)
    if not cwd:
        return {rel}
    out, d = set(), cwd
    while True:
        out.add(posixpath.normpath(posixpath.join(d, rel)))
        parent = posixpath.dirname(d)
        if parent == d:
            return out
        d = parent


def _pipelines(stages):
    groups, current = [], []
    for s in stages:
        current.append(s)
        if not s.piped:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


def _pipeline_shows(pipe):
    """Whether a pipeline's first stage's lines reach the agent intact:
    the producer prints file text, and every later stage passes lines on
    (``cat a | grep x`` does; ``cat a | wc -l | tr -d ' '`` does not)."""
    name, args, _ = _program(pipe[0])
    if not _shows_content(name, args):
        return False
    for stage in pipe[1:]:
        later, largs, _ = _program(stage)
        if later not in _PRESERVES or not _shows_content(later, largs) \
                or stage.redirects:
            return False
    return not any(s.stdout_redirected for s in pipe)


def _silent(pipe):
    """``true`` / ``:``: a pipeline that prints nothing."""
    return len(pipe) == 1 and pipe[0].argv[:1] in (["true"], [":"]) \
        and not pipe[0].redirects


def _sole_producer(pipes):
    """The one single-stage pipeline that printed everything, if any."""
    producers = [p for p in pipes if not _silent(p)]
    if len(producers) == 1 and len(producers[0]) == 1:
        return producers[0]
    return None


def _credited_pipelines(stages):
    """Pipelines whose reads can be credited: none in a command that runs
    anything in the background (the read may race the write) or whose
    group output is redirected; none joined by ``||`` (one side failed or
    never ran), except ``A || true``, where only A can have printed."""
    pipes = _pipelines(stages)
    if stages.grouped_output or any(s.sep == "&" for s in stages):
        return pipes, set()
    ok, prev = set(), ";"
    for k, pipe in enumerate(pipes):
        after = pipes[k + 1] if k + 1 < len(pipes) else None
        joined = pipe[-1].sep == "||" and not (after and _silent(after))
        if prev != "||" and not joined:
            ok.add(k)
        prev = pipe[-1].sep
    return pipes, ok


def read_operands(command, output="", cwd="", home=None):
    """``(content, listed)``: paths a read-only command showed the content
    of, and paths it only listed or summarized.

    A pipeline counts as showing content only if its producer prints file
    text and every later stage passes those lines on (``cat a | grep x``
    does; ``cat a | wc -l``, ``grep -q``, ``jq length``, ``git diff --stat``
    and ``> /dev/null`` do not); what it read is then demoted to listed.
    Output (emptiness, ``path:`` prefixes, diff headers) is trusted only
    when one stage produced all of it, and nothing is shown if nothing
    was printed.
    """
    stages = parse(command, cwd, home) or Stages()
    pipes, credited = _credited_pipelines(stages)
    printed = bool((output or "").strip())
    sole = _sole_producer(pipes)
    content, listed = set(), set()
    for k, pipe in enumerate(pipes):
        pipe_content, pipe_listed = set(), set()
        for j, stage in enumerate(pipe):
            c, l_ = _stage_reads(stage, output, pipe is sole)
            pipe_content |= c
            pipe_listed |= l_
        if k in credited and printed and _pipeline_shows(pipe):
            content |= pipe_content
        else:
            listed |= pipe_content
        listed |= pipe_listed
    return content, listed


# ---------------------------------------------------------------- effects

def _copy_targets(name, args, cwd, home):
    """``(dst, alt)`` per file a ``cp`` writes (alt: the other reading of an
    ambiguous ``cp a b``, b possibly a directory). Glob or variable sources
    into a directory can't be named: nothing is owed for them."""
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
    out = []
    for s in sources:
        if set(s) & set("*?[$`"):
            continue
        inside = posixpath.join(target, posixpath.basename(s.rstrip("/")))
        if target_dir is not None or len(sources) > 1:
            dst, alt = inside, None
        elif target.endswith("/"):
            dst, alt = inside, None
        else:
            dst, alt = target, inside
        dst = resolve(dst, cwd, home)
        if dst:
            out.append((dst, resolve(alt, cwd, home) if alt else None))
    return out


def _writes(stage):
    """``(path, alt)`` per file a stage writes: stdout/both redirects
    (stderr logs are not the edit), ``tee`` operands, ``sed -i`` files,
    ``sort -o``, ``cp`` destinations. Targets that can't be placed (devices,
    process substitutions, variables) are never owed: no read could pay."""
    name, args, cwd = _program(stage)
    out = []
    for fd, op, target in stage.redirects:
        if op in (">", ">>", ">|", "&>", "&>>", ">&") and fd in ("", "1") \
                and not _harmless_target(op, target):
            out.append(target)
    if name == "tee":
        out += _split_args("tee", args)[1]
    elif name == "sed":
        in_place, _, files = _sed_parts(args)
        if in_place:
            out += files
    elif name == "sort":
        opts, _ = _split_args("sort", args)
        out += [v for f, v in opts if f in ("-o", "--output") and v]
    placed = []
    for p in out:
        if not set(p) & set("*?[") and not _harmless_target(">", p):
            r = resolve(p, cwd, stage.home)
            if r:
                placed.append((r, None))
    if name == "cp":
        placed += _copy_targets(name, args, cwd, stage.home)
    return placed


def _moves(stage):
    """``(src, dst, alt)`` per moved file; ``alt`` is the other reading of
    an ambiguous move (``mv a b`` where b may be a directory, or
    ``mv src/ lib/`` where lib may not exist yet). A glob source moved into
    a directory comes back with ``src`` as the (resolved) pattern and
    ``dst`` as the directory, flagged by ``alt`` being ``"*glob*"``."""
    name, args, cwd = _program(stage)
    if name not in ("mv", "git mv"):
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
    moves = []
    for s in sources:
        if set(s) & set("$`"):
            continue
        if set(s) & set("*?["):
            src, dst = resolve(s, cwd, stage.home), resolve(target, cwd, stage.home)
            if src and dst:
                moves.append((src, dst, "*glob*"))
            continue
        base = posixpath.basename(s.rstrip("/"))
        inside = posixpath.join(target, base)
        if target_dir is not None or len(sources) > 1:
            dst, alt = inside, None
        elif target.endswith("/"):
            dst, alt = inside, target.rstrip("/")
        else:
            dst, alt = target, inside
        src = resolve(s, cwd, stage.home)
        dst = resolve(dst, cwd, stage.home)
        alt = resolve(alt, cwd, stage.home) if alt else None
        if src and dst:
            moves.append((src, dst, alt))
    return moves


def _removes(stage):
    """``(path, recursive)`` per operand of an ``rm``-like stage. ``git rm
    --cached`` and dry runs remove nothing from disk."""
    name, args, cwd = _program(stage)
    if name not in ("rm", "git rm", "unlink", "rmdir"):
        return []
    opts, ops = _split_args(name, args)
    flags = {f for f, _ in opts}
    if name == "git rm" and (flags & {"--cached", "-n", "--dry-run"}):
        return []
    recursive = bool(_short_flags(opts) & set("rR")) or "--recursive" in flags \
        or name == "rmdir"
    out = []
    for o in ops:
        r = resolve(o, cwd, stage.home)
        if r:
            out.append((r, recursive))
    return out


_KNOWN_EFFECTS = frozenset({"tee", "sed", "mv", "git mv", "rm", "git rm",
                            "unlink", "rmdir", "mkdir", "touch", "cp", "sort"})


def effects(command, cwd="", output="", home=None):
    """Ordered effects of a (possibly mutating) command:
    ``("write", path, alt)``, ``("move", src, dst, alt)``,
    ``("remove", path, recursive)``, ``("read", path)`` for content a stage
    showed the agent, and ``("opaque",)`` for a stage whose writes can't be
    known (a script, a build, a substitution). None when the command can't
    be parsed."""
    stages = parse(command, cwd, home)
    if stages is None:
        return None
    out = []
    pipes, credited = _credited_pipelines(stages)
    printed = bool((output or "").strip())
    sole = _sole_producer(pipes)
    for k, pipe in enumerate(pipes):
        reads = set()
        for j, stage in enumerate(pipe):
            name, args, _ = _program(stage)
            known = _stage_read_only(stage) or (name in _KNOWN_EFFECTS
                                                and not stage.subst)
            if not known:
                out.append(("opaque",))
            out += [("write",) + w for w in _writes(stage)]
            out += [("move",) + m for m in _moves(stage)]
            out += [("remove",) + r for r in _removes(stage)]
            if _stage_read_only(stage):
                reads |= _stage_reads(stage, output, pipe is sole)[0]
        if k in credited and printed and _pipeline_shows(pipe):
            out += [("read", p) for p in sorted(reads)]
    return out
