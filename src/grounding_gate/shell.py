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
import sys

# ------------------------------------------------------------------ lexer

_OPERATORS = ("&&", "||", "|&", ";;", "|", ";", "&", "(", ")")
_REDIRECTS = ("&>>", "&>", ">>", ">|", ">&", "<<<", "<<-", "<<", "<&", "<>", ">", "<")
_MAX_COMMAND = 100_000          # longer than this: not analysed (unknown)


class _Word(str):
    """A shell word with quotes removed. ``subst`` marks a command, process
    or arithmetic substitution inside it (its effects are unknown);
    ``expands`` a ``$`` outside single quotes (its text is not what the
    program sees)."""
    subst = False
    expands = False


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
    expands = False
    heredocs = []          # (delimiter, strip_tabs, quoted) awaiting a newline
    i, n = 0, len(command)

    def end_word():
        nonlocal word, in_word, subst, expands
        if in_word:
            w = _Word("".join(word))
            w.subst = subst
            w.expands = expands or subst
            tokens.append(("word", w))
        word, in_word, subst, expands = [], False, False, False

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
                expands = expands or d == "$"
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
        expands = expands or c == "$"
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
    final_cwds = frozenset()
    loop_cd = False
    defines_function = False


class Stage:
    """One simple command: ``argv`` (env assignments, keywords and
    ``sudo``/``time`` style prefixes removed), its redirects as
    ``(fd, op, target)``, whether anything in it is a substitution, whether
    its stdout is piped into the next stage / its stdin piped from the
    previous one, and the directory and home relative paths resolve
    against."""

    def __init__(self, argv, redirects, subst, piped, piped_in, cwd, home, sep=";",
                 cwd_alt=None):
        self.argv, self.redirects, self.subst = argv, redirects, subst
        self.piped, self.piped_in = piped, piped_in
        self.cwd, self.home, self.sep = cwd, home, sep
        # after `cd x;` (not `&&`) the cd may have failed: the command then
        # ran in cwd_alt instead. Reads under an uncertain directory earn no
        # credit; writes are owed at both places.
        self.cwd_alt = cwd_alt

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
_OPENERS = frozenset({"{", "if", "while", "until"})
_CONDITIONAL = re.compile(
    r"(?:^|[;&|\n({!]\s*|\btime\s+)(?:if|while|until|case|for|select)\s")


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
    stages, saved, dirstack, groups = Stages(), [], [], []
    base_alt = None
    # `||`, `if`/`while`/`case` or a background job anywhere makes every
    # stage's running uncertain: a successful call no longer proves it ran
    conditional = "||" in command or bool(_CONDITIONAL.search(command))
    argv, redirects, subst = [], [], False
    expect_target, piped_in, just_closed = None, False, False
    in_pattern = False          # reading a case pattern (`x)`), not a command
    last_sep, pending_alt, exited, errexit, or_cd = ";", None, False, False, None
    list_pin = None

    def flush(sep):
        nonlocal argv, redirects, subst, base, piped_in, base_alt, last_sep, \
            pending_alt, exited, errexit, or_cd, list_pin
        piped = sep in ("|", "|&")
        if argv and argv[0] in _CLOSERS and (piped or redirects):
            stages.grouped_output = True
        words, closed = argv, 0
        while words and words[0] in _CLOSERS:
            words, closed = words[1:], closed + 1    # (a file named `done` stays)
        if words[:1] == ["[["]:
            redirects = []                # `[[ a > b ]]` compares strings
        opened = []
        while words and (_ASSIGNMENT.match(words[0]) or words[0] in _PREFIXES):
            if words[0] in ("sudo", "env", "command", "time") and len(words) > 1 \
                    and words[1].startswith("-"):
                break            # `sudo -u www ...`: options we can't read
            if words[0] in _OPENERS:
                opened.append(words[0])
            words = words[1:]
        if words and words[0] in ("for", "select"):
            opened.append(words[0])
        for kind in opened:
            groups.append((base, base_alt, kind))
        if words[:1] == ["case"]:
            words = []                  # `case WORD in` runs nothing
        if words or redirects or subst:
            stage = Stage(list(words), redirects, subst, piped, piped_in,
                          base, home, sep, base_alt)
            stage.raw = list(argv)          # with assignments, for substitutions
            stage.certain = not conditional and sep != "&" and not exited
            stage.depth = len(saved)
            name = words[0] if words else ""
            if name in ("exit", "return", "exec") and (name != "exec" or len(words) > 1):
                exited = True             # later stages may never run
            if name == "set" and any(w in ("-e", "errexit") or re.match(r"^-\w*e", w)
                                     for w in words[1:]):
                errexit = True
            if name in ("cd", "pushd", "popd") and any(
                    g[2] in ("for", "while", "until", "select") for g in groups):
                stages.loop_cd = True      # a loop may cd again and again
            stages.append(stage)
            guarded = last_sep in ("&&", "||")
            last_sep = sep
            if name in ("cd", "pushd", "popd") and not piped and sep != "&" \
                    and not piped_in:
                if name == "popd":
                    new = dirstack.pop() if dirstack else None
                else:
                    target = _cd_target(words)
                    if name == "pushd":
                        dirstack.append(base)
                    new = None if target == "-" else resolve(target, base, home)
                if or_cd is not None:
                    new, or_cd = None, None   # `cd x || cd y`: which one?
                # after `cd x;` the next command runs even if cd failed, so
                # only `cd x && ...` pins where it runs; after `[ x ] && cd y
                # && z` only z's list is pinned: a later list may run where
                # the cd never happened
                if sep == "&&" or errexit and sep in (";", "\n") and not guarded:
                    base_alt = None
                    if guarded:
                        pending_alt = base
                    list_pin = base
                elif sep == "||":
                    or_cd = base          # `cd x || exit` pins; `cd x || y` doesn't
                    base_alt = None
                else:
                    base_alt = base
                base = new
            elif or_cd is not None:
                if name not in ("exit", "return") or sep in ("&&", "||", "|", "|&"):
                    base = None           # `cd x || other`: can't say where
                    base_alt = or_cd
                or_cd = None
            if sep == "||" and list_pin is not None:
                # `cd x && y || z`: z also runs when the cd failed
                base_alt = base_alt if base_alt is not None else list_pin
            if sep not in ("&&", "||", "|", "|&"):
                list_pin = None
            if sep not in ("&&", "|", "|&") and pending_alt is not None:
                base_alt = base_alt if base_alt is not None else pending_alt
                pending_alt = None
        for _ in range(closed):
            if groups:
                outer_base, outer_alt, kind = groups.pop()
                if piped:
                    # a piped group ran in a subshell: its cd ended with it
                    base, base_alt = outer_base, outer_alt
                elif kind != "{" and base != outer_base and base_alt is None:
                    # a loop may run no times, a branch may not be taken
                    base_alt = outer_base
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
        if kind == "word" and in_pattern and value == "esac":
            in_pattern = False
            argv.append(value)
        elif kind == "word":
            argv.append(value)
            subst = subst or value.subst
            if value == "in" and argv[:1] == ["case"] and len(argv) == 3:
                in_pattern = True
                groups.append((base, base_alt, "case"))
        elif in_pattern and value == "(" and not argv:
            continue                     # `(x)` pattern form
        elif value == "(" and argv and kind == "op":
            stages.defines_function = True   # `f() { ...; }`: runs later, if ever
            flush(";")
            saved.append((base, base_alt))
        elif in_pattern and value == ")":
            # the end of a case pattern: the words before it were the pattern
            argv, redirects, subst, in_pattern = [], [], False, False
        elif kind == "redir":
            expect_target = value
        elif value == "(":
            flush(";")
            saved.append((base, base_alt))
        elif value == ")":
            flush(";")
            if saved:
                base, base_alt = saved.pop()
            just_closed = True
        else:                                    # | |& ; && || & ;;
            if value in (";;", ";&", ";;&"):
                in_pattern = True
            flush(value)
    flush(";")
    # a stage after `&&` ran only if everything before it in its list
    # succeeded; the exit status is the LAST list's, so a successful call
    # proves that only for the last list (`a > t && mv t f; echo` doesn't)
    last_list = len(stages) - 1
    while last_list > 0 and stages[last_list - 1].sep in ("&&", "|", "|&"):
        last_list -= 1
    # where the command ends up: the hook may report this directory
    stages.final_cwds = {base, base_alt} - {None} if base is not None else {None}
    guarded = False
    for k, stage in enumerate(stages):
        if k and stages[k - 1].sep not in ("&&", "|", "|&"):
            guarded = False                      # a new list
        if guarded and k < last_list:
            stage.certain = False
        guarded = guarded or stage.sep == "&&"
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
    r"system|getline|close|fflush|BEGIN|END|@|\||"
    r"\bprintf?\b[^;}]*?(?<![<>=!])>(?!=)")
_PYTHON = re.compile(r"^python(?:3(?:\.\d+)?)?$")
_REVISION = re.compile(r"^(?:HEAD|FETCH_HEAD|ORIG_HEAD|MERGE_HEAD|@)(?:[~^].*)?$|"
                       r"\.\.|[~^]\d*$|^[0-9a-f]{7,40}$|@\{")


_ALIASES = {"gsed": "sed", "gawk": "awk", "mawk": "awk", "nawk": "awk",
            "ggrep": "grep", "gsort": "sort", "gcp": "cp", "gmv": "mv",
            "grm": "rm", "python3": "python3"}


# programs that run the rest of their argv as a command: (options that take
# a value, number of fixed operands before the command)
_WRAPPERS = {
    "timeout": ({"-s", "--signal", "-k", "--kill-after"}, 1),
    "nice": ({"-n", "--adjustment"}, 0),
    "stdbuf": ({"-i", "-o", "-e", "--input", "--output", "--error"}, 0),
    "ionice": ({"-c", "-n", "-p", "--class", "--classdata"}, 0),
    "nohup": (set(), 0), "command": (set(), 0),
    "time": ({"-f", "--format"}, 0),
}


_SYSTEM_BIN = ("/bin/", "/usr/bin/", "/usr/local/bin/", "/opt/homebrew/bin/",
               "/usr/sbin/", "/sbin/")


def _system_name(word):
    """``/usr/bin/sed`` is ``sed``; any other path keeps its spelling."""
    if "/" in word and any(word.startswith(d) and "/" not in word[len(d):]
                           for d in _SYSTEM_BIN):
        return word.rsplit("/", 1)[1]
    return word


def _unwrap(name, args):
    """The command a wrapper runs (``timeout 5 sed ...`` runs ``sed ...``),
    or None when its options can't be read (the stage then stays opaque)."""
    takes_value, fixed = _WRAPPERS[name]
    k = 0
    while k < len(args) and args[k].startswith("-") and args[k] != "-":
        a = args[k]
        if a == "--":
            k += 1
            break
        if name == "command" and a != "-p":
            return None                    # -v/-V describe, they don't run
        if name == "time" and (a.startswith("-o") or a.startswith("--o")
                               or a.startswith("-a")):
            return None                    # time -o FILE writes its report
        if name == "nice" and re.match(r"^-\d+$", a):
            k += 1
            continue
        if a in takes_value:
            k += 2
            continue
        if "=" in a or name in ("timeout", "time", "nohup") or (
                name == "stdbuf" and len(a) > 2) or (
                name in ("nice", "ionice") and len(a) > 2 and a[2:].isdigit()):
            k += 1
            continue
        return None
    k += fixed
    return args[k:] if k < len(args) else None


def _program(stage):
    """``(name, args, cwd)``; git subcommands are named ``git diff`` etc.,
    with global options skipped (``-C DIR`` moves the base)."""
    argv, cwd = list(stage.argv), stage.cwd
    for _ in range(8):
        if argv and _system_name(argv[0]) == "busybox":
            argv = argv[1:]
        if not argv:
            return "", [], cwd
        # /usr/bin/sed is sed; gsed and gawk are GNU sed and awk. A repo's
        # own script/test or ./cat is NOT the system program.
        base = _system_name(argv[0])
        base = _ALIASES.get(base, base)
        if base != argv[0]:
            argv = [base] + argv[1:]
        if base not in _WRAPPERS:
            break
        inner = _unwrap(base, argv[1:])
        if not inner:
            break
        argv = inner
    else:
        return "wrapper", argv, cwd        # too deep: stays opaque
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


def _sed_parts(args, bsd=True):
    """``(in_place, scripts, files)`` for a sed argument list. Any short
    bundle with ``i`` (``-ni``, ``-Ei``) edits in place; BSD ``-i ''`` and
    ``-i .bak`` take a separate backup suffix, recognised only when a
    script and a file still follow it (``-e s/x/y/ -i .env`` edits .env).
    With ``bsd`` False (strict mode on a GNU system) only ``-i ''`` is."""
    args = [_expand_long(a, _SED_LONG) for a in args]
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
                if nxt == "" or (bsd and re.match(r"^\.[\w.-]{1,16}$", nxt)
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


# the platform's sed: BSD reads `-i .bak` as a backup suffix, GNU as a file
_BSD_SED = sys.platform == "darwin" or "bsd" in sys.platform
_SED_LONG = ("--in-place", "--expression", "--file", "--quiet", "--silent",
             "--regexp-extended", "--null-data", "--zero-terminated",
             "--separate", "--sandbox", "--unbuffered", "--line-length",
             "--posix", "--debug", "--follow-symlinks", "--help", "--version")


def _expand_long(arg, names):
    """GNU accepts any unambiguous prefix of a long option (``--in``)."""
    if not arg.startswith("--") or arg == "--":
        return arg
    name, eq, value = arg.partition("=")
    hits = [n for n in names if n.startswith(name)]
    return hits[0] + eq + value if len(hits) == 1 else arg


_SED_SIMPLE = frozenset("dpPDnNgGhHx=lzF{}")


def _bracket_end(s, i):
    """Index just past the bracket expression opening at ``s[i]`` ('['),
    or -1."""
    j = i + 1
    if j < len(s) and s[j] == "^":
        j += 1
    if j < len(s) and s[j] == "]":
        j += 1
    while j < len(s):
        if s[j] == "[" and j + 1 < len(s) and s[j + 1] in ":.=":
            close = s.find(s[j + 1] + "]", j + 2)
            if close < 0:
                return -1
            j = close + 2
            continue
        if s[j] == "]":
            return j + 1
        if s[j] == "\n":
            return -1
        j += 1
    return -1


def _sed_safe(script):
    """True when a sed script provably neither writes a file nor runs a
    command. A single linear pass over its commands (address, ``!``,
    command, arguments); anything it doesn't recognise is unsafe, so
    ``w``/``W``/``e``, ``s///w`` and ``s///e`` can't hide in an address, a
    bracket expression or an odd delimiter."""
    if script is None or len(script) > 8000:
        return False
    s, n = script, len(script)

    def part(i, d, brackets):
        # i: just past the opening delimiter; returns just past the closing one
        while i < n:
            c = s[i]
            if c == "\\":
                i += 2
                continue
            if c == "\n" and brackets:
                return -1
            if c == "[" and brackets:
                j = _bracket_end(s, i)
                if j < 0 or d in s[i + 1:j - 1]:
                    return -1              # delimiter inside [...]: ambiguous
                i = j
                continue
            if c == d:
                return i + 1
            i += 1
        return -1

    def blanks(i):
        while i < n and s[i] in " \t":
            i += 1
        return i

    def address(i):
        if i < n and s[i].isdigit():
            while i < n and s[i].isdigit():
                i += 1
            if i < n and s[i] == "~":
                i += 1
                while i < n and s[i].isdigit():
                    i += 1
            return i
        if i < n and s[i] == "$":
            return i + 1
        if i < n and s[i] == "/":
            i = part(i + 1, "/", True)
            while 0 <= i < n and s[i] in "IM":
                i += 1
            return i
        if i < n and s[i] == "\\":
            return -1                          # \cREGEXc: not modelled
        return i

    def ends(i):
        i = blanks(i)
        return i if i >= n or s[i] in ";\n}#" else -1

    def line_end(i):
        while i < n and s[i] != "\n":
            i += 2 if s[i] == "\\" else 1
        return i

    i = 0
    while i < n:
        if s[i] in " \t\n;":
            i += 1
            continue
        if s[i] == "#":
            i = line_end(i)
            continue
        j = address(i)
        if j < 0:
            return False
        if j != i:
            j = blanks(j)
            if j < n and s[j] == ",":
                j = blanks(j + 1)
                if j < n and s[j] in "+~":
                    j += 1
                    while j < n and s[j].isdigit():
                        j += 1
                else:
                    k = address(j)
                    if k <= j:
                        return False
                    j = k
        i = blanks(j)
        if i < n and s[i] == "!":
            i = blanks(i + 1)
        if i >= n:
            return False
        c, i = s[i], i + 1
        if c in _SED_SIMPLE:
            i = ends(i) if c != "{" else i
        elif c in "qQ":
            i = blanks(i)
            while i < n and s[i].isdigit():
                i += 1
            i = ends(i)
        elif c in "btT:":
            while i < n and s[i] not in ";\n":
                i += 1
        elif c in "aicrR":
            i = line_end(i)                       # text or a file it reads
        elif c in "sy":
            if i >= n or s[i] in "\\\n":
                return False
            d = s[i]
            i = part(i + 1, d, c == "s")
            if i < 0:
                return False
            i = part(i, d, False)
            if i < 0:
                return False
            while c == "s" and i < n and s[i] in "gpIiMm0123456789":
                i += 1
            i = ends(i)
        else:
            return False                          # w, W, e, or unknown
        if i < 0:
            return False
    return True


_SED_SHORT_FLAGS = frozenset("nErsuzb")
_SED_LONG_FLAGS = frozenset({"--quiet", "--silent", "--regexp-extended",
                             "--null-data", "--zero-terminated", "--separate",
                             "--unbuffered", "--posix", "--debug", "--sandbox",
                             "--follow-symlinks"})


def _sed_strict(args, bsd):
    """Strict reading of a sed command line, getopt style: ``(ok, in_place,
    suffix, scripts, files)``. ``ok`` is False for anything not fully
    understood (``-f`` script files, unknown or ambiguous options, a script
    whose text depends on a variable or substitution)."""
    bad = (False, False, None, [], [])
    args = [_expand_long(a, _SED_LONG) for a in args]
    scripts, files, in_place, suffix, has_e, k = [], [], False, None, False, 0
    while k < len(args):
        a = args[k]
        if a == "--":
            files += args[k + 1:]
            break
        if a.startswith("--"):
            name, eq, value = a.partition("=")
            if name == "--in-place":
                in_place, suffix = True, value or None
            elif name == "--expression":
                if not eq:
                    if k + 1 >= len(args):
                        return bad
                    k += 1
                    value = args[k]
                scripts.append(value)
                has_e = True
            elif name == "--line-length":
                k += 0 if eq else 1
            elif name not in _SED_LONG_FLAGS:
                return bad
        elif a.startswith("-") and a != "-":
            for j in range(1, len(a)):
                letter = a[j]
                if letter in _SED_SHORT_FLAGS:
                    continue
                if letter == "l" and bsd:
                    continue
                rest = a[j + 1:]
                if letter == "i":
                    in_place = True
                    if rest:
                        suffix = rest
                    elif bsd and k + 1 < len(args):
                        k += 1
                        suffix = args[k] or None
                    break
                if letter in "el":
                    if not rest:
                        if k + 1 >= len(args):
                            return bad
                        k += 1
                        rest = args[k]
                    if letter == "e":
                        scripts.append(rest)
                        has_e = True
                    break
                return bad                     # -f, or unknown
        else:
            files.append(a)
        k += 1
    if not has_e:
        if not files:
            return bad
        scripts.append(files.pop(0))
    if any(getattr(x, "expands", False) or getattr(x, "subst", False) for x in scripts):
        return bad
    return True, in_place, suffix, scripts, files


_AWK_STRING = re.compile(r'"(?:[^"\\]|\\.)*"')


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
    code = _AWK_STRING.sub('""', program)
    safe = len(program) <= 2000 and not _AWK_UNSAFE.search(code)
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
        return bool(_sort_outputs(args))
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
_DEVICES = frozenset({"/dev/null", "/dev/stdout", "/dev/stderr", "/dev/tty",
                      "/dev/zero", "/dev/full"})


def _harmless_target(op, target):
    """Redirect targets that write no file: /dev/*, fd dups, closing."""
    return (target in _DEVICES
            or (op in (">&", "<&") and (target.isdigit() or target == "-"))
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
    if _dangerous_env(stage):
        return False
    if not name:
        return True
    if _writes_via_option(name, args) or _runs_programs(stage, name, args):
        return False
    if name.startswith("git "):
        return name[4:] in _READ_ONLY_GIT
    if name in ("sed", "awk", "json.tool"):
        return _is_viewer(name, args)
    return name in _READ_ONLY


def final_cwds(command, cwd, home=None):
    """The directories ``command`` may end in when run from ``cwd`` (None in
    the set: somewhere unknown)."""
    stages = parse(command, cwd, home)
    return {None} if stages is None else set(stages.final_cwds)


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
    uncertain = stage.cwd_alt is not None and stage.cwd_alt != cwd
    for group in (content, listed):
        out = set()
        for p in group:
            if p.startswith("\0"):
                if not uncertain:
                    out |= _repo_candidates(p[1:], cwd)
            elif ":" not in p or p.startswith("/") or p.startswith("."):
                if uncertain and not p.startswith("/") and not p.startswith("~"):
                    continue
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

_COPY_VALUE_OPTS = {
    "install": ("-m", "--mode", "-o", "--owner", "-g", "--group", "-S", "--suffix"),
    "cp": ("-S", "--suffix"),
    "rsync": ("-e", "--rsh", "--exclude", "--include", "--filter", "-f",
              "--exclude-from", "--include-from", "--files-from", "--chmod",
              "--chown", "--rsync-path", "--log-file", "--backup-dir",
              "--suffix", "--temp-dir", "-T", "--partial-dir", "--compare-dest",
              "--copy-dest", "--link-dest", "--port", "--password-file"),
}


# the options strict mode understands for each copier; anything else (a
# backup, a bundled -t, an abbreviation, a remote path) can't be placed
_COPY_LONG = {
    "cp": {"--recursive", "--force", "--verbose", "--archive", "--preserve",
           "--no-preserve", "--dereference", "--no-dereference", "--no-clobber",
           "--interactive", "--no-target-directory", "--link", "--symbolic-link",
           "--sparse", "--reflink", "--remove-destination", "--one-file-system",
           "--strip-trailing-slashes", "--attributes-only", "--update",
           "--target-directory"},
    "mv": {"--force", "--verbose", "--no-clobber", "--interactive",
           "--no-target-directory", "--update", "--target-directory",
           "--strip-trailing-slashes"},
    "install": {"--mode", "--owner", "--group", "--directory", "--verbose",
                "--compare", "--preserve-timestamps", "--strip",
                "--target-directory", "--no-target-directory"},
    "rsync": {"--archive", "--recursive", "--verbose", "--compress", "--delete",
              "--progress", "--checksum", "--times", "--perms", "--human-readable",
              "--exclude", "--include", "--quiet", "--links", "--update",
              "--itemize-changes", "--stats", "--partial", "--dry-run",
              "--ignore-existing", "--no-perms", "--no-owner", "--no-group",
              "--omit-dir-times", "--delete-after", "--delete-excluded"},
}
_COPY_SHORT = {"cp": "rRafpvnidLHPlsuxTZ", "mv": "fvniuTZ", "install": "vCcpsDdT",
               "rsync": "avrzhPqcutlpgoDHSxWn"}
_COPY_VALUED = {"-t", "-m", "-o", "-g", "--target-directory", "--mode",
                "--owner", "--group", "--exclude", "--include"}


def _copy_options_ok(name, args):
    """Strict: every option of a cp/mv/install/rsync is one whose effect on
    the destination is modelled."""
    k = 0
    while k < len(args):
        a = args[k]
        if a == "--":
            return name != "rsync" or not any(":" in x for x in args[k + 1:])
        if a.startswith("--"):
            opt, eq, _ = a.partition("=")
            if opt not in _COPY_LONG[name]:
                return False
            k += 2 if opt in _COPY_VALUED and not eq else 1
            continue
        if a.startswith("-") and a != "-":
            if a in _COPY_VALUED and (a == "-t" or name == "install"):
                k += 2
                continue
            if not set(a[1:]) <= set(_COPY_SHORT[name]):
                return False
        elif name == "rsync" and ":" in a:
            return False                   # a remote path
        k += 1
    return True


def _copy_targets(name, args, cwd, home):
    """``(dst, alt)`` per file a ``cp`` writes (alt: the other reading of an
    ambiguous ``cp a b``, b possibly a directory). Glob or variable sources
    into a directory can't be named: nothing is owed for them."""
    target_dir, ops, k = None, [], 0
    takes_value = _COPY_VALUE_OPTS.get(name, ())
    while k < len(args):
        a = args[k]
        if a in ("-t", "--target-directory") and k + 1 < len(args):
            target_dir = args[k + 1]
            k += 2
            continue
        if a in takes_value:
            k += 2                                 # install -m 644
            continue
        if a.startswith("--target-directory="):
            target_dir = a.split("=", 1)[1]
        elif a == "--parents" or name == "rsync" and a in ("-R", "--relative"):
            return [(UNPLACED + name + " " + a, None)]   # keeps source dirs
        elif not a.startswith("-"):
            ops.append(a)
        k += 1
    if target_dir is None:
        if len(ops) < 2:
            return []
        *sources, target = ops
    else:
        sources, target = ops, target_dir
    if set(target) & set("*?[$`"):
        return [(UNPLACED + name + " into " + target, None)]   # which file?
    out = []
    for s in sources:
        if set(s) & set("*?[$`"):
            out.append((UNPLACED + name + " " + s, None))   # which files landed?
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


def _writes(stage, strict=False):
    """``(path, alt, kind)`` per file a stage writes.

    ``kind`` says what ``alt`` means: ``"dir"`` for an ambiguous destination
    (``cp a b``: b or b/a; only one can be a readable file, so either may
    pay), ``"cwd"`` for an uncertain directory (after ``cd x;`` the file may
    be in either place, both real, so strict mode owes both). Stdout/both
    redirects, ``tee``, ``sed -i``, ``sort -o``, ``awk -i inplace``,
    ``perl``/``ruby -i``, ``truncate``, ``dd of=``, ``sponge``, ``curl -o``,
    ``wget -O``, ``git checkout -- f``/``git restore f``, ``cp``/``install``/
    ``rsync`` destinations; with ``strict``, redirects on any descriptor too
    (``2> err.log``). A target that can't be placed (a variable, a glob, an
    unknown directory, a patch's files, a download's name) comes back as
    ``UNPLACED + text``: no read can pay it.
    """
    name, args, cwd = _program(stage)
    out, unplaced = [], []
    for fd, op, target in stage.redirects:
        if op in (">", ">>", ">|", "&>", "&>>", ">&", "<>") \
                and (fd in ("", "1") or strict) and not _harmless_target(op, target):
            out.append(target)
    if name == "tee":
        out += _split_args("tee", args)[1]
    elif name == "sed" and strict:
        # getopt-style, the way this machine's sed reads `-i .x`
        ok, in_place, suffix, scripts, files = _sed_strict(args, _BSD_SED)
        if not ok or not all(map(_sed_safe, scripts)):
            unplaced.append("sed command not fully understood")
        if in_place:
            out += files
            if suffix and "*" in suffix:
                unplaced.append("sed backup " + suffix)
            elif suffix:
                out += [f + suffix for f in files]
    elif name == "sed":
        in_place, scripts, files = _sed_parts(args)
        if in_place:
            out += files
        if not _is_viewer(name, args) and not all(map(_sed_safe, scripts)):
            unplaced.append("sed script that writes or runs commands")
    elif name == "sort":
        out += _sort_outputs(args)
    elif name == "awk" and _awk_in_place(args):
        out += _split_args("awk", args)[1][1:]
    elif name in ("perl", "ruby") and _perl_in_place(args):
        out += _perl_files(args)
    elif name == "truncate":
        opts, ops = _split_args("truncate", [a for a in args])
        out += [o for o in ops if not _looks_like_size(o, args)]
    elif name == "dd":
        out += [a[3:] for a in args if a.startswith("of=")]
    elif name == "sponge":
        out += _split_args("sponge", args)[1]
    elif name == "curl":
        opts, _ = _bundled_options(args, _CURL_VALUE_LETTERS, _CURL_FILE_OPTS +
                                   ("--output", "--config", "--output-dir"))
        for opt, value in opts:
            if opt in ("-o", "--output") and value not in (None, "-"):
                out.append(value)
            elif opt in ("-O", "--remote-name", "--remote-name-all"):
                unplaced.append("curl -O")
            elif strict and opt in ("-K", "--config", "--output-dir"):
                unplaced.append("curl " + opt)
            elif strict and opt in _CURL_FILE_OPTS and value is not None:
                out.append(value)
        if strict and _abbreviates(opts, _CURL_FILE_OPTS + _CURL_OTHER_WRITERS):
            unplaced.append("curl option abbreviation")
    elif name == "wget":
        opts, ops = _bundled_options(args, _WGET_VALUE_LETTERS, _WGET_LONG_VALUES)
        named = [v for o, v in opts if o in ("-O", "--output-document") and v]
        out += [v for v in named if v != "-"]             # -O- is stdout
        if not named and ops:
            unplaced.append("wget download")
        if strict:
            out += [v for o, v in opts if v and o in (
                "-o", "-a", "--output-file", "--append-output", "--save-cookies",
                "--rejected-log")]
            if any(o in ("-e", "--execute", "--warc-file", "-i", "--input-file")
                   for o, _ in opts) or _abbreviates(opts, _WGET_LONG_VALUES):
                unplaced.append("wget options")
    elif name == "touch" and strict:
        out += _bundled_options(args, set("rdt"), ("--reference", "--date"))[1]
    elif name == "patch":
        unplaced.append("patch")           # the diff names the files
    elif name in ("git checkout", "git restore"):
        opts, ops = _split_args(name, args)
        flags = {f.split("=", 1)[0] for f, _ in opts}
        if name == "git restore" and "--staged" in flags and "--worktree" not in flags \
                and "-W" not in flags:
            pass                           # index only
        elif name == "git restore" or "--" in args:
            paths = args[args.index("--") + 1:] if "--" in args else ops
            out += paths
    placed = []
    for p in out:
        if _harmless_target(">", p):
            continue
        if p.startswith("/dev/") or p == "-":
            placed.append((UNPLACED + p, None, None))      # stdin, a fd, shm
            continue
        if set(p) & set("*?["):
            placed.append((UNPLACED + p, None, None))      # a glob: which files?
            continue
        r = resolve(p, cwd, stage.home)
        alt = resolve(p, stage.cwd_alt, stage.home) if stage.cwd_alt is not None else None
        if r:
            placed.append((r, alt if alt and alt != r else None, "cwd"))
        elif alt:
            # the directory it moved to is unknown: owe the fallback place and
            # an unpayable entry (it may have been written somewhere else)
            placed.append((alt, None, None))
            placed.append((UNPLACED + p, None, None))
        else:
            placed.append((UNPLACED + p, None, None))
    if name in ("cp", "install", "rsync"):
        placed += [(d, a, "dir") for d, a in _copy_targets(name, args, cwd, stage.home)]
    if strict and name in _COPY_LONG and not _copy_options_ok(name, args):
        unplaced.append(name + " options not fully understood (backups, targets)")
    placed += [(UNPLACED + u, None, None) for u in unplaced]
    return placed


# curl options that write a file of their own (headers, cookies, traces)
_CURL_FILE_OPTS = ("-D", "--dump-header", "-c", "--cookie-jar", "--trace",
                   "--trace-ascii", "--libcurl", "--stderr", "--etag-save",
                   "--hsts", "--alt-svc")


_CURL_VALUE_LETTERS = frozenset("AbcCdDeEFhHKmoPQrtTuUwxXyYz")
_WGET_VALUE_LETTERS = frozenset("oaeiBOtTwQPlARDXIU")
_WGET_LONG_VALUES = ("--output-document", "--output-file", "--append-output",
                     "--save-cookies", "--rejected-log", "--execute",
                     "--warc-file", "--input-file")


_CURL_OTHER_WRITERS = ("--output", "--config", "--output-dir", "--remote-name",
                       "--remote-name-all", "--remote-header-name")


def _abbreviates(opts, writers):
    """A long option that is a shortened spelling of a writing one
    (``--dump-he``): the programs accept unambiguous prefixes."""
    return any(o.startswith("--") and o not in writers and len(o) > 2 and
               any(w.startswith(o) for w in writers) for o, _ in opts)


def _bundled_options(args, value_letters, long_values):
    """``([(option, value)], operands)`` read the getopt way: in a short
    bundle (``-sSLo f``, ``-qO-``) the first letter that takes a value
    takes the rest of the bundle, or the next word."""
    opts, ops, k = [], [], 0
    while k < len(args):
        a = args[k]
        if a == "--":
            ops += args[k + 1:]
            break
        if a.startswith("--"):
            name, eq, value = a.partition("=")
            if eq:
                opts.append((name, value))
            elif name in long_values and k + 1 < len(args):
                opts.append((name, args[k + 1]))
                k += 1
            else:
                opts.append((name, None))
        elif a.startswith("-") and len(a) > 1:
            for j, letter in enumerate(a[1:], 1):
                if letter in value_letters:
                    rest = a[j + 1:]
                    if not rest and k + 1 < len(args):
                        rest = args[k + 1]
                        k += 1
                    opts.append(("-" + letter, rest))
                    break
                opts.append(("-" + letter, None))
        else:
            ops.append(a)
        k += 1
    return opts, ops


def _option_values(args, names):
    out = []
    for k, a in enumerate(args):
        for n in names:
            if a == n and k + 1 < len(args):
                out.append(args[k + 1])
            elif n.startswith("--") and a.startswith(n + "="):
                out.append(a.split("=", 1)[1])
            elif not n.startswith("--") and a.startswith(n) and len(a) > len(n):
                out.append(a[len(n):])
    return out


def _looks_like_size(operand, args):
    """truncate's -s/-r values are not files."""
    k = args.index(operand) if operand in args else -1
    return k > 0 and args[k - 1] in ("-s", "--size", "-r", "--reference")


# prefix of a write the parser could not place (``cd $DIR && sed -i ... f``,
# ``echo x > $OUT``): no read can pay it, which strict mode wants
UNPLACED = "<unplaced write> "


def _awk_in_place(args):
    """gawk ``-i inplace`` / ``-iinplace`` / ``--include=inplace``."""
    for k, a in enumerate(args):
        if a in ("-i", "--include") and k + 1 < len(args) and "inplace" in args[k + 1]:
            return True
        if (a.startswith("-i") or a.startswith("--include=")) and "inplace" in a:
            return True
    return False


def _perl_in_place(args):
    """``perl -i``, ``-pi``, ``-i.bak``, ``-pi -e ...``."""
    return any(a.startswith("-") and not a.startswith("--") and "i" in a[1:]
               and not a.startswith("-I") for a in args if a != "-")


def _perl_files(args):
    """Files after perl's options and ``-e``/``-E`` programs."""
    files, k, has_e = [], 0, False
    while k < len(args):
        a = args[k]
        if a in ("-e", "-E") and k + 1 < len(args):
            has_e = True
            k += 2
            continue
        if a.startswith("-"):
            has_e = has_e or a.endswith("e") and len(a) > 1 and not a.startswith("--")
            if a.endswith("e") and k + 1 < len(args) and not a.startswith("--"):
                k += 2                     # bundled -pie 'program'
                continue
            k += 1
            continue
        files.append(a)
        k += 1
    return files if has_e else files[1:]   # without -e the first is the script


def _sort_outputs(args):
    """``sort -o F``, ``-uo F``, ``-oF``, ``--output=F``, ``--output F``."""
    args = [_expand_long(a, ("--output",)) if a.startswith("--o") else a
            for a in args]
    out, k = [], 0
    while k < len(args):
        a = args[k]
        if a.startswith("--output="):
            out.append(a.split("=", 1)[1])
        elif a == "--output" and k + 1 < len(args):
            out.append(args[k + 1])
            k += 1
        elif a.startswith("-") and not a.startswith("--") and "o" in a[1:]:
            rest = a[a.index("o", 1) + 1:]
            if rest:
                out.append(rest)
            elif k + 1 < len(args):
                out.append(args[k + 1])
                k += 1
        k += 1
    return out


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
    # a no-clobber or interactive mv may not move anything
    certain = getattr(stage, "certain", True) and not any(
        a in ("-n", "-i", "--no-clobber", "--interactive") or a.startswith("--update")
        or (a.startswith("-") and not a.startswith("--") and set(a[1:]) & set("niu"))
        for a in args)
    moves = []
    for s in sources:
        if set(s) & set("$`"):
            continue
        if set(s) & set("*?["):
            src, dst = resolve(s, cwd, stage.home), resolve(target, cwd, stage.home)
            if src and dst:
                moves.append((src, dst, "*glob*", certain))
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
            moves.append((src, dst, alt, certain))
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
    certain = getattr(stage, "certain", True) and not (
        _short_flags(opts) & set("iI") or "--interactive" in flags)
    out = []
    for o in ops:
        r = resolve(o, cwd, stage.home)
        if r:
            out.append((r, recursive, certain))
    return out


def _substitutions(stage, process=True):
    """The command texts inside a stage's substitutions: ``$(...)``,
    backticks, and (with ``process``) ``<(...)``/``>(...)``, in words and
    redirect targets."""
    texts = [str(w) for w in getattr(stage, "raw", stage.argv)] + \
        [t for _, _, t in stage.redirects]
    out = []
    for text in texts:
        i = 0
        while i < len(text):
            if text.startswith("$(", i) or process and (
                    text.startswith("<(", i) or text.startswith(">(", i)):
                try:
                    j = _read_balanced(text, i + 2)
                except ValueError:
                    break
                out.append(text[i + 2:j - 1])
                i = j
            elif text[i] == "`":
                j = text.find("`", i + 1)
                if j < 0:
                    break
                out.append(text[i + 1:j])
                i = j + 1
            else:
                i += 1
    return out


# git subcommands that print or change only git's own records, not files
_QUIET_GIT = frozenset({"branch", "remote", "describe", "rev-list", "tag",
                        "shortlog", "reflog", "cat-file", "ls-remote",
                        "merge-base", "name-rev", "for-each-ref", "show-ref",
                        "symbolic-ref", "check-ignore", "count-objects"})
# environment variables that make a later program run something else
_RISKY_ENV = re.compile(
    r"^(?:GIT_\w*|\w*PAGER|LESS\w*|PATH|LD_\w+|DYLD_\w+|\w*_COMMAND|EDITOR|"
    r"VISUAL|BASH_ENV|ENV|PYTHON\w*|NODE_OPTIONS|PERL5\w*|RUBYOPT|"
    r"PROMPT_COMMAND|SHELLOPTS|BASHOPTS|IFS)=")


def _dangerous_env(stage):
    """``GIT_EXTERNAL_DIFF=x git diff``, ``PAGER=./x git log``, ``export
    GIT_CONFIG_...``: the environment makes a program run another."""
    raw = [str(w) for w in getattr(stage, "raw", stage.argv)]
    k = 0
    while k < len(raw) and _ASSIGNMENT.match(raw[k]):
        if _RISKY_ENV.match(raw[k]):
            return True
        k += 1
    if k < len(raw) and raw[k] in ("export", "declare", "typeset", "readonly",
                                   "local", "env"):
        return any(_RISKY_ENV.match(w.lstrip("-")) for w in raw[k + 1:])
    return False


def _known_program(stage):
    """Whether everything the stage's program itself writes is modelled
    (redirects and substitutions are parsed separately): a read-only
    program, a known writer, or a builtin that writes nothing."""
    name, args, _ = _program(stage)
    if _runs_programs(stage, name, args) or _dangerous_env(stage):
        return False
    if not name:
        return True
    if name == "exec":
        return not args                  # `exec 3>f` only redirects
    if name == "command":
        return args[:1] in (["-v"], ["-V"])    # describes, runs nothing
    if name in _NO_WRITES or name in _KNOWN_EFFECTS or name in ("curl", "wget"):
        return True             # (their file options are parsed by _writes)
    if _writes_via_option(name, args):
        return False
    if name == "git config":
        return any(a in ("--get", "--get-all", "--get-regexp", "--list", "-l")
                   for a in args)
    if name.startswith("git "):
        return name[4:] in _READ_ONLY_GIT or name[4:] in _QUIET_GIT
    if name == "awk":
        # anything that could redirect, pipe, run or load: not a viewer
        program = (_split_args("awk", args)[1] or [""])[0]
        return _is_viewer(name, args) and not re.search(
            r"[>|@]|system|getline|close|fflush|ENVIRON|PROCINFO", program)
    if name == "json.tool":
        return _is_viewer(name, args)
    if name == "uniq":
        return len(_bundled_options(args, set("fsw"), ("--skip-fields",
                   "--skip-chars", "--check-chars"))[1]) <= 1
    if name == "find":
        return not any(a in _FIND_ACTIONS for a in args)
    return name in _READ_ONLY


def _runs_programs(stage, name, args):
    """Read-only programs told to run something else: ``git -c`` (hooks,
    fsmonitor, pagers), ``git grep -O``, ``rg --pre``, ``file -C``."""
    argv = list(stage.argv)
    if argv and _system_name(argv[0]) == "git":
        for a in argv[1:]:
            if not a.startswith("-"):
                break
            if a.startswith("-c") or a.startswith("--config-env") \
                    or a.startswith("--exec-path"):
                return True
    if name == "git grep":
        return any(a.startswith("-O") or a.startswith("--open-files-in-pager")
                   for a in args)
    if name == "rg":
        return any(a.startswith("--pre") for a in args)
    if name == "file":
        return any(a == "-C" or a == "--compile" for a in args)
    return False


_FIND_ACTIONS = frozenset({"-exec", "-execdir", "-ok", "-okdir", "-delete",
                           "-fprint", "-fprint0", "-fprintf", "-fls"})


def _inner_writes(stage, cwd, home, strict=False, depth=0):
    """Files written by commands inside a stage's substitutions (``x=$(sed
    -i ... f)``). Their output was captured, not shown, so only their writes
    matter; nested substitutions are followed a few levels deep. Process
    substitutions (``> >(tee log)``) are usually logging, so only strict
    mode owes what they write."""
    found = []
    if depth > 4:
        return [(UNPLACED + "substitutions nested too deep", None, None)]
    for text in _substitutions(stage, process=strict):
        inner = parse(text, cwd, home)
        for s in inner or ():
            found += _writes(s, strict)
            if strict and not _known_program(s):
                name = _program(s)[0]
                found.append((UNPLACED + "substituted command " + name, None, None))
            if s.subst:
                found += _inner_writes(s, cwd, home, strict, depth + 1)
    return found


# builtins and programs that change no file themselves (their redirects are
# still parsed); loop and case headers only name words
_NO_WRITES = frozenset({
    "true", "false", ":", "export", "unset", "set", "shift", "local", "declare",
    "typeset", "readonly", "exit", "return", "break", "continue", "sleep",
    "wait", "read", "for", "select", "case", "in", "shopt", "alias",
    "unalias", "hash", "type", "umask", "ulimit", "jobs", "kill", "exec", "[[",
    "git add", "git fetch"})
_BRACES = re.compile(r"\{[^{}\s]*(?:,|\.\.)[^{}\s]*\}")

_KNOWN_EFFECTS = frozenset({"tee", "sed", "mv", "git mv", "rm", "git rm",
                            "unlink", "rmdir", "mkdir", "touch", "cp", "sort",
                            "truncate", "sponge", "install", "rsync"})
# note: awk -i inplace and perl -i writes are recorded, but those programs
# stay opaque (a script can do anything else too)


def effects(command, cwd="", output="", home=None, strict=False):
    """Ordered effects of a (possibly mutating) command:
    ``("write", path, alt, kind)``, ``("move", src, dst, alt, certain)``,
    ``("remove", path, recursive, certain)``, ``("read", path)`` for content a stage
    showed the agent, and ``("opaque", argv)`` for a stage whose writes can't be
    known (a script, a build, a substitution). None when the command can't
    be parsed."""
    stages = parse(command, cwd, home)
    if stages is None:
        return None
    out = []
    if strict and stages.loop_cd:
        out.append(("write", UNPLACED + "cd inside a loop", None, None))
    if strict and stages.defines_function:
        out.append(("write", UNPLACED + "a shell function", None, None))
    pipes, credited = _credited_pipelines(stages)
    printed = bool((output or "").strip())
    sole = _sole_producer(pipes)
    for k, pipe in enumerate(pipes):
        reads = set()
        for j, stage in enumerate(pipe):
            name, args, _ = _program(stage)
            if not _known_program(stage):
                out.append(("opaque", tuple(name.split()) + tuple(args)))
            if strict and not _stage_read_only(stage) and any(
                    _BRACES.search(str(w)) for w in stage.argv):
                # cp f{,.bak}: the operands the program sees aren't these
                out.append(("write", UNPLACED + "brace expansion", None, None))
            out += [("write",) + w for w in _writes(stage, strict)]
            if stage.subst:
                out += [("write",) + w for w in _inner_writes(stage, stage.cwd,
                                                               stage.home, strict)]
            if strict and stage.sep == "&" and not _stage_read_only(stage):
                # a background job keeps writing after this call returns
                out.append(("write", UNPLACED + ("background job" if not getattr(
                    stage, "depth", 0) else "background job in a subshell"), None, None))
            if name == "wait" and not args and stage.sep != "&" and stage.certain \
                    and getattr(stage, "depth", 0) == 0 and not stage.piped \
                    and not stage.piped_in:
                # every job started so far has finished
                out = [e for e in out
                       if e[:2] != ("write", UNPLACED + "background job")]
            out += [("move",) + m for m in _moves(stage)]
            out += [("remove",) + r for r in _removes(stage)]
            if _stage_read_only(stage):
                reads |= _stage_reads(stage, output, pipe is sole)[0]
        if k in credited and printed and _pipeline_shows(pipe):
            out += [("read", p) for p in sorted(reads)]
    return out
