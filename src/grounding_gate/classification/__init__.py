"""Task classification for review — categorize each finished turn against a
written standard, optionally audited by a classifier model.

The gate decides whether a turn's claims are backed. This layer answers a
different question for the humans who review agent work afterwards: *what
kind of task was this, how risky is it, and what should a reviewer check?*

Three parts:

* **The standard** (:data:`STANDARD`, id ``GG-TASK-1``): a fixed, versioned
  set of task categories and risk flags. Each has a definition, a base risk
  tier and a reviewer checklist, so two reviews of the same kind of task
  check the same things. Pass your own :class:`TaskStandard` to change it.
* **The structural classifier** (:func:`classify_task`): zero-token rules
  over what the turn actually did (the files it changed, the commands it
  ran, the gate's verdict). Deterministic, so the same turn always gets the
  same category.
* **The audit** (:class:`ClassificationAuditor`, e.g.
  :class:`~grounding_gate.classification.llm.LLMClassificationAuditor`): a
  classifier model reviews the structural classification against the same
  standard. Like the ``verify_with`` tier it can only escalate: it may add
  flags and raise the risk tier, never remove or lower them, and any
  disagreement on the category (or an auditor failure) marks the task for
  human review. The structural category stays the recorded one; the
  auditor's is kept beside it.

Like :mod:`grounding_gate.verifiers`, this module is stdlib-only. The model
auditor lives in :mod:`grounding_gate.classification.llm` and imports its SDK
lazily.
"""

import posixpath
import re
from dataclasses import asdict, dataclass, field
from typing import Optional, Protocol, runtime_checkable

from ..shell import is_read_only

RISK_LEVELS = ("low", "medium", "high")


def _max_risk(*levels):
    return max(levels, key=RISK_LEVELS.index) if levels else "low"


# ------------------------------------------------------------------ standard

@dataclass(frozen=True)
class Category:
    """One kind of task: what it is, how risky by default, what to check."""
    id: str
    name: str
    definition: str
    risk: str
    checklist: tuple


@dataclass(frozen=True)
class Flag:
    """A risk property a task of any category can have."""
    id: str
    definition: str
    risk: str
    checklist: tuple


@dataclass(frozen=True)
class TaskStandard:
    """A versioned review standard: categories (exactly one is the task's
    primary), flags (any number), and the order that picks the primary
    category when a task touches several (earlier wins)."""
    id: str
    version: str
    categories: tuple
    flags: tuple

    def category(self, category_id):
        return next(c for c in self.categories if c.id == category_id)

    def flag(self, flag_id):
        return next(f for f in self.flags if f.id == flag_id)

    @property
    def category_ids(self):
        return tuple(c.id for c in self.categories)

    @property
    def flag_ids(self):
        return tuple(f.id for f in self.flags)

    def describe(self):
        """The standard as plain text (what a model auditor is shown)."""
        lines = ["Review standard %s v%s" % (self.id, self.version), "",
                 "Categories (a task has exactly one primary category; when "
                 "it touches several, the first listed here wins):"]
        for c in self.categories:
            lines.append("- %s (%s, base risk %s): %s" % (c.id, c.name, c.risk,
                                                          c.definition))
        lines += ["", "Risk flags (a task may have any number):"]
        for f in self.flags:
            lines.append("- %s (risk %s): %s" % (f.id, f.risk, f.definition))
        return "\n".join(lines)


STANDARD = TaskStandard(
    id="GG-TASK-1",
    version="1.0",
    categories=(
        Category(
            "operations", "Operations",
            "Acted outside the working tree: published, pushed, deployed, "
            "installed or removed packages, or wrote through a remote or "
            "untrusted tool.",
            "high",
            ("Confirm each external action was requested and is reversible "
             "or intended to be permanent",
             "Check the target (remote, registry, environment) is the right one",
             "Confirm nothing was published or pushed before it was verified")),
        Category(
            "config", "Configuration",
            "Changed build, CI, dependency, environment or tool configuration.",
            "medium",
            ("Check the configuration still parses and the build or CI runs",
             "Look for widened permissions, disabled checks or changed "
             "versions")),
        Category(
            "code", "Code change",
            "Changed source code.",
            "medium",
            ("Read the diff of every changed source file",
             "Check tests cover the change and were run after it",
             "Confirm the claimed behaviour matches the code")),
        Category(
            "data", "Data change",
            "Changed data files, fixtures, datasets or binary assets.",
            "medium",
            ("Check the data's format and size are what consumers expect",
             "Confirm no real or personal data was added by mistake")),
        Category(
            "tests", "Test change",
            "Changed only tests (and possibly docs).",
            "low",
            ("Check no assertion was weakened, skipped or deleted to get green",
             "Confirm the tests were run after the change")),
        Category(
            "docs", "Documentation",
            "Changed only documentation.",
            "low",
            ("Check the docs describe what the code actually does",)),
        Category(
            "execution", "Execution",
            "Ran commands (tests, builds, scripts) but changed no files.",
            "low",
            ("Check the reported results match the commands' output",)),
        Category(
            "inquiry", "Inquiry",
            "Read or searched, changed nothing, and answered.",
            "low",
            ("Check the answer is supported by what was read this turn",)),
    ),
    flags=(
        Flag("unverified_exit",
             "The gate let the turn end through its escape valve: its claims "
             "are not backed by a qualifying read.",
             "high",
             ("Treat every claim in the answer as unverified until you check it",)),
        Flag("external_effect",
             "Had an effect outside the working tree (network write, push, "
             "publish, deploy, remote tool write).",
             "high",
             ("Verify each external effect at its destination",)),
        Flag("destructive",
             "Deleted or overwrote data in bulk or irreversibly (recursive "
             "delete, hard reset, force push, clean).",
             "high",
             ("Confirm nothing needed was lost; check backups or history",)),
        Flag("security_sensitive",
             "Touched credentials, secrets, keys, permissions or auth code.",
             "high",
             ("Check no secret was added, printed or committed",
              "Review permission and auth changes line by line")),
        Flag("ci_change",
             "Changed CI or release automation.",
             "high",
             ("Check the pipeline still gates what it gated before",)),
        Flag("dependency_change",
             "Added, removed or upgraded dependencies.",
             "medium",
             ("Check new or upgraded packages are expected and trusted",
              "Check lockfiles changed consistently with manifests")),
        Flag("unmodelled_commands",
             "Ran commands whose file effects the gate could not model "
             "(strict reads).",
             "medium",
             ("Look for files those commands may have changed",)),
        Flag("subagent_changes",
             "Subagents changed files during the turn.",
             "medium",
             ("Review the subagents' changes as well as the agent's",)),
        Flag("large_change",
             "Changed many files at once.",
             "medium",
             ("Review file by file; look for unintended edits",)),
        Flag("failed_calls",
             "Some tool calls failed and may have left partial changes.",
             "medium",
             ("Check files touched by failed calls for partial edits",)),
    ),
)


# ------------------------------------------------------------------ records

@dataclass
class TaskRecord:
    """What one turn did, as the wiring layer saw it (the adapter builds this
    at the end of each turn).

    ``calls``: ``[{"tool", "input", "failed", "subagent"}]`` in order.
    ``changed_files``: paths the turn changed (absolute where known).
    ``verdict``: ``"verified"``, ``"grounded"``, ``"unverified"`` or
    ``"conversational"``. ``unplaced``: obligations no read could pay
    (strict reads' ``<unplaced write>`` entries).
    """
    calls: list = field(default_factory=list)
    changed_files: list = field(default_factory=list)
    verdict: str = "conversational"
    claim_type: str = "none"
    unplaced: list = field(default_factory=list)

    @property
    def commands(self):
        out = []
        for call in self.calls:
            inp = call.get("input")
            if isinstance(inp, dict) and isinstance(inp.get("command"), str):
                out.append(inp["command"])
        return out


@dataclass
class TaskClassification:
    """The result reviewers see. ``evidence`` maps each category and flag
    to the concrete things that produced it; ``audit`` is filled in by an
    auditor (see :func:`apply_audit`)."""
    standard: str
    category: str
    secondary: list
    flags: list
    risk: str
    checklist: list
    evidence: dict
    needs_human_review: bool = False
    audit: Optional[dict] = None

    def to_dict(self):
        return asdict(self)


@dataclass
class AuditResult:
    """A classifier model's review of a structural classification."""
    category: str
    flags: list
    risk: str
    agrees: bool
    rationale: str = ""
    auditor: str = ""


@runtime_checkable
class ClassificationAuditor(Protocol):
    """``audit(record, classification, standard)`` returns an
    :class:`AuditResult`, or ``None`` when it could not audit (an API
    error, a refusal, an unparseable answer). ``None`` marks the task for
    human review; it never lowers anything."""

    def audit(self, record, classification, standard):  # pragma: no cover
        ...


# ------------------------------------------------------------- file rules

_DOC_EXT = {".md", ".rst", ".txt", ".adoc", ".mdx"}
_DOC_NAMES = {"readme", "changelog", "license", "contributing", "authors",
              "notice", "history", "code_of_conduct", "security"}
_DATA_EXT = {".csv", ".tsv", ".jsonl", ".ndjson", ".parquet", ".arrow", ".db",
             ".sqlite", ".sqlite3", ".xlsx", ".xls", ".png", ".jpg", ".jpeg",
             ".gif", ".svg", ".webp", ".ico", ".pdf", ".pkl", ".npy", ".npz",
             ".h5", ".bin", ".zip", ".tar", ".gz"}
_CONFIG_EXT = {".toml", ".ini", ".cfg", ".conf", ".yaml", ".yml", ".json",
               ".properties", ".env", ".lock", ".plist", ".xml", ".gradle",
               ".tf", ".tfvars", ".nix", ".editorconfig"}
_CONFIG_NAMES = {"dockerfile", "makefile", "procfile", "gemfile", "rakefile",
                 "vagrantfile", "jenkinsfile", "justfile", "brewfile",
                 ".gitignore", ".gitattributes", ".dockerignore", ".npmrc",
                 ".nvmrc", ".python-version", ".tool-versions", ".env",
                 "go.mod", "go.sum", "requirements.txt", "setup.py",
                 "setup.cfg", "tox.ini", "noxfile.py", "conftest.py",
                 "manage.py", "codeowners"}
_MANIFESTS = re.compile(
    r"(?:^|/)(?:package\.json|package-lock\.json|yarn\.lock|pnpm-lock\.yaml|"
    r"requirements[\w.-]*\.txt|pyproject\.toml|poetry\.lock|uv\.lock|Pipfile(?:\.lock)?|"
    r"setup\.py|setup\.cfg|Cargo\.toml|Cargo\.lock|go\.mod|go\.sum|Gemfile(?:\.lock)?|"
    r"composer\.(?:json|lock)|build\.gradle(?:\.kts)?|pom\.xml|environment\.ya?ml)$",
    re.I)
_CI = re.compile(r"(?:^|/)(?:\.github/workflows/|\.gitlab-ci\.yml$|\.circleci/|"
                 r"Jenkinsfile$|azure-pipelines\.yml$|\.buildkite/|\.travis\.yml$|"
                 r"bitbucket-pipelines\.yml$|\.github/actions/)", re.I)
_SECRETS = re.compile(r"(?:^|/)(?:\.env(?:\.[\w-]+)?|[^/]*secret[^/]*|[^/]*credential[^/]*|"
                      r"[^/]*\.pem|[^/]*\.key|id_[rd]sa[^/]*|[^/]*token[^/]*|"
                      r"\.npmrc|\.pypirc|\.netrc|[^/]*passw[^/]*|[^/]*auth[^/]*|"
                      r"[^/]*permission[^/]*|[^/]*\.p12|[^/]*keystore[^/]*)$", re.I)
_TEST = re.compile(r"(?:^|/)(?:tests?|__tests__|spec|specs|testdata|fixtures)/|"
                   r"(?:^|/)test_[^/]*$|_test\.[^/.]+$|\.(?:test|spec)\.[^/.]+$|"
                   r"(?:^|/)conftest\.py$", re.I)


def file_kind(path):
    """``docs``, ``tests``, ``config``, ``data`` or ``code`` for one path."""
    base = posixpath.basename(path).lower()
    stem, ext = posixpath.splitext(base)
    if _CI.search(path) or _MANIFESTS.search(path):
        return "config"
    if _TEST.search(path):
        return "tests"
    if ext in _DOC_EXT or stem in _DOC_NAMES or "/docs/" in "/" + path.lower():
        return "docs" if ext not in _CONFIG_EXT else "config"
    if base in _CONFIG_NAMES or ext in _CONFIG_EXT or base.startswith(".env"):
        return "config"
    if ext in _DATA_EXT:
        return "data"
    return "code"


# ---------------------------------------------------------- command rules

_EXTERNAL = re.compile(
    r"\b(?:git\s+push|npm\s+publish|yarn\s+(?:npm\s+)?publish|pnpm\s+publish|"
    r"cargo\s+publish|gem\s+push|twine\s+upload|docker\s+push|podman\s+push|"
    r"kubectl\s+(?:apply|delete|create|replace|patch|rollout|scale)|"
    r"helm\s+(?:install|upgrade|uninstall|rollback)|"
    r"terraform\s+(?:apply|destroy|import)|pulumi\s+(?:up|destroy)|"
    r"gh\s+(?:pr|issue|release|repo)\s+(?:create|merge|close|edit|delete|comment)|"
    r"aws\s+\S+\s+(?:put|create|delete|update|deploy|cp|sync|rm)\w*|"
    r"gcloud\s+.*\b(?:deploy|create|delete|update)\b|"
    r"(?:fly|vercel|netlify|heroku)\s+deploy|scp\s|rsync\s+.*\S+:|ssh\s)")
_HTTP_WRITE = re.compile(r"\bcurl\b.*(?:-X\s*(?:POST|PUT|PATCH|DELETE)|--request\s*"
                         r"(?:POST|PUT|PATCH|DELETE)|\s-d\b|\s--data|\s-F\b|\s--form|"
                         r"\s-T\b|--upload-file)|\bwget\b.*--post", re.I)
_DEPENDENCY = re.compile(
    r"\b(?:npm\s+(?:install|i|add|uninstall|remove|update|ci)|"
    r"yarn\s+(?:add|remove|install|upgrade)|pnpm\s+(?:add|remove|install|update|i)|"
    r"pip3?\s+(?:install|uninstall)|python3?\s+-m\s+pip\s+(?:install|uninstall)|"
    r"uv\s+(?:add|remove|pip\s+install|sync)|poetry\s+(?:add|remove|install|update)|"
    r"cargo\s+(?:add|remove|update|install)|go\s+(?:get|install)|go\s+mod\s+tidy|"
    r"bundle\s+(?:install|add|update)|gem\s+install|composer\s+(?:require|install|update)|"
    r"brew\s+install|apt(?:-get)?\s+install|conda\s+install)\b")
_DESTRUCTIVE = re.compile(
    r"\brm\s+(?:-\w*[rR]\w*|--recursive)|\bgit\s+reset\s+--hard|\bgit\s+clean\s+-\w*f|"
    r"\bgit\s+push\s+.*(?:--force|\s-f\b|--force-with-lease)|\bgit\s+branch\s+-D|"
    r"\bgit\s+checkout\s+--\s+\.|\bgit\s+restore\s+\.|\bgit\s+stash\s+(?:drop|clear)|"
    r"\bdd\s+.*of=/dev/|\bmkfs|\bshred\b|\btruncate\s+-s\s*0|"
    r"\b(?:DROP|TRUNCATE)\s+(?:TABLE|DATABASE)\b|\bDELETE\s+FROM\b", re.I)
_SECURITY_CMD = re.compile(r"\b(?:chmod|chown|sudo|ssh-keygen|gpg|openssl\s+genrsa|"
                           r"setfacl|usermod|passwd)\b")
_READ_TOOLS = {"Read", "NotebookRead", "Grep", "Glob", "WebFetch", "WebSearch",
               "LS", "TodoWrite", "TodoRead", "BashOutput", "TaskOutput"}


def _is_mcp_write(tool):
    return tool.startswith("mcp__") and tool.rsplit("__", 1)[-1] in (
        "write_file", "edit_file", "move_file", "create_directory")


def classify_task(record, standard=STANDARD, trusted_mcp_servers=("filesystem", "fs"),
                  large_change=20):
    """The structural, zero-token classification of one turn.

    Every changed file is sorted into docs / tests / config / data / code
    by its path; commands and tool calls add ``operations`` and the risk
    flags; the primary category is the first of the standard's categories
    the turn touched. Returns a :class:`TaskClassification` whose
    ``evidence`` says what produced each category and flag.
    """
    evidence, touched, flags = {}, set(), set()

    def note(key, why):
        evidence.setdefault(key, [])
        if why not in evidence[key] and len(evidence[key]) < 12:
            evidence[key].append(why)

    files = sorted(set(str(f) for f in record.changed_files))
    for path in files:
        kind = file_kind(path)
        touched.add(kind)
        note(kind, path)
        if _CI.search(path):
            flags.add("ci_change")
            note("ci_change", path)
        if _MANIFESTS.search(path):
            flags.add("dependency_change")
            note("dependency_change", path)
        if _SECRETS.search(path):
            flags.add("security_sensitive")
            note("security_sensitive", path)

    for call in record.calls:
        tool = str(call.get("tool") or "")
        if call.get("failed"):
            flags.add("failed_calls")
            note("failed_calls", tool)
        if call.get("subagent") and tool not in _READ_TOOLS:
            flags.add("subagent_changes")
            note("subagent_changes", tool)
        if tool.startswith("mcp__") and _is_mcp_write(tool):
            server = tool.split("__")[1] if tool.count("__") == 2 else ""
            if server not in trusted_mcp_servers:
                touched.add("operations")
                flags.add("external_effect")
                note("operations", tool)
                note("external_effect", tool)

    for command in record.commands:
        short = command if len(command) <= 160 else command[:157] + "..."
        operation = False
        if _EXTERNAL.search(command) or _HTTP_WRITE.search(command):
            operation = True
            touched.add("operations")
            flags.add("external_effect")
            note("operations", short)
            note("external_effect", short)
        if _DEPENDENCY.search(command):
            operation = True
            touched.add("operations")
            flags.add("dependency_change")
            note("operations", short)
            note("dependency_change", short)
        if _DESTRUCTIVE.search(command):
            flags.add("destructive")
            note("destructive", short)
        if _SECURITY_CMD.search(command):
            flags.add("security_sensitive")
            note("security_sensitive", short)
        if not operation and not is_read_only(command):
            touched.add("execution")            # (dropped if files changed)
            note("execution", short)

    if record.unplaced:
        flags.add("unmodelled_commands")
        for entry in record.unplaced[:12]:
            note("unmodelled_commands", entry)
    if record.verdict == "unverified":
        flags.add("unverified_exit")
        note("unverified_exit", "the gate's escape valve ended the turn")
    if len(files) >= large_change:
        flags.add("large_change")
        note("large_change", "%d files changed" % len(files))

    if not touched:
        touched.add("inquiry")
        note("inquiry", "no file changes and no commands")
    if files:
        touched.discard("execution")
    order = standard.category_ids
    ranked = [c for c in order if c in touched]
    primary, secondary = ranked[0], ranked[1:]
    flag_list = [f for f in standard.flag_ids if f in flags]
    risk = _max_risk(standard.category(primary).risk,
                     *(standard.flag(f).risk for f in flag_list))
    return TaskClassification(
        standard="%s v%s" % (standard.id, standard.version),
        category=primary,
        secondary=secondary,
        flags=flag_list,
        risk=risk,
        checklist=_checklist(standard, primary, flag_list),
        evidence={k: v for k, v in evidence.items()
                  if k in ranked or k in flag_list},
        needs_human_review=risk == "high",
    )


def _checklist(standard, category, flags):
    items = list(standard.category(category).checklist)
    for f in flags:
        items += [i for i in standard.flag(f).checklist if i not in items]
    return items


def apply_audit(classification, audit, standard=STANDARD):
    """Merge an auditor's view into a structural classification.

    The audit can only escalate: its flags are added, the risk becomes the
    higher of the two, and the checklist grows to match. A different
    category, or no audit at all (``None``: the auditor failed or
    declined), marks the task for human review. The structural category is
    kept as the recorded one; the auditor's stays in ``audit``.
    """
    if audit is None:
        classification.audit = {"status": "unavailable"}
        classification.needs_human_review = True
        return classification
    extra = [f for f in audit.flags if f in standard.flag_ids
             and f not in classification.flags]
    flags = [f for f in standard.flag_ids if f in classification.flags or f in extra]
    risk = _max_risk(classification.risk, audit.risk if audit.risk in RISK_LEVELS
                     else "low", *(standard.flag(f).risk for f in flags))
    disputed = audit.category != classification.category or not audit.agrees
    classification.flags = flags
    classification.risk = risk
    classification.checklist = _checklist(standard, classification.category, flags)
    for f in extra:
        classification.evidence.setdefault(f, []).append(
            "added by audit: " + (audit.rationale or audit.auditor or "auditor"))
    classification.needs_human_review = (classification.needs_human_review or disputed
                                         or risk == "high")
    classification.audit = {
        "status": "disputed" if disputed else "agreed",
        "auditor": audit.auditor,
        "category": audit.category,
        "flags": list(audit.flags),
        "risk": audit.risk,
        "rationale": audit.rationale,
    }
    return classification


def audit_classification(record, classification, auditor, standard=STANDARD):
    """Run ``auditor`` and merge its result; an auditor that raises counts
    as unavailable (the task goes to human review), never as agreement."""
    try:
        result = auditor.audit(record, classification, standard)
    except Exception:                                      # noqa: BLE001
        result = None
    if result is not None and result.category not in standard.category_ids:
        result = None
    return apply_audit(classification, result, standard)


class StubAuditor:
    """Deterministic auditor for tests: returns ``result`` (an
    :class:`AuditResult`, or ``None`` for unavailable) or, with ``rule``,
    ``rule(record, classification, standard)``."""

    def __init__(self, result=None, rule=None):
        self._result, self._rule = result, rule

    def audit(self, record, classification, standard):
        if self._rule is not None:
            return self._rule(record, classification, standard)
        return self._result


def render_review(classification, standard=STANDARD):
    """A short Markdown card for a reviewer: category, risk, flags with
    their evidence, the audit outcome, and the checklist."""
    c = classification
    name = standard.category(c.category).name
    lines = ["### %s — %s risk%s" % (name, c.risk,
                                     " — needs human review" if c.needs_human_review
                                     else ""),
             "", "Standard: %s. Category: `%s`" % (c.standard, c.category)
             + (" (also: %s)" % ", ".join("`%s`" % s for s in c.secondary)
                if c.secondary else "") + "."]
    if c.flags:
        lines += ["", "Flags:"]
        for f in c.flags:
            why = "; ".join(c.evidence.get(f, [])[:3])
            lines.append("- `%s`%s" % (f, " — " + why if why else ""))
    if c.audit:
        status = c.audit.get("status")
        if status == "unavailable":
            lines += ["", "Audit: unavailable (the classifier model could not "
                      "review this task)."]
        else:
            if status == "agreed":
                why = ""
            elif c.audit.get("category") != c.category:
                why = " (auditor says `%s`)" % c.audit.get("category")
            else:
                why = " (same category; auditor flags: %s)" % (
                    ", ".join("`%s`" % f for f in c.audit.get("flags") or []) or "none")
            lines += ["", "Audit: %s by %s%s." % (
                status, c.audit.get("auditor") or "auditor", why)]
            if c.audit.get("rationale"):
                lines.append("> " + c.audit["rationale"])
    lines += ["", "Checklist:"] + ["- [ ] " + item for item in c.checklist]
    return "\n".join(lines)


__all__ = [
    "STANDARD", "RISK_LEVELS", "Category", "Flag", "TaskStandard", "TaskRecord",
    "TaskClassification", "AuditResult", "ClassificationAuditor", "StubAuditor",
    "classify_task", "apply_audit", "audit_classification", "render_review",
    "file_kind",
]
