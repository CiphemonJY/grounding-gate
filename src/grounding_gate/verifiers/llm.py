"""Optional LLM reference verifier for the ``verify_with`` tier.

This is the one place in the project that can talk to the Anthropic API, and it
is dependency-isolated exactly like ``adapters/claude_agent_sdk.py``: the module
top imports only stdlib, and the SDK is imported LAZILY inside
``LLMVerifier.__init__`` — and only on the ``client is None`` branch. So
``import grounding_gate`` and ``from grounding_gate.verifiers import ...`` never
pull in ``anthropic``; only constructing an ``LLMVerifier`` without injecting a
client does. Install the SDK via the optional extra::

    pip install grounding-gate[llm]

HONEST NOTE ON THE METHOD (do not overstate this).  The paper this tier echoes
reads a yes/no *probability* straight off the model's output logits. The
Anthropic Messages API exposes NO scoring-token logprobs (confirmed against the
claude-api reference), so we cannot read that probability directly. Instead we
DECOMPOSE the claim into criteria and, for each, do REPEATED EVALUATION: draw
``k`` independent yes/no samples from the API's own inherent sampling
distribution and average them. That Monte-Carlo mean is an *estimator* of the
same expectation the paper reads off the logits — nothing more. The cost is real
(``k`` model calls per criterion, latency ``k`` x one call); the number returned
is a sample mean with sampling error ``~1/sqrt(k)``, not a logit probability.

We deliberately send NO sampling parameters (no ``temperature`` / ``top_p`` /
``top_k``): current models (e.g. ``claude-opus-4-8``) reject those with a 400,
and REPEATED EVALUATION relies on the API's built-in sampling across the ``k``
calls, not on a temperature knob. Re-adding ``temperature`` would break every
real run — the signature omits it on purpose.
"""

DEFAULT_MODEL = "claude-opus-4-8"


def _build_prompt(claim, observations, question):
    """Assemble the single-criterion yes/no prompt.

    Cites the observations the wiring layer retained so the model scores the
    claim against what was actually observed, then asks the ONE criterion
    question and constrains the answer to a bare YES/NO so ``_is_yes`` can parse
    it cheaply within a tiny ``max_tokens`` budget.
    """
    content = str(claim.get("content", "")) if isinstance(claim, dict) else str(claim)
    lines = ["Answer only YES or NO.", ""]
    lines.append("Claim: " + content)
    lines.append("")
    if observations:
        lines.append("Cited observations:")
        for obs in observations:
            if isinstance(obs, dict):
                lines.append(
                    "- tool=%s args=%s result=%s"
                    % (obs.get("tool"), obs.get("args"), obs.get("result")))
            else:
                lines.append("- " + str(obs))
    else:
        lines.append("Cited observations: (none)")
    lines.append("")
    lines.append("Question: " + str(question))
    lines.append("Answer only YES or NO.")
    return "\n".join(lines)


def _is_yes(message):
    """Guarded parse of the first text block -> starts-with-'y' (case-insensitive).

    Tolerant of odd shapes: any missing/blank text block reads as NO rather than
    raising, so a malformed response fails toward NOT-confirming (a downgrade,
    the safe direction) instead of crashing the boundary.
    """
    try:
        for block in message.content:
            text = getattr(block, "text", None)
            if isinstance(text, str) and text.strip():
                return text.lower().lstrip().startswith("y")
    except (AttributeError, TypeError):
        pass
    return False


class LLMVerifier:
    """Reference ``verify_with`` verifier: criteria decomposition + K-sample vote.

    Args:
        model: model id (default ``claude-opus-4-8``).
        k: independent yes/no samples per criterion; ``k >= 1`` or ``ValueError``.
            The per-criterion score is the fraction of the ``k`` samples that
            answered YES (a Monte-Carlo estimate; see the module docstring).
        client: an object exposing ``messages.create(...)``. Injected clients
            (real or fake) skip the SDK import entirely, which is how the tests
            exercise this class with a canned fake and no ``anthropic`` installed.
            When ``None`` (the default) the Anthropic SDK is imported lazily and
            a default client is constructed.
        max_tokens: cap on the yes/no completion; tiny by design.

    Note the signature carries NO ``temperature`` / ``top_p`` / ``top_k`` — see
    the module docstring for why (they 400 on current models, and REPEATED
    EVALUATION does not need them).
    """

    def __init__(self, model=DEFAULT_MODEL, k=5, client=None, max_tokens=8):
        # Validate k FIRST so a bad k raises without needing the SDK — keeps the
        # error hermetic and fast.
        if k < 1:
            raise ValueError("k must be >= 1 (got %r): REPEATED EVALUATION needs "
                             "at least one sample per criterion" % (k,))
        self.model = model
        self.k = k
        self.max_tokens = max_tokens
        if client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise ImportError(
                    "LLMVerifier() needs the Anthropic SDK: "
                    "pip install grounding-gate[llm] (or inject a client=... "
                    "that exposes messages.create(...))") from exc
            client = anthropic.Anthropic()
        self.client = client

    def score(self, claim, observations, criteria):
        """Return ``{criterion_name: monte_carlo_yes_rate}`` over the criteria.

        DECOMPOSITION (one question per criterion) x REPEATED EVALUATION (``k``
        independent samples each). An empty ``criteria`` yields ``{}`` which
        :func:`~grounding_gate.verifiers.aggregate` treats as ABSTAIN, so a claim
        tier with no criteria never forces a downgrade.
        """
        out = {}
        for name, question in criteria:
            yes = 0
            for _ in range(self.k):
                if self._sample_yes(claim, observations, question):
                    yes += 1
            out[name] = yes / self.k
        return out

    def _sample_yes(self, claim, observations, question):
        """One Messages API call -> bool. NO sampling params (see module docs).

        Fails toward NO on ANY API-call error (rate limit, timeout, connection
        drop): the whole tier can only ever DOWNGRADE a structural ACCEPT, so a
        verifier outage must degrade to a NO vote (the strict, safe direction)
        rather than propagate out through ``boundary_check`` and trap the agent.
        This mirrors ``_is_yes``, which already fails a malformed RESPONSE toward
        NO — here the same fail-safe covers the CALL itself. Catching
        ``Exception`` (not ``BaseException``) is deliberate: ``KeyboardInterrupt``
        / ``SystemExit`` still propagate.
        """
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user",
                           "content": _build_prompt(claim, observations, question)}],
            )
        except Exception:
            return False
        return _is_yes(message)
