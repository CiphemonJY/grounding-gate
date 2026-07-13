"""Verifier-tier suite — the optional ``verify_with`` layer.

Fully HERMETIC: the LLM verifier is exercised with an injected fake client, so
no network call is made and ``anthropic`` is never imported. The floor's
zero-LLM guarantee is pinned here too (``test_no_llm_leak_into_core``).

Runs under pytest, or with zero dependencies: ``python tests/test_verifiers.py``.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

try:
    import grounding_gate  # noqa: F401  (installed)
except ImportError:        # zero-install fallback: run from a raw checkout
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grounding_gate.verifiers import (
    GRANULARITY,
    StubVerifier,
    Verifier,
    aggregate,
    criteria_for,
)


# ---------------------------------------------------------------- StubVerifier

def test_stub_fixed_score():
    crit = criteria_for("assertion")
    assert aggregate(StubVerifier(1.0).score({"content": "x"}, [], crit)) >= 0.5
    assert aggregate(StubVerifier(0.0).score({"content": "x"}, [], crit)) < 0.5


def test_stub_rule_receives_claim_obs_criteria():
    captured = {}

    def rule(claim, observations, criteria):
        captured["claim"] = claim
        captured["observations"] = observations
        captured["criteria"] = criteria
        return 1.0

    obs = [{"tool": "read", "args": "file.txt", "result": "d", "tier": "observed"}]
    crit = criteria_for("assertion")
    StubVerifier(rule=rule).score({"claim_type": "assertion", "content": "x"}, obs, crit)
    assert isinstance(captured["claim"], dict) and "claim_type" in captured["claim"]
    assert isinstance(captured["observations"], list)
    assert captured["observations"] == obs
    # criteria are (name, question) pairs
    assert all(isinstance(p, tuple) and len(p) == 2 for p in captured["criteria"])


# ---------------------------------------------------------------- criteria/aggregate

def test_criteria_for_keys_off_tier():
    assert [n for n, q in criteria_for("completion")] == ["effect_shown", "no_overreach"]
    assert [n for n, q in criteria_for("assertion")] == ["claim_supported", "no_overreach"]
    # strict-G holds an assertion to the completion (verified) bar
    assert criteria_for("assertion", strict_g=True) == criteria_for("completion")
    # unknown / non-claim tiers get no criteria
    assert criteria_for("unverified") == ()
    assert criteria_for("none") == ()
    assert criteria_for("banana") == ()


def test_aggregate_min_and_abstain():
    assert aggregate({"a": 0.7, "b": 0.3, "c": 0.9}) == 0.3   # weakest governs
    assert aggregate({}) is None                              # empty -> abstain
    assert aggregate(None) is None                            # none  -> abstain
    assert aggregate(0.7) == 0.7                              # float passthrough
    assert GRANULARITY == 0.5


def test_verifier_protocol_runtime_checkable():
    assert isinstance(StubVerifier(), Verifier)

    class NotAVerifier:
        pass

    assert not isinstance(NotAVerifier(), Verifier)


# ---------------------------------------------------------------- LLMVerifier (fake client)

class _FakeMessage:
    def __init__(self, text):
        self.content = [SimpleNamespace(type="text", text=text)]


class _FakeMessages:
    def __init__(self, responder):
        self._responder = responder
        self.calls = 0

    def create(self, **kwargs):
        # the real API 400s on sampling params — assert we NEVER send them
        assert "temperature" not in kwargs
        assert "top_p" not in kwargs
        assert "top_k" not in kwargs
        assert set(kwargs) == {"model", "max_tokens", "messages"}
        self.calls += 1
        return _FakeMessage(self._responder(self.calls))


class _FakeClient:
    def __init__(self, responder):
        self.messages = _FakeMessages(responder)


def test_llm_verifier_hermetic_fake_client():
    from grounding_gate.verifiers.llm import LLMVerifier

    crit = (("claim_supported", "q?"),)
    claim = {"claim_type": "assertion", "content": "x"}

    all_yes = LLMVerifier(client=_FakeClient(lambda n: "YES"), k=4)
    assert all_yes.score(claim, [], crit) == {"claim_supported": 1.0}

    all_no = LLMVerifier(client=_FakeClient(lambda n: "no"), k=4)
    assert all_no.score(claim, [], crit) == {"claim_supported": 0.0}

    # alternating YES/NO over an even k -> exactly half
    alt = LLMVerifier(client=_FakeClient(lambda n: "YES" if n % 2 else "NO"), k=4)
    assert alt.score(claim, [], crit) == {"claim_supported": 0.5}
    assert alt.client.messages.calls == 4   # one API call per sample

    assert "anthropic" not in sys.modules   # fake client -> no SDK import


class _RaisingMessages:
    def create(self, **kwargs):
        raise RuntimeError("simulated API outage (rate limit / timeout / drop)")


class _RaisingClient:
    def __init__(self):
        self.messages = _RaisingMessages()


def test_llm_verifier_fails_closed_on_api_error():
    # A verifier outage must NEVER propagate through the boundary and trap the
    # agent: a failed sample degrades to a NO vote (0.0), which the wiring then
    # downgrades to the always-legal `unverified` path. Fail-safe toward
    # downgrade, exactly like _is_yes does for a malformed response.
    from grounding_gate.verifiers.llm import LLMVerifier
    crit = (("claim_supported", "q?"),)
    claim = {"claim_type": "assertion", "content": "x"}
    v = LLMVerifier(client=_RaisingClient(), k=4)
    assert v.score(claim, [], crit) == {"claim_supported": 0.0}


def test_llm_verifier_k_must_be_positive():
    from grounding_gate.verifiers.llm import LLMVerifier
    try:
        LLMVerifier(client=_FakeClient(lambda n: "YES"), k=0)
        assert False, "expected ValueError for k=0"
    except ValueError:
        pass


def test_llm_verifier_missing_sdk_raises_pip_hint():
    import importlib.util
    from grounding_gate.verifiers.llm import LLMVerifier
    if importlib.util.find_spec("anthropic") is not None:
        return   # SDK installed: the missing-SDK path isn't exercised in this env
    try:
        LLMVerifier()   # client=None + no SDK -> ImportError with the extra hint
        assert False, "expected ImportError when anthropic is absent"
    except ImportError as exc:
        assert "grounding-gate[llm]" in str(exc)


# ---------------------------------------------------------------- zero-LLM floor

def test_no_llm_leak_into_core():
    # importing the core AND the verifiers base must not pull in the LLM SDK
    import grounding_gate  # noqa: F401
    from grounding_gate.verifiers import StubVerifier as _Stub  # noqa: F401
    assert "anthropic" not in sys.modules


# ------------------------------------------------------- bare-python runner

if __name__ == "__main__":
    import inspect
    failures = []
    cases = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)
             and not inspect.signature(f).parameters]
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
