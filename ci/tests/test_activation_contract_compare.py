from ci.activation_contract_compare import compare


def probe(verdict, output_hash="same", checks=10, failures=None):
    return {
        "verdict": verdict,
        "output_hash": output_hash,
        "summary": {
            "checks": checks,
            "failures": failures or [],
        },
    }


def test_head_oracle_is_gating_and_base_is_diagnostic():
    result = compare(
        probe("test_failure", "before", failures=["old-bug"]),
        probe("passed", "after"),
    )

    assert result["verdict"] == "passed"
    assert result["correctness"] == {
        "match": True,
        "base_contract": "failed",
        "head_contract": "passed",
        "behavior_changed": True,
    }


def test_head_contract_failure_fails_even_when_base_passed():
    result = compare(
        probe("passed"),
        probe("test_failure", failures=["gate-gradient"]),
    )

    assert result["verdict"] == "test_failure"
    assert result["correctness"]["match"] is False
    assert result["head"]["failures"] == ["gate-gradient"]


def test_equivalent_refactor_does_not_require_behavior_change():
    result = compare(probe("passed"), probe("passed"))

    assert result["verdict"] == "passed"
    assert result["correctness"]["behavior_changed"] is False


def test_unavailable_base_does_not_override_a_valid_head():
    result = compare(
        {"verdict": "unavailable", "error": "symbol absent"},
        probe("passed", "head"),
    )

    assert result["verdict"] == "passed"
    assert result["base"]["error"] == "symbol absent"
