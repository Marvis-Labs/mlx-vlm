from types import SimpleNamespace

import pytest

from ci.model_path_probe import checkpoint, summarize


def test_checkpoint_requires_pinned_repo():
    job = {"hf_checkpoint": {"repo": "org/model", "revision": "abc123"}}
    assert checkpoint(job) == ("org/model", "abc123")


@pytest.mark.parametrize(
    "value",
    [None, {}, {"repo": "org/model"}, {"revision": "abc123"}],
)
def test_checkpoint_rejects_incomplete_metadata(value):
    with pytest.raises(ValueError):
        checkpoint({"hf_checkpoint": value})


def test_summarize_reports_e2e_metrics(monkeypatch):
    monkeypatch.setattr("ci.model_path_probe.time.perf_counter", lambda: 3.0)
    result = SimpleNamespace(
        text="a cat",
        token=42,
        prompt_tokens=12,
        generation_tokens=2,
        prompt_tps=100.0,
        generation_tps=20.0,
        peak_memory=18.5,
        finish_reason="length",
    )
    findings = summarize([result], 1.0, 250.0)
    assert findings["generated_text"] == "a cat"
    assert findings["prefill_tps"] == 100.0
    assert findings["decode_tps"] == 20.0
    assert findings["ttft_ms"] == 250.0
    assert findings["wall_ms"] == 2000.0
    assert findings["peak_memory_gib"] == 18.5
