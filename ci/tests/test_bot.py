import pytest

from ci.bot import BotOutput, BotOutputError


def job(model, mode):
    value = {
        "id": f"model_path:{model}:{mode}",
        "component": "model_path",
        "model": model,
        "mode": mode,
        "changed_paths": [f"mlx_vlm/models/{model}/model.py"],
    }
    if mode == "synthetic":
        value["synthetic"] = {"adapter": model, "profile": "dense_vlm"}
    else:
        value["hf_checkpoint"] = {
            "repo": f"mlx-community/{model}-4bit",
            "revision": "abcdef1234567890",
        }
    return value


def record(*, jobs=None, gates=None, errors=None, results=None, kind="ci_control"):
    return {
        "schema_version": 1,
        "kind": kind,
        "head_sha": "abc123",
        "outcome": "ready",
        "run_url": "https://example.com/run",
        "jobs": jobs or [],
        "gates": gates or [],
        "errors": errors or [],
        "results": results or [],
    }


def test_three_models_render_as_three_model_path_sections():
    jobs = [
        job(model, mode)
        for model in ("qwen2_vl", "gemma3", "pixtral")
        for mode in ("synthetic", "hf_checkpoint")
    ]

    rendered = BotOutput(record(jobs=jobs)).render()

    assert rendered.count("<details open>") == 3
    assert rendered.count("· ModelPath · Awaiting /ci run") == 3
    assert rendered.count("| Synthetic | Planned |") == 3
    assert rendered.count("| HF checkpoint | Planned |") == 3
    assert rendered.index("<strong>gemma3</strong>") < rendered.index(
        "<strong>pixtral</strong>"
    )
    assert rendered.index("<strong>pixtral</strong>") < rendered.index(
        "<strong>qwen2_vl</strong>"
    )


def test_model_result_metrics_stay_inside_its_section():
    jobs = [job("qwen2_vl", mode) for mode in ("synthetic", "hf_checkpoint")]
    results = [
        {
            "component": "model_path",
            "model": "qwen2_vl",
            "mode": "hf_checkpoint",
            "outcome": "passed",
            "metrics": {
                "decode_tps": {
                    "base": 19.8,
                    "head": 20.4,
                    "change_pct": 3.03,
                    "verdict": "improved",
                    "unit": "tok/s",
                },
                "ttft_ms": {
                    "base": 812,
                    "head": 774,
                    "change_pct": -4.68,
                    "verdict": "improved",
                    "unit": "ms",
                },
            },
        }
    ]

    rendered = BotOutput(record(jobs=jobs, results=results)).render()

    assert "<strong>qwen2_vl</strong> · ModelPath · Passed" in rendered
    assert (
        "| HF checkpoint | decode_tps | 19.8 tok/s | 20.4 tok/s | +3.03% | improved |"
        in rendered
    )
    assert (
        "| HF checkpoint | ttft_ms | 812 ms | 774 ms | -4.68% | improved |" in rendered
    )


def test_model_failure_does_not_change_sibling_section_state():
    jobs = [
        job(model, mode)
        for model in ("pixtral", "qwen2_vl")
        for mode in ("synthetic", "hf_checkpoint")
    ]
    errors = [
        {
            "code": "invalid_synthetic_config",
            "component": "model_path",
            "subject": "pixtral",
            "details": {
                "mode": "synthetic",
                "changed_paths": ["mlx_vlm/models/pixtral/model.py"],
            },
        }
    ]

    rendered = BotOutput(record(jobs=jobs, errors=errors)).render()

    assert "<strong>pixtral</strong> · ModelPath · Blocked" in rendered
    assert "<strong>qwen2_vl</strong> · ModelPath · Awaiting /ci run" in rendered
    assert "Status: **Blocked**" in rendered


def test_new_model_uses_model_path_section_and_approval_state():
    pending = [job("new_family", mode) for mode in ("synthetic", "hf_checkpoint")]
    for item in pending:
        item["component"] = "new_model_path"
    gate = {
        "id": "new_model_path:new_family:abc123",
        "component": "new_model_path",
        "model": "new_family",
        "status": "awaiting_maintainer_approval",
        "changed_paths": ["mlx_vlm/models/new_family/model.py"],
        "pending_jobs": pending,
    }

    rendered = BotOutput(record(jobs=[], gates=[gate], kind="ci_control")).render()

    assert (
        "<strong>new_family</strong> · ModelPath · Awaiting maintainer approval"
        in rendered
    )
    assert rendered.count("| Synthetic | Awaiting approval |") == 1
    assert rendered.count("| HF checkpoint | Awaiting approval |") == 1


def test_model_sections_suppress_mentions_and_escape_tables():
    unsafe = job("@reviewer|model", "synthetic")

    rendered = BotOutput(record(jobs=[unsafe])).render()

    assert "@\u200breviewer\\|model" in rendered
    assert "@reviewer|model" not in rendered


def test_unknown_component_requires_its_own_renderer():
    unknown = {
        "id": "component_path:cache",
        "component": "component_path",
        "mode": "default",
    }

    with pytest.raises(BotOutputError, match="component_path"):
        BotOutput(record(jobs=[unknown])).render()
