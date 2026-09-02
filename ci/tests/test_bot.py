import pytest

from ci.bot import BotOutput, BotOutputError


def job(model, mode=None):
    value = {
        "id": f"model_path:{model}",
        "work_type": "ModelPath",
        "component": "model_path",
        "model": model,
        "phases": ["synthetic", "hf_checkpoint"],
        "changed_paths": [f"mlx_vlm/models/{model}/model.py"],
        "synthetic": {"adapter": model, "profile": "dense_vlm"},
        "hf_checkpoint": {
            "repo": f"mlx-community/{model}-4bit",
            "revision": "abcdef1234567890",
        },
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


def test_comment_has_no_title_and_identifies_commit():
    rendered = BotOutput(record(jobs=[job("qwen2_vl", "hf_checkpoint")])).render()

    assert not rendered.startswith("## ")
    assert "Commit: `abc123`" in rendered
    assert rendered.startswith("<!-- mlx-vlm:ci:plan -->")


def test_execution_comment_has_attempt_specific_marker():
    value = record(jobs=[job("qwen2_vl", "hf_checkpoint")], kind="ci_execution")
    value["attempt_id"] = "123456"

    rendered = BotOutput(value).render()

    assert rendered.startswith("<!-- mlx-vlm:ci:attempt:123456 -->")
    assert "Attempt: `123456`" in rendered


def test_correctness_failure_makes_performance_advisory_and_lists_runner_cache():
    result = {
        "component": "model_path",
        "model": "qwen2_vl",
        "outcome": "test_failure",
        "device": "mini-1",
        "cache": {"before": "complete", "after": "complete", "reused": True},
        "phases": {
            "synthetic": {
                "outcome": "passed",
                "findings": {"correctness": {"match": True}},
            },
            "hf_checkpoint": {
                "outcome": "test_failure",
                "findings": {
                    "correctness": {
                        "match": False,
                        "base_output_hash": "base",
                        "head_output_hash": "head",
                    },
                    "metrics": {
                        "decode_tps": {
                            "base": 10,
                            "head": 12,
                            "change_pct": 20,
                            "verdict": "improved",
                            "unit": "tok/s",
                        }
                    },
                },
            },
        },
    }
    value = record(jobs=[job("qwen2_vl")], results=[result], kind="ci_execution")
    value["attempt_id"] = "42"

    rendered = BotOutput(value).render()

    assert "· ModelPath · Test failed" in rendered
    assert (
        "| HF checkpoint | decode_tps | 10 tok/s | 12 tok/s | +20.00% | advisory |"
        in rendered
    )
    assert (
        "Performance measurements are advisory because correctness failed" in rendered
    )
    assert "Runner: mini-1" in rendered
    assert "cache reused" in rendered


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


def test_blocked_attempt_marks_unrelated_model_work_not_run():
    value = record(
        jobs=[job("qwen2_vl")],
        errors=[
            {
                "code": "security_policy_violation",
                "component": "security_change",
                "subject": "pull_request",
            }
        ],
    )
    value.update({"kind": "ci_execution", "outcome": "blocked"})

    rendered = BotOutput(value).render()

    assert "<strong>qwen2_vl</strong> · ModelPath · Not run" in rendered
    assert "| Synthetic | Not run |" in rendered
    assert "Not run because another CI check blocked this attempt." in rendered


def test_new_model_uses_model_path_section_and_approval_state():
    pending = job("new_family")
    gate = {
        "id": "new_model_path:new_family:abc123",
        "component": "new_model_path",
        "model": "new_family",
        "status": "awaiting_maintainer_approval",
        "changed_paths": ["mlx_vlm/models/new_family/model.py"],
        "pending_work": pending,
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


def test_mlp_change_renders_symbol_summary_and_model_path_provenance():
    work = job("qwen2_vl")
    work["phases"] = ["mlp_contract", "hf_checkpoint"]
    work["mlp_contract"] = {
        "symbols": ["SwiGLUMLP"],
        "consumer": "qwen2_vl",
    }
    work["origins"] = [{"change_type": "MLPChange", "symbol": "SwiGLUMLP"}]
    value = record(jobs=[work])
    value["components"] = ["mlp_change"]

    rendered = BotOutput(value).render()

    assert "<strong>SwiGLUMLP</strong> · MLPChange · Awaiting /ci run" in rendered
    assert "1 ModelPath jobs; 1 with HF checkpoints" in rendered
    assert "<strong>qwen2_vl</strong> · ModelPath · Awaiting /ci run" in rendered
    assert "| MLP contract | Planned | SwiGLUMLP |" in rendered


def cache_job():
    return {
        "id": "kv_cache_change:dense",
        "work_type": "KVCacheChange",
        "component": "kv_cache_change",
        "profile": "dense",
        "changed_paths": ["mlx_vlm/models/cache.py"],
        "phases": ["kv_cache_contract"],
        "head_sha": "head-sha",
        "contract_sha": "contract-sha",
        "kv_cache_contract": {
            "profile": "dense",
            "implementations": ["KVCache", "SimpleKVCache", "BatchKVCache"],
        },
    }


def activation_job(profile="swiglu"):
    symbols = ["swiglu"] if profile == "swiglu" else ["xielu", "XieLU"]
    downstream = ["SwiGLUMLP", "SwitchGLU"] if profile == "swiglu" else ["apertus"]
    return {
        "id": f"activation_change:{profile}",
        "work_type": "ActivationChange",
        "component": "activation_change",
        "profile": profile,
        "changed_paths": ["mlx_vlm/models/activations.py"],
        "phases": ["activation_contract"],
        "base_sha": "base-sha",
        "head_sha": "head-sha",
        "contract_sha": "contract-sha",
        "activation_contract": {
            "profile": profile,
            "symbols": symbols,
            "downstream": downstream,
        },
    }


def test_activation_change_renders_one_section_per_profile():
    value = record(jobs=[activation_job("swiglu"), activation_job("xielu")])
    value["components"] = ["activation_change"]

    rendered = BotOutput(value).render()

    assert rendered.count("· ActivationChange · Awaiting /ci run") == 2
    assert "<strong>SwiGLU</strong>" in rendered
    assert "<strong>XieLU</strong>" in rendered
    assert "swiglu; downstream: SwiGLUMLP, SwitchGLU" in rendered


def test_activation_execution_reports_base_as_diagnostic_and_head_as_gating():
    result = {
        "component": "activation_change",
        "job_id": "activation_change:swiglu",
        "outcome": "passed",
        "device": "mini-1",
        "phases": {
            "activation_contract": {
                "outcome": "passed",
                "findings": {
                    "verdict": "passed",
                    "correctness": {
                        "match": True,
                        "base_contract": "failed",
                        "head_contract": "passed",
                        "behavior_changed": True,
                    },
                    "head": {"checks": 11, "failures": []},
                },
            }
        },
    }
    value = record(jobs=[activation_job()], results=[result], kind="ci_execution")
    value["components"] = ["activation_change"]

    rendered = BotOutput(value).render()

    assert "<strong>SwiGLU</strong> · ActivationChange · Passed" in rendered
    assert (
        "Base contract: failed; head contract: passed; behavior changed: yes"
        in rendered
    )
    assert "Head checks: 11; failures: none" in rendered
    assert "Runner: mini-1" in rendered
    assert "Cache: not applicable" in rendered
    assert "Correctness: passed" in rendered
    assert "Performance: not run" in rendered
    assert "Terminal state: Passed" in rendered


def test_activation_execution_explains_probe_failure():
    result = {
        "component": "activation_change",
        "job_id": "activation_change:xielu",
        "outcome": "test_failure",
        "phases": {
            "activation_contract": {
                "outcome": "test_failure",
                "findings": {
                    "verdict": "test_failure",
                    "correctness": {
                        "match": False,
                        "base_contract": "passed",
                        "head_contract": "failed",
                        "behavior_changed": True,
                    },
                    "base": {"checks": 11, "failures": []},
                    "head": {
                        "checks": 0,
                        "failures": [],
                        "error": "ImportError: XieLU missing",
                    },
                },
            }
        },
    }
    value = record(
        jobs=[activation_job("xielu")], results=[result], kind="ci_execution"
    )
    value["components"] = ["activation_change"]

    rendered = BotOutput(value).render()

    assert "<strong>XieLU</strong> · ActivationChange · Test failed" in rendered
    assert "Head probe error: ImportError: XieLU missing" in rendered
    assert "Correctness: failed" in rendered
    assert "Terminal state: Test failed" in rendered


def test_kv_cache_change_renders_a_profile_section_before_execution():
    value = record(jobs=[cache_job()])
    value["components"] = ["kv_cache_change"]

    rendered = BotOutput(value).render()

    assert "<strong>dense</strong> · KVCacheChange · Awaiting /ci run" in rendered
    assert (
        "| KV cache contract | Planned | KVCache, SimpleKVCache, BatchKVCache |"
        in rendered
    )
    assert "Head: head-sha; trusted contract: contract-sha" in rendered


def test_kv_cache_execution_lists_every_contract_run():
    result = {
        "component": "kv_cache_change",
        "job_id": "kv_cache_change:dense",
        "outcome": "passed",
        "device": "mini-1",
        "phases": {
            "kv_cache_contract": {
                "outcome": "passed",
                "findings": {
                    "verdict": "passed",
                    "checks": 168,
                    "cases": [
                        {
                            "case": "KVCache",
                            "checks": 102,
                            "runs": [
                                {"sequence": "append-trim-resume"},
                                {"sequence": "snapshot-restore-resume"},
                            ],
                            "failures": [],
                        },
                        {
                            "case": "SimpleKVCache",
                            "checks": 66,
                            "runs": [{"sequence": "append"}],
                            "failures": [],
                        },
                        {
                            "case": "BatchKVCache",
                            "checks": 81,
                            "runs": [{"sequence": "padding-finalize-filter-extract"}],
                            "failures": [],
                        },
                    ],
                },
            }
        },
    }
    value = record(jobs=[cache_job()], results=[result], kind="ci_execution")
    value["components"] = ["kv_cache_change"]

    rendered = BotOutput(value).render()

    assert "<strong>dense</strong> · KVCacheChange · Passed" in rendered
    assert (
        "KVCache: 102 checks; append-trim-resume, snapshot-restore-resume" in rendered
    )
    assert "SimpleKVCache: 66 checks; append" in rendered
    assert "BatchKVCache: 81 checks; padding-finalize-filter-extract" in rendered
    assert "Runner: mini-1" in rendered
    assert "Cache: not applicable" in rendered
    assert "Correctness: passed" in rendered
    assert "Performance: not run" in rendered
    assert "Terminal state: Passed" in rendered


def test_kv_cache_execution_compacts_seeded_state_machine_runs():
    runs = [
        {"sequence": "append-trim-resume"},
        *({"sequence": f"kv-state-machine-{index}"} for index in range(5)),
    ]
    result = {
        "component": "kv_cache_change",
        "job_id": "kv_cache_change:dense",
        "outcome": "passed",
        "phases": {
            "kv_cache_contract": {
                "outcome": "passed",
                "findings": {
                    "cases": [
                        {
                            "case": "KVCache",
                            "checks": 774,
                            "runs": runs,
                            "failures": [],
                        }
                    ]
                },
            }
        },
    }
    value = record(jobs=[cache_job()], results=[result], kind="ci_execution")
    value["components"] = ["kv_cache_change"]

    rendered = BotOutput(value).render()

    assert "append-trim-resume, 5 seeded state-machine runs" in rendered
    assert "kv-state-machine-0" not in rendered


def test_kv_cache_execution_compacts_long_deterministic_run_lists():
    runs = [
        {"sequence": name}
        for name in (
            "left-padding-prefill-decode",
            "right-padding-finalize-decode",
            "block-window-crossing",
            "filter-and-extract",
            "merge-and-extend",
            "snapshot-restore-resume",
            "trim-before-saturation",
            "reset-and-reuse",
        )
    ]
    result = {
        "component": "kv_cache_change",
        "job_id": "kv_cache_change:dense",
        "outcome": "passed",
        "phases": {
            "kv_cache_contract": {
                "outcome": "passed",
                "findings": {
                    "cases": [
                        {
                            "case": "BatchRotatingKVCache",
                            "checks": 1085,
                            "runs": runs,
                            "failures": [],
                        }
                    ]
                },
            }
        },
    }
    value = record(jobs=[cache_job()], results=[result], kind="ci_execution")
    value["components"] = ["kv_cache_change"]

    rendered = BotOutput(value).render()

    assert "filter-and-extract, and 4 more deterministic runs." in rendered
    assert "trim-before-saturat\n" not in rendered


def test_kv_cache_runner_crash_is_terminal_not_planned():
    result = {
        "component": "kv_cache_change",
        "job_id": "kv_cache_change:dense",
        "profile": "dense",
        "outcome": "infrastructure_failure",
        "findings": {"error": "runner produced no result"},
    }
    value = record(jobs=[cache_job()], results=[result], kind="ci_execution")
    value["components"] = ["kv_cache_change"]

    rendered = BotOutput(value).render()

    assert "<strong>dense</strong> · KVCacheChange · Infrastructure failed" in rendered
    assert "| KV cache contract | Infrastructure failed |" in rendered
    assert "Contract execution failed: runner produced no result." in rendered
    assert "Correctness: not reported" in rendered
    assert "Terminal state: Infrastructure failed" in rendered


def test_no_eligible_runner_is_reported_inside_affected_model_section():
    jobs = [job(model, "hf_checkpoint") for model in ("pixtral", "qwen2_vl")]
    results = [
        {
            "component": "model_path",
            "model": "qwen2_vl",
            "mode": "hf_checkpoint",
            "outcome": "no_eligible_runner",
            "required_memory_gib": 64,
            "required_disk_gib": 24,
            "attempts": [
                {
                    "device": "mini",
                    "memory_gib": 16,
                    "reason": "declined_memory",
                }
            ],
            "unavailable": [
                {
                    "device": "studio",
                    "memory_gib": 32,
                    "reason": "declined_busy",
                },
                {
                    "device": "m5",
                    "memory_gib": 128,
                    "reason": "leased",
                    "attempt_id": "12345",
                    "expires_at": "2026-08-30T00:05:00Z",
                },
            ],
        }
    ]

    rendered = BotOutput(record(jobs=jobs, results=results)).render()

    assert "<strong>qwen2_vl</strong> · ModelPath · No eligible runner" in rendered
    assert "<strong>pixtral</strong> · ModelPath · Awaiting /ci run" in rendered
    assert "Required: 64 GiB memory and 24 GiB disk" in rendered
    assert "mini (16 GiB): declined_memory" in rendered
    assert "studio (32 GiB): declined_busy" in rendered
    assert (
        "m5 (128 GiB): leased by attempt 12345 until 2026-08-30T00:05:00Z" in rendered
    )
    assert "Retry with /ci run" in rendered


def test_model_path_findings_and_cache_are_reported_in_its_section():
    jobs = [job("qwen3_vl_moe", "hf_checkpoint")]
    results = [
        {
            "component": "model_path",
            "model": "qwen3_vl_moe",
            "mode": "hf_checkpoint",
            "outcome": "passed",
            "cache": {"before": "complete", "after": "complete", "reused": True},
            "findings": {
                "prefill_tps": 1745.189,
                "decode_tps": 122.94,
                "ttft_ms": 459.447,
                "peak_memory_gib": 19.3518,
                "output_hash": "a76d435439ce33d5",
            },
        }
    ]

    rendered = BotOutput(record(jobs=jobs, results=results)).render()

    assert "HF checkpoint findings (cache reused)" in rendered
    assert "prefill 1745.189 tok/s" in rendered
    assert "decode 122.94 tok/s" in rendered
    assert "TTFT 459.447 ms" in rendered
    assert "peak memory 19.3518 GiB" in rendered
    assert "output a76d435439ce33d5" in rendered


def test_security_profile_renders_as_an_independent_static_section():
    value = {
        "kind": "ci_control",
        "head_sha": "a" * 40,
        "outcome": "blocked",
        "components": ["security_change"],
        "jobs": [],
        "gates": [],
        "checks": [
            {
                "component": "security_change",
                "profile": "unsafe_deserialization",
                "status": "blocked",
                "changed_paths": ["mlx_vlm/models/example/convert.py"],
                "findings": [
                    {
                        "category": "unsafe_deserialization",
                        "rule": "torch_load_without_weights_only",
                    }
                ],
            }
        ],
        "errors": [
            {
                "component": "security_change",
                "subject": "pull_request",
                "code": "security_policy_violation",
                "details": {"profile": "unsafe_deserialization"},
            }
        ],
    }

    rendered = BotOutput(value).render()

    assert "unsafe_deserialization" in rendered
    assert "SecurityChange" in rendered
    assert "torch_load_without_weights_only" in rendered
    assert "Blocked" in rendered


def test_security_scanner_error_cannot_render_as_passed():
    value = {
        "kind": "ci_control",
        "head_sha": "a" * 40,
        "outcome": "blocked",
        "components": ["security_change"],
        "jobs": [],
        "gates": [],
        "checks": [],
        "errors": [
            {
                "component": "security_change",
                "subject": "pull_request",
                "code": "security_scan_failed",
                "details": {"changed_paths": ["mlx_vlm/server/api.py"]},
            }
        ],
    }

    rendered = BotOutput(value).render()

    assert "pull_request" in rendered
    assert "SecurityChange" in rendered
    assert "Blocked" in rendered
    assert "mlx_vlm/server/api.py" in rendered
