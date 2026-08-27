"""Tests for the per-architecture capability report.

Configs come from the local Hugging Face cache and every test skips when the
architecture it needs is absent, so the suite stays offline.
"""

from unittest.mock import patch

import pytest

from mlx_vlm.tests import models_registry as registry
from mlx_vlm.tests.capabilities import Capabilities, capabilities


def _require(arch):
    if arch not in registry._cached_configs():
        pytest.skip(f"no {arch} config in the local Hugging Face cache")
    return arch


def test_nothing_applies_by_default():
    empty = Capabilities(arch="stub")
    assert empty.applicable() == ()
    assert empty.signature() == "text-only"


def test_a_signature_lists_only_what_applies():
    caps = Capabilities(arch="stub", kv_quant=True)
    assert "image_in" not in caps.signature()


def test_speculative_reports_the_drafter_kind_not_a_flag():
    caps = Capabilities(arch="stub", speculative="mtp")
    assert caps.speculative == "mtp"
    assert "speculative" in caps.applicable()


def test_a_recurrent_model_cannot_quantize_or_bound_its_cache():
    """`ArraysCache` holds recurrent state: no `to_quantized`, no trim."""
    caps = capabilities(_require("mamba2"))
    assert not caps.kv_quant
    assert not caps.trimmable
    assert caps.apc_exact, "whole-prefix snapshots still work"


def test_a_hybrid_reports_more_than_one_cache_class():
    caps = capabilities(_require("qwen3_5"))
    assert caps.hybrid_cache
    assert len(set(caps.cache_kinds)) > 1


def test_a_model_that_builds_its_own_cache_cannot_take_a_bound():
    """qwen3_5 defines a zero-argument `make_cache`, so the bound cannot reach it."""
    assert not capabilities(_require("qwen3_5")).bounded_kv


def test_a_model_without_make_cache_gets_the_bounded_default():
    assert capabilities(_require("qwen2")).bounded_kv


def test_a_vision_model_reports_an_image_path():
    assert capabilities(_require("gemma3")).image_in


def test_a_text_model_reports_no_image_path():
    assert not capabilities(_require("qwen2")).image_in


def test_an_existing_model_can_be_reused_instead_of_rebuilt():
    arch = _require("qwen2")
    model = registry.build_tiny(arch)
    assert capabilities(arch, model=model) == capabilities(arch, model=model)


# --- the recorded matrix ---------------------------------------------------


def _recorded():
    import json
    import os

    from mlx_vlm.tests import generate_capabilities

    if not os.path.exists(generate_capabilities.RECORD):
        pytest.skip("no recorded capability matrix")
    return json.load(open(generate_capabilities.RECORD))


def test_the_record_is_not_empty():
    assert _recorded(), "capabilities.json should describe at least one architecture"


def test_recorded_capabilities_still_hold():
    """A change to what an architecture can do has to be recorded deliberately.

    Only architectures resolvable on this machine are compared, so a partial
    Hugging Face cache narrows the check rather than failing it.
    """
    from dataclasses import asdict

    from mlx_vlm.tests.generate_capabilities import architectures

    recorded = _recorded()
    checked, drifted = 0, {}
    for arch in architectures():
        if arch not in recorded:
            continue
        try:
            caps = capabilities(arch)
        except Exception:
            continue
        current = asdict(caps)
        current.pop("arch")
        current["cache_kinds"] = sorted(set(current["cache_kinds"]))
        checked += 1
        for feature, was in recorded[arch].items():
            now = current.get(feature)
            if now != was:
                drifted[f"{arch}.{feature}"] = {"recorded": was, "now": now}

    if not checked:
        pytest.skip("no recorded architecture is resolvable here")
    assert not drifted, (
        f"capability drift in {len(drifted)} field(s): {drifted}. If intended, "
        "rerun python -m mlx_vlm.tests.generate_capabilities and commit the result."
    )


# --- representatives -------------------------------------------------------


def test_representatives_keep_one_architecture_per_signature():
    from mlx_vlm.tests.capabilities import representatives

    a = Capabilities(arch="a", kv_quant=True)
    b = Capabilities(arch="b", kv_quant=True)
    c = Capabilities(arch="c", image_in=True)
    chosen = representatives([a, b, c])
    assert len(chosen) == 2
    assert {r.signature() for r in chosen} == {a.signature(), c.signature()}


def test_representatives_are_chosen_deterministically():
    from mlx_vlm.tests.capabilities import representatives

    rows = [Capabilities(arch=name, kv_quant=True) for name in ("z", "a", "m")]
    assert representatives(rows)[0].arch == "a"
    assert representatives(reversed(rows))[0].arch == "a"


def test_representatives_cover_every_signature_in_the_record():
    from mlx_vlm.tests.capabilities import representatives
    from mlx_vlm.tests.generate_capabilities import architectures

    rows = []
    for arch in architectures():
        try:
            rows.append(capabilities(arch))
        except Exception:
            continue
    if not rows:
        pytest.skip("no architecture resolvable here")
    chosen = representatives(rows)
    assert {r.signature() for r in chosen} == {r.signature() for r in rows}
    assert len(chosen) <= len(rows)


def test_capability_does_not_depend_on_how_far_the_model_was_scaled_down():
    """A capability that changes with the scale-down is an artefact, not a fact.

    Attention patterns have a period of their own, and scaling below it drops a
    layer type: the gemma family reported no hybrid cache at four layers and a
    hybrid one at twelve.
    """
    import mlx.core as mx

    from mlx_vlm.tests.generate_capabilities import architectures

    original = dict(registry.SMALL_CONFIG)
    archs = architectures()[:12]
    if not archs:
        pytest.skip("no architecture resolvable here")
    try:
        runs = {}
        for depth in (4, 9):
            registry.SMALL_CONFIG = {**original, "num_hidden_layers": depth}
            runs[depth] = {}
            for arch in archs:
                try:
                    runs[depth][arch] = capabilities(arch).signature()
                except Exception as error:
                    runs[depth][arch] = f"failed: {type(error).__name__}"
                mx.clear_cache()
    finally:
        registry.SMALL_CONFIG = original

    differing = {
        arch: (runs[4][arch], runs[9][arch])
        for arch in archs
        if runs[4][arch] != runs[9][arch]
    }
    assert not differing, f"capability varies with scale-down depth: {differing}"


def test_a_declared_repository_beats_whatever_is_cached():
    """Twenty model types have several cached checkpoints; the answer must not
    depend on which one the filesystem lists first."""
    arch = next(iter(registry.REGISTRY), None)
    if arch is None:
        pytest.skip("registry is empty")
    example = registry.REGISTRY[arch]
    with (
        patch.object(
            registry, "_cached_configs", return_value={arch: {"model_type": "decoy"}}
        ),
        patch.object(registry, "_config_for_repo", return_value=None) as by_repo,
        patch.object(registry, "_fetch_config", return_value={"model_type": arch}),
    ):
        registry.load_config(arch)
    by_repo.assert_called_once_with(example.default)
