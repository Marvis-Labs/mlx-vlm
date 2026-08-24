"""Tests for the per-architecture capability report.

Configs come from the local Hugging Face cache and every test skips when the
architecture it needs is absent, so the suite stays offline.
"""

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
    caps = Capabilities(arch="stub", batch=True, kv_quant=True)
    assert caps.applicable() == ("kv_quant", "batch")
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
