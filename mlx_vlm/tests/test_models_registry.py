"""Tests for building small models from real configs.

Every test that needs a config resolves it from the local Hugging Face cache and
skips when it is not there, so the suite stays offline. A machine that has
downloaded a checkpoint of a given architecture gets real coverage; a clean
runner gets skips rather than downloads.
"""

from unittest.mock import patch

import pytest

from mlx_vlm.models.cache import make_prompt_cache
from mlx_vlm.tests import models_registry as registry


def _cached(arch):
    """The arch's cached config, or skip."""
    config = registry._cached_configs().get(arch)
    if config is None:
        pytest.skip(f"no {arch} config in the local Hugging Face cache")
    return config


def _any_cached_arch():
    for arch in ("qwen2", "qwen3", "llama", "gemma3"):
        if arch in registry._cached_configs():
            return arch
    pytest.skip("no config for a common architecture in the local cache")


# --- the record ------------------------------------------------------------


def test_an_example_needs_only_a_repository():
    example = registry.ArchExample("mlx-community/Qwen3-0.6B-4bit")
    assert example.default == "mlx-community/Qwen3-0.6B-4bit"
    assert example.extras == {}
    assert example.config_overrides == {}
    assert example.skip is None


def test_registry_holds_only_architectures_needing_something_declared():
    """It is not a table of every supported model, and must not become one."""
    assert len(registry.REGISTRY) < 50
    for arch, example in registry.REGISTRY.items():
        assert example.config_overrides or example.skip or example.extras, (
            f"{arch} declares nothing, so it should resolve from the cache "
            "and not appear here"
        )


# --- resolving a config ----------------------------------------------------


def test_unknown_architecture_says_how_to_fix_it():
    with patch.object(registry, "_cached_configs", return_value={}):
        with pytest.raises(LookupError, match="Add an ArchExample"):
            registry.load_config("not_a_real_architecture")


def test_a_skipped_architecture_reports_its_reason():
    example = registry.ArchExample("x", skip="needs a real checkpoint")
    with patch.dict(registry.REGISTRY, {"stub": example}, clear=False):
        with pytest.raises(LookupError, match="needs a real checkpoint"):
            registry.load_config("stub")


def test_the_config_is_scaled_down():
    arch = _any_cached_arch()
    config = registry.load_config(arch)
    text = config.get("text_config") or config
    assert text["num_hidden_layers"] == registry.SMALL_CONFIG["num_hidden_layers"]
    assert text["hidden_size"] == registry.SMALL_CONFIG["hidden_size"]


def test_overrides_win_over_the_scale_down():
    arch = _any_cached_arch()
    example = registry.ArchExample("x", config_overrides={"num_hidden_layers": 3})
    with patch.dict(registry.REGISTRY, {arch: example}, clear=False):
        config = registry.load_config(arch)
    text = config.get("text_config") or config
    assert text["num_hidden_layers"] == 3


def test_a_cached_config_is_never_downloaded():
    arch = _any_cached_arch()
    with patch.object(registry, "_fetch_config", side_effect=AssertionError("network")):
        registry.load_config(arch)


# --- building --------------------------------------------------------------


def test_a_built_model_makes_a_usable_cache():
    arch = _any_cached_arch()
    model = registry.build_tiny(arch)
    language_model = getattr(model, "language_model", model)
    cache = make_prompt_cache(language_model)
    assert cache, "a built model should produce cache entries"


@pytest.mark.parametrize("arch", sorted(registry.REGISTRY))
def test_declared_architectures_build(arch):
    """Each entry exists because the architecture failed without it."""
    example = registry.REGISTRY[arch]
    if example.skip:
        pytest.skip(example.skip)
    _cached(arch)
    model = registry.build_tiny(arch)
    language_model = getattr(model, "language_model", model)
    assert make_prompt_cache(language_model)
