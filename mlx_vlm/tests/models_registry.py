"""Small, weight-free instances of every supported architecture.

A test that needs a model of a given architecture has two bad options: download
a checkpoint, or hand-write a config. The first is slow and needs the network;
the second is why ``test_models.py`` carries hundreds of copy-pasted configs
that drift from the real thing.

This builds one instead, from a real ``config.json`` scaled down to a few small
layers and initialised randomly. The config comes from the local Hugging Face
cache when a checkpoint of that architecture has been downloaded, and otherwise
from the repository named here -- ``config.json`` alone, a couple of kilobytes,
no weights.

Only architectures that need something declared appear below. Everything else
resolves from the cache, so this is not a table of all supported models and does
not have to be kept complete.

Usage::

    from mlx_vlm.tests.models_registry import build_tiny

    model = build_tiny("qwen3_5")
"""

from __future__ import annotations

import glob
import importlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

__all__ = ["ArchExample", "REGISTRY", "load_config", "build_tiny"]

# Dimensions every architecture is scaled down to. Small enough that random
# initialisation is instant, large enough to keep head/group divisions valid.
SMALL_CONFIG: Mapping[str, Any] = {
    "num_hidden_layers": 4,
    "hidden_size": 64,
    "intermediate_size": 64,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "vocab_size": 128,
    "max_position_embeddings": 256,
    "moe_intermediate_size": 64,
    "shared_expert_intermediate_size": 64,
    "num_experts": 2,
    "num_local_experts": 2,
    "n_routed_experts": 2,
    "num_experts_per_tok": 1,
}

# Sub-configs that `load_model` fills in before building the model config.
MODULE_CONFIGS = ("text", "vision", "perceiver", "projector", "audio")


@dataclass(frozen=True)
class ArchExample:
    """What an architecture needs in order to be built small.

    Args:
        default: repository whose ``config.json`` describes this architecture.
        extras: further repositories, by name, for tests that want a specific
            variant rather than the representative one.
        trust_remote_code: whether loading this architecture's processor or
            config executes code from the repository.
        speculative_model: a drafter paired with ``default``, for tests that
            need speculative decoding.
        config_overrides: applied after the generic scale-down. Some configs
            carry arrays sized by a dimension the scale-down changes, and only
            the architecture knows which.
        skip: why this architecture cannot be built small. Set this instead of
            leaving it silently broken.
    """

    default: str
    extras: Mapping[str, str] = field(default_factory=dict)
    trust_remote_code: bool = False
    speculative_model: Optional[str] = None
    config_overrides: Mapping[str, Any] = field(default_factory=dict)
    skip: Optional[str] = None


REGISTRY: Mapping[str, ArchExample] = {
    # Its config declares 35 layers with the last 20 sharing KV; scaling the
    # stack down without narrowing that window indexes past the new depth.
    "gemma4": ArchExample(
        default="mlx-community/gemma-4-4b-it-4bit",
        config_overrides={"num_kv_shared_layers": 2},
    ),
    # A dense checkpoint (`num_experts: 0`) that the shared expert count would
    # otherwise turn into a MoE with no expert dimensions. Its layer pattern has
    # to keep one full-attention layer, and the arrays beside it match depth.
    "laguna": ArchExample(
        default="mlx-community/Laguna-8B-4bit",
        config_overrides={
            "num_experts": 0,
            "layer_types": [
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
            ],
            "sliding_windows": [512, 512, 512, 512],
            "eagle_aux_hidden_state_layer_ids": [0, 1, 2],
        },
    ),
    # LongRoPE carries one scaling factor per rotary dimension, so the factors
    # have to shrink with the head dimension.
    "phi3": ArchExample(
        default="mlx-community/Phi-3-mini-4k-instruct-4bit",
        config_overrides={
            "rope_scaling": {
                "type": "longrope",
                "long_factor": [1.0] * 6,
                "short_factor": [1.0] * 6,
            }
        },
    ),
    # `no_rope_layers` is indexed by layer, so it has to match the new depth.
    "smollm3": ArchExample(
        default="mlx-community/SmolLM3-3B-4bit",
        config_overrides={"no_rope_layers": [1, 1, 1, 1]},
    ),
}


def _cached_configs() -> Mapping[str, dict]:
    """Every architecture with a `config.json` already in the local HF cache."""
    from mlx_vlm.utils import MODEL_REMAPPING

    found: dict[str, dict] = {}
    pattern = os.path.expanduser(
        "~/.cache/huggingface/hub/models--*/snapshots/*/config.json"
    )
    for path in glob.glob(pattern):
        try:
            config = json.load(open(path))
        except (OSError, ValueError):
            continue
        model_type = (config.get("model_type") or "").lower()
        found.setdefault(MODEL_REMAPPING.get(model_type, model_type), config)
    return found


def _scaled(config: dict, overrides: Mapping[str, Any]) -> dict:
    """Scale one config level down, keeping layer-length arrays consistent."""
    scaled = {key: SMALL_CONFIG.get(key, value) for key, value in config.items()}
    scaled.update(overrides)
    layers = scaled.get("num_hidden_layers")
    layer_types = scaled.get("layer_types")
    if isinstance(layer_types, list) and isinstance(layers, int):
        scaled["layer_types"] = (layer_types * 8)[:layers]
    return scaled


def load_config(arch: str) -> dict:
    """Return a scaled-down config for ``arch``.

    Raises:
        LookupError: if no config is cached and none is registered.
    """
    example = REGISTRY.get(arch)
    if example is not None and example.skip:
        raise LookupError(f"{arch} cannot be built small: {example.skip}")

    config = _cached_configs().get(arch)
    if config is None:
        if example is None:
            raise LookupError(
                f"No cached config for {arch!r} and none registered. Add an "
                f"ArchExample for it in {__name__}."
            )
        config = _fetch_config(example)

    overrides = example.config_overrides if example else {}
    scaled = _scaled(config, overrides)
    for name in ("text_config", "vision_config", "audio_config"):
        scaled.setdefault(name, {})
        if isinstance(scaled[name], dict):
            scaled[name] = _scaled(scaled[name], overrides)
    return scaled


def _fetch_config(example: ArchExample) -> dict:
    """Download only `config.json` for a registered repository."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        example.default,
        "config.json",
        revision="main",
    )
    return json.load(open(path))


def build_tiny(arch: str):
    """Build a randomly initialised, few-layer model of ``arch``.

    Follows the same three-step config construction as ``load_model``: a flat
    ``from_dict`` leaves sub-configs as plain dicts, so the module configs have
    to be filled in before the model is built.
    """
    from mlx_vlm.utils import apply_generation_config_defaults, update_module_configs

    module = importlib.import_module(f"mlx_vlm.models.{arch}")
    config = load_config(arch)

    model_config = module.ModelConfig.from_dict(config)
    model_config = update_module_configs(
        model_config, module, config, list(MODULE_CONFIGS)
    )
    model_config = apply_generation_config_defaults(model_config, config)
    return module.Model(model_config)
