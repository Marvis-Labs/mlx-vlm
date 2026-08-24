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

import dataclasses
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
    # Its config declares the last twenty layers as sharing KV; scaling the
    # stack down without narrowing that window indexes past the new depth.
    "gemma4": ArchExample(
        default="mlx-community/gemma-4-e2b-it-4bit",
        config_overrides={"num_kv_shared_layers": 0},
    ),
    # A dense checkpoint (`num_experts: 0`) that a shared expert count would
    # otherwise turn into a MoE with no expert dimensions.
    "laguna": ArchExample(
        default="mlx-community/Laguna-S-2.1-oQ6e",
        config_overrides={"num_experts": 0},
    ),
    # Pinned so the LongRoPE factor trimming is measured against a known head
    # dimension rather than whichever Phi checkpoint happens to be cached.
    "phi3": ArchExample(default="mlx-community/Phi-3-mini-4k-instruct-4bit"),
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
    for path in sorted(glob.glob(pattern)):
        try:
            config = json.load(open(path))
        except (OSError, ValueError):
            continue
        model_type = (config.get("model_type") or "").lower()
        found.setdefault(MODEL_REMAPPING.get(model_type, model_type), config)
    return found


def _depth_covering_the_pattern(layer_types, floor: int) -> int:
    """Enough layers for the shortest prefix holding every attention type.

    Attention patterns repeat with a period of their own -- gemma-class configs
    alternate four sliding layers with one full-attention layer, a period of
    five. Scaling below that drops a type entirely, and the model then reports a
    capability its real checkpoint does have. So the depth follows the pattern
    rather than a fixed number.
    """
    if not isinstance(layer_types, list) or not layer_types:
        return floor
    wanted, seen = set(layer_types), set()
    for index, entry in enumerate(layer_types):
        seen.add(entry)
        if seen == wanted:
            return max(floor, index + 1)
    return max(floor, len(layer_types))


def _trim_rope_factors(scaled: dict) -> None:
    """LongRoPE carries one factor per rotary dimension, not per layer."""
    head_dim = scaled.get("head_dim")
    if not isinstance(head_dim, int):
        return
    keep = max(head_dim // 2, 1)

    def trim(mapping):
        for key in ("long_factor", "short_factor"):
            factors = mapping.get(key)
            if isinstance(factors, list) and len(factors) > keep:
                mapping[key] = factors[:keep]

    trim(scaled)
    rope = scaled.get("rope_scaling")
    if isinstance(rope, dict):
        rope = dict(rope)
        trim(rope)
        scaled["rope_scaling"] = rope


def _is_period(name: str, value: Any) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value > 0
        and (name.endswith("_pattern") or name.endswith("_interval"))
    )


def _declared_period(config: dict, module=None) -> int:
    """The longest attention period declared for this architecture.

    Some architectures spell the pattern out as a list; others give only its
    length -- ``sliding_window_pattern``, ``full_attention_interval`` -- and
    derive each layer's type from the index. Several give it only as a dataclass
    default, absent from the checkpoint's config, so the config classes have to
    be read too. Any of them left uncovered loses a layer type when the stack is
    scaled down, and the model then reports a capability its real checkpoint
    does have.
    """
    periods = [value for key, value in config.items() if _is_period(key, value)]
    if module is not None:
        for name in ("ModelConfig", "TextConfig"):
            config_class = getattr(module, name, None)
            if config_class is None or not dataclasses.is_dataclass(config_class):
                continue
            periods += [
                field.default
                for field in dataclasses.fields(config_class)
                if _is_period(field.name, field.default)
            ]
    return max(periods, default=1)


def _scaled(config: dict, overrides: Mapping[str, Any], module=None) -> dict:
    """Scale one config level down, keeping length-derived arrays consistent."""
    scaled = {key: SMALL_CONFIG.get(key, value) for key, value in config.items()}

    floor = max(SMALL_CONFIG["num_hidden_layers"], _declared_period(config, module))
    layer_types = config.get("layer_types")
    if isinstance(layer_types, list) and layer_types:
        depth = _depth_covering_the_pattern(layer_types, floor)
        scaled["num_hidden_layers"] = depth
        scaled["layer_types"] = layer_types[:depth]
    elif floor > SMALL_CONFIG["num_hidden_layers"]:
        scaled["num_hidden_layers"] = floor

    # Any other list the config sizes by layer count has to follow the depth.
    # Only length is used to identify them: guessing at meaning from the values
    # misreads lists that merely happen to hold small integers.
    original_depth = config.get("num_hidden_layers")
    depth = scaled.get("num_hidden_layers")
    if isinstance(original_depth, int) and isinstance(depth, int):
        for key, value in list(scaled.items()):
            if (
                key != "layer_types"
                and isinstance(value, list)
                and len(value) == original_depth
            ):
                scaled[key] = value[:depth]

    _trim_rope_factors(scaled)
    scaled.update(overrides)
    return scaled


def load_config(arch: str) -> dict:
    """Return a scaled-down config for ``arch``.

    Raises:
        LookupError: if no config is cached and none is registered.
    """
    example = REGISTRY.get(arch)
    if example is not None and example.skip:
        raise LookupError(f"{arch} cannot be built small: {example.skip}")

    if example is not None:
        # A declared repository wins over whatever happens to be cached. Many
        # checkpoints share one model type -- twenty-eight of them for a single
        # architecture here -- so picking by directory order would make the
        # answer depend on the machine.
        config = _config_for_repo(example.default) or _fetch_config(example)
    else:
        config = _cached_configs().get(arch)
        if config is None:
            raise LookupError(
                f"No cached config for {arch!r} and none registered. Add an "
                f"ArchExample for it in {__name__}."
            )

    overrides = example.config_overrides if example else {}
    try:
        module = importlib.import_module(f"mlx_vlm.models.{arch}")
    except ImportError:
        module = None
    scaled = _scaled(config, overrides, module)
    for name in ("text_config", "vision_config", "audio_config"):
        scaled.setdefault(name, {})
        if isinstance(scaled[name], dict):
            scaled[name] = _scaled(scaled[name], overrides, module)
    return scaled


def _config_for_repo(repo: str) -> Optional[dict]:
    """A specific repository's cached config, found by name not by model type."""
    directory = "models--" + repo.replace("/", "--")
    pattern = os.path.expanduser(
        f"~/.cache/huggingface/hub/{directory}/snapshots/*/config.json"
    )
    for path in sorted(glob.glob(pattern)):
        try:
            return json.load(open(path))
        except (OSError, ValueError):
            continue
    return None


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
