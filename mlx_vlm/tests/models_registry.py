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
    # Discovered by matching a published config.json whose model_type names
    # the architecture, so the recorded row does not depend on which
    # checkpoints happen to be cached on the machine that regenerated it.
    "afmoe": ArchExample(
        default="optimum-intel-internal-testing/tiny-random-trinity",
        trust_remote_code=True,
    ),
    "apertus": ArchExample(default="swiss-ai/Apertus-8B-Instruct-2509"),
    "aya_vision": ArchExample(default="unsloth/aya-vision-32b"),
    "baichuan_m1": ArchExample(
        default="baichuan-inc/Baichuan-M1-14B-Instruct", trust_remote_code=True
    ),
    "bailing_moe": ArchExample(
        default="inclusionAI/Ling-mini-2.0", trust_remote_code=True
    ),
    "bailing_moe_linear": ArchExample(
        default="inclusionAI/Ring-mini-linear-2.0", trust_remote_code=True
    ),
    "bitnet": ArchExample(
        default="microsoft/bitnet-b1.58-2B-4T", trust_remote_code=True
    ),
    "cohere": ArchExample(default="trl-internal-testing/tiny-CohereForCausalLM"),
    "cohere2": ArchExample(default="trl-internal-testing/tiny-Cohere2ForCausalLM"),
    "cohere2_moe": ArchExample(default="CohereLabs/North-Mini-Code-1.0"),
    "cohere_compass": ArchExample(default="CohereLabs/North-Micro-Vision-Instruct"),
    "colqwen2_5": ArchExample(default="qnguyen3/colqwen2.5-v0.2-mlx"),
    "dbrx": ArchExample(default="trl-internal-testing/tiny-DbrxForCausalLM"),
    "deepseek_v2": ArchExample(
        default="deepseek-ai/DeepSeek-V2-Lite-Chat", trust_remote_code=True
    ),
    "deepseek_v3": ArchExample(
        default="deepseek-ai/DeepSeek-R1", trust_remote_code=True
    ),
    "deepseek_v32": ArchExample(default="deepseek-ai/DeepSeek-V3.2"),
    "deepseek_vl_v2": ArchExample(
        default="deepseek-ai/DeepSeek-OCR", trust_remote_code=True
    ),
    "deepseekocr": ArchExample(
        default="mlx-community/DeepSeek-OCR-8bit", trust_remote_code=True
    ),
    "deepseekocr_2": ArchExample(
        default="mlx-community/DeepSeek-OCR-2-6bit", trust_remote_code=True
    ),
    "diffusion_gemma": ArchExample(default="google/diffusiongemma-26B-A4B-it"),
    "dots1": ArchExample(default="dots-studio/dots.llm1.base"),
    "dots_ocr": ArchExample(default="dots-studio/dots.mocr", trust_remote_code=True),
    "ernie4_5": ArchExample(default="baidu/ERNIE-4.5-0.3B-PT"),
    "ernie4_5_moe": ArchExample(default="baidu/ERNIE-4.5-21B-A3B-PT"),
    "ernie4_5_moe_vl": ArchExample(
        default="baidu/ERNIE-4.5-VL-28B-A3B-PT", trust_remote_code=True
    ),
    "exaone": ArchExample(
        default="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct-AWQ", trust_remote_code=True
    ),
    "exaone4": ArchExample(default="LGAI-EXAONE/EXAONE-4.0-32B"),
    "exaone_moe": ArchExample(default="LGAI-EXAONE/K-EXAONE-236B-A23B"),
    "falcon_ocr": ArchExample(default="tiiuae/Falcon-OCR", trust_remote_code=True),
    "falcon_perception": ArchExample(
        default="tiiuae/Falcon-Perception", trust_remote_code=True
    ),
    "florence2": ArchExample(
        default="microsoft/Florence-2-base", trust_remote_code=True
    ),
    "gemma3n": ArchExample(default="unsloth/gemma-3n-E4B-it"),
    "gemma4_text": ArchExample(default="farbodtavakkoli/OTel-2.0-LLM-31B-IT"),
    "glm4": ArchExample(default="zai-org/GLM-4-9B-0414"),
    "glm4_moe": ArchExample(default="trl-internal-testing/tiny-Glm4MoeForCausalLM"),
    "glm4v": ArchExample(default="zai-org/GLM-4.1V-9B-Thinking"),
    "glm4v_moe": ArchExample(default="zai-org/GLM-4.5V"),
    "glm_moe_dsa": ArchExample(default="zai-org/GLM-5.2"),
    "glm_ocr": ArchExample(default="zai-org/GLM-OCR"),
    "gpt2": ArchExample(default="openai-community/gpt2"),
    "gpt_bigcode": ArchExample(default="bigcode/tiny_starcoder_py"),
    "gpt_neox": ArchExample(default="EleutherAI/pythia-160m"),
    "granite4_vision": ArchExample(
        default="ibm-granite/granite-vision-4.1-4b", trust_remote_code=True
    ),
    "granite_vision": ArchExample(default="mlx-community/granite-vision-3.2-2b-nvfp4"),
    "granitemoe": ArchExample(default="ibm-research/PowerMoE-3b"),
    "granitemoehybrid": ArchExample(default="ibm-granite/granite-4.0-h-tiny"),
    "helium": ArchExample(default="kyutai/helium-1-preview-2b"),
    "hrm_text": ArchExample(default="sapientinc/HRM-Text-1B"),
    "hunyuan_v1_dense": ArchExample(
        default="optimum-intel-internal-testing/tiny-random-hunyuan-v1-dense"
    ),
    "hunyuan_vl": ArchExample(default="tencent/HunyuanOCR"),
    "idefics2": ArchExample(default="HuggingFaceM4/idefics2-8b"),
    "internlm2": ArchExample(
        default="internlm/internlm2-1_8b-reward", trust_remote_code=True
    ),
    "internlm3": ArchExample(
        default="internlm/internlm3-8b-instruct", trust_remote_code=True
    ),
    "internvl_chat": ArchExample(
        default="OpenGVLab/InternVL2-2B", trust_remote_code=True
    ),
    "iquestloopcoder": ArchExample(
        default="IQuestLab/IQuest-Coder-V1-40B-Loop-Instruct", trust_remote_code=True
    ),
    "jamba": ArchExample(default="ai21labs/Jamba-tiny-dev"),
    "kimi_k25": ArchExample(default="moonshotai/Kimi-K2.6", trust_remote_code=True),
    "kimi_k3": ArchExample(default="moonshotai/Kimi-K3", trust_remote_code=True),
    "kimi_linear": ArchExample(
        default="moonshotai/Kimi-Linear-48B-A3B-Instruct", trust_remote_code=True
    ),
    "kimi_vl": ArchExample(
        default="moonshotai/Kimi-VL-A3B-Instruct", trust_remote_code=True
    ),
    "llada2_moe": ArchExample(
        default="inclusionAI/LLaDA2.0-mini", trust_remote_code=True
    ),
    "llama4": ArchExample(default="unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF"),
    "llama4_text": ArchExample(default="trl-internal-testing/tiny-Llama4ForCausalLM"),
    "llama_bidirec": ArchExample(
        default="nvidia/llama-nemotron-rerank-1b-v2", trust_remote_code=True
    ),
    "llava": ArchExample(default="llava-hf/llava-1.5-7b-hf"),
    "llava_next": ArchExample(default="llava-hf/llava-v1.6-mistral-7b-hf"),
    "llmjpvl": ArchExample(
        default="llm-jp/llm-jp-4-vl-9b-beta", trust_remote_code=True
    ),
    "locateanything": ArchExample(
        default="nvidia/LocateAnything-3B", trust_remote_code=True
    ),
    "longcat_flash": ArchExample(
        default="yujiepan/longcat-flash-tiny-random", trust_remote_code=True
    ),
    "longcat_flash_ngram": ArchExample(
        default="Intel/LongCat-Flash-Lite-int4-AutoRound", trust_remote_code=True
    ),
    "mage_flow": ArchExample(default="ajh-code/Mage-Flow-Turbo-XPO3-NVFP4"),
    "mage_vl": ArchExample(default="microsoft/Mage-VL", trust_remote_code=True),
    "mamba": ArchExample(default="state-spaces/mamba-130m-hf"),
    "mellum": ArchExample(default="JetBrains/Mellum2-12B-A2.5B-Base"),
    "mimo": ArchExample(default="XiaomiMiMo/MiMo-7B-RL", trust_remote_code=True),
    "mimo_v2_flash": ArchExample(
        default="XiaomiMiMo/MiMo-V2-Flash", trust_remote_code=True
    ),
    "minicpm": ArchExample(default="openbmb/MiniCPM4.1-8B", trust_remote_code=True),
    "minicpm3": ArchExample(default="openbmb/MiniCPM3-4B", trust_remote_code=True),
    "minicpmo": ArchExample(default="openbmb/MiniCPM-o-4_5", trust_remote_code=True),
    "minicpmv4_6": ArchExample(default="openbmb/MiniCPM-V-4.6"),
    "minimax_h3": ArchExample(default="ddalcu/MiniMax-H3-FL2VA-MLX-Serve-8bit"),
    "minimax_m3": ArchExample(default="pipenetwork/MiniMax-M3-MLX-mixed-3_6bit"),
    "minimax_m3_vl": ArchExample(
        default="MiniMaxAI/MiniMax-M3-MXFP8", trust_remote_code=True
    ),
    "ministral3": ArchExample(default="nvidia/Nemotron-3-Embed-1B-BF16"),
    "mistral4": ArchExample(
        default="onnx-internal-testing/tiny-random-Mistral4ForCausalLM"
    ),
    "mixtral": ArchExample(default="mistralai/Mixtral-8x7B-Instruct-v0.1"),
    "mllama": ArchExample(default="unsloth/Llama-3.2-11B-Vision-Instruct"),
    "molmo2": ArchExample(default="allenai/Molmo2-8B", trust_remote_code=True),
    "molmo_point": ArchExample(default="allenai/MolmoPoint-8B", trust_remote_code=True),
    "moondream3": ArchExample(
        default="moondream/moondream3-preview", trust_remote_code=True
    ),
    "multi_modality": ArchExample(default="deepseek-ai/Janus-Pro-1B"),
    "nanochat": ArchExample(default="nanochat-students/nanochat-d20"),
    "nemotron": ArchExample(default="thhaus/nemotron3-8b"),
    "nemotron_labs_diffusion": ArchExample(
        default="nvidia/Nemotron-Labs-Diffusion-8B", trust_remote_code=True
    ),
    "nemotron_nas": ArchExample(
        default="cyankiwi/Llama-3_3-Nemotron-Super-49B-v1_5-AWQ-4bit",
        trust_remote_code=True,
    ),
    "nemotron_parse": ArchExample(
        default="nvidia/NVIDIA-Nemotron-Parse-v1.1", trust_remote_code=True
    ),
    "nemotron_voicechat": ArchExample(
        default="mlx-community/NemotronLabs-VoiceChat-11B-8bit"
    ),
    "olmo2": ArchExample(default="allenai/OLMo-2-0425-1B"),
    "olmo3": ArchExample(default="allenai/Olmo-3-7B-Instruct"),
    "openelm": ArchExample(
        default="apple/OpenELM-1_1B-Instruct", trust_remote_code=True
    ),
    "paddleocr_vl": ArchExample(
        default="PaddlePaddle/PaddleOCR-VL-1.6", trust_remote_code=True
    ),
    "phi": ArchExample(default="microsoft/phi-2"),
    "phi3_v": ArchExample(
        default="microsoft/Phi-3.5-vision-instruct", trust_remote_code=True
    ),
    "phi3small": ArchExample(
        default="microsoft/Phi-3-small-8k-instruct", trust_remote_code=True
    ),
    "phi4mm": ArchExample(
        default="nvidia/Phi-4-multimodal-instruct-NVFP4", trust_remote_code=True
    ),
    "phimoe": ArchExample(
        default="microsoft/Phi-tiny-MoE-instruct", trust_remote_code=True
    ),
    "phixtral": ArchExample(
        default="mzbac/phi-2-2x3-hf-4bit-mlx", trust_remote_code=True
    ),
    "plamo": ArchExample(default="pfnet/plamo-embedding-1b", trust_remote_code=True),
    "plamo2": ArchExample(default="pfnet/plamo-2-1b", trust_remote_code=True),
    "plamo2vl": ArchExample(default="pfnet/plamo-2.1-8b-vl", trust_remote_code=True),
    "qwen": ArchExample(default="Qwen/Qwen-72B", trust_remote_code=True),
    "qwen2_moe": ArchExample(default="Qwen/Qwen1.5-MoE-A2.7B"),
    "qwen3_moe": ArchExample(default="Qwen/Qwen3-30B-A3B"),
    "qwen3_next": ArchExample(default="Qwen/Qwen3-Coder-Next-FP8"),
    "qwen3_vl_moe": ArchExample(default="QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ"),
    "recurrent_gemma": ArchExample(
        default="RichardErkhov/google_-_recurrentgemma-2b-it-8bits"
    ),
    "rt_detr_v2": ArchExample(default="docling-project/docling-layout-heron"),
    "rwkv7": ArchExample(default="fla-hub/rwkv7-0.4B-g1", trust_remote_code=True),
    "seed_oss": ArchExample(default="ByteDance-Seed/Seed-OSS-36B-Instruct"),
    "siglip": ArchExample(default="google/siglip2-giant-opt-patch16-384"),
    "smolvlm": ArchExample(default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct"),
    "solar_open": ArchExample(default="upstage/Solar-Open-100B"),
    "stablelm": ArchExample(default="stabilityai/stablelm-3b-4e1t"),
    "starcoder2": ArchExample(default="bigcode/starcoder2-3b"),
    "step3p5": ArchExample(default="stepfun-ai/Step-3.5-Flash", trust_remote_code=True),
    "step3p7": ArchExample(default="stepfun-ai/Step-3.7-Flash", trust_remote_code=True),
    "telechat3": ArchExample(
        default="Tele-AI/TeleChat3-36B-Thinking", trust_remote_code=True
    ),
    "youtu_llm": ArchExample(
        default="mlx-community/Youtu-LLM-2B-mlx-4bit", trust_remote_code=True
    ),
    "youtu_vl": ArchExample(
        default="tencent/Youtu-VL-4B-Instruct", trust_remote_code=True
    ),
    "zaya1_vl": ArchExample(default="Zyphra/ZAYA1-VL-8B"),
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
