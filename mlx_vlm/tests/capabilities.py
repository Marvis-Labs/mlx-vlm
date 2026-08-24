"""What each architecture can be run with.

Not every model supports every way of running one. A text-only model has no
image path, a recurrent cache cannot be quantized, and a model whose
``make_cache`` takes no arguments cannot honour a KV bound. Exercising a feature
against an architecture that does not have it either raises or, worse, silently
does nothing and looks like a pass.

Some of this is already declared, in three different places -- a drafter table,
per-model ``chunked_prefill_policy`` hooks, ``make_cache`` signatures -- and the
rest is structural, decided by the caches a model actually builds. This reads
whichever is authoritative for each feature and answers in one place.

Structural facts are read from a built model rather than declared, because the
cache a model constructs is the ground truth and a declaration beside it could
only drift. Declared facts are read from the declaration, because nothing about
a model's shape reveals which drafter pairs with it.

Usage::

    from mlx_vlm.tests.capabilities import capabilities

    caps = capabilities("qwen3_5")
    if caps.bounded_kv:
        ...
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Optional, Tuple

__all__ = ["Capabilities", "capabilities", "representatives"]


@dataclass(frozen=True)
class Capabilities:
    """The ways one architecture can be run.

    ``speculative`` carries the drafter kind rather than a flag, since knowing
    a model supports speculation is useless without knowing which drafter to
    pair with it.

    Batch conversion is deliberately absent: every cache class in the tree has a
    batched counterpart, so the answer was always yes and the column carried no
    information.
    """

    arch: str
    cache_kinds: Tuple[str, ...] = ()
    image_in: bool = False
    audio_in: bool = False
    hybrid_cache: bool = False
    kv_quant: bool = False
    bounded_kv: bool = False
    trimmable: bool = False
    apc_exact: bool = False
    apc_block: bool = False
    chunked_prefill: bool = False
    speculative: Optional[str] = None

    def applicable(self) -> Tuple[str, ...]:
        """The feature names that apply, for building a sweep or a test plan."""
        flags = (
            "image_in",
            "audio_in",
            "hybrid_cache",
            "kv_quant",
            "bounded_kv",
            "trimmable",
            "apc_exact",
            "apc_block",
            "chunked_prefill",
        )
        names = [name for name in flags if getattr(self, name)]
        if self.speculative:
            names.append("speculative")
        return tuple(names)

    def signature(self) -> str:
        """Architectures sharing a signature can share one representative."""
        return ",".join(self.applicable()) or "text-only"


def _honours_a_kv_bound(language_model) -> bool:
    """Whether a KV bound reaches the cache this model builds.

    A model with no ``make_cache`` gets the bounded default, so it qualifies. A
    model that builds its own only qualifies if it takes the bound.
    """
    make_cache = getattr(language_model, "make_cache", None)
    if make_cache is None:
        return True
    try:
        parameters = inspect.signature(make_cache).parameters
    except (TypeError, ValueError):
        return False
    return any(name in parameters for name in ("max_size", "max_kv_size"))


def _drafter_kind(arch: str) -> Optional[str]:
    """The drafter declared for this architecture, if any."""
    from mlx_vlm.speculative.drafters import DRAFTER_KIND_BY_MODEL_TYPE

    if arch in DRAFTER_KIND_BY_MODEL_TYPE:
        return DRAFTER_KIND_BY_MODEL_TYPE[arch]
    for model_type, kind in DRAFTER_KIND_BY_MODEL_TYPE.items():
        if model_type.startswith(arch):
            return kind
    return None


def capabilities(arch: str, model=None) -> Capabilities:
    """Report what ``arch`` can be run with.

    Args:
        arch: model directory name, e.g. ``"qwen3_5"``.
        model: an already-built model, to avoid building a second one.
    """
    from mlx_vlm import apc_adapters as apc
    from mlx_vlm.models.cache import make_prompt_cache
    from mlx_vlm.tests.models_registry import build_tiny

    if model is None:
        model = build_tiny(arch)
    language_model = getattr(model, "language_model", model)
    cache = make_prompt_cache(language_model)
    kinds = tuple(type(entry).__name__ for entry in cache)

    def every(predicate) -> bool:
        return bool(cache) and all(predicate(entry) for entry in cache)

    return Capabilities(
        arch=arch,
        cache_kinds=kinds,
        image_in=any(
            hasattr(model, name) for name in ("vision_tower", "vision_model", "visual")
        ),
        audio_in=any(
            hasattr(model, name)
            for name in ("audio_tower", "audio_model", "audio_encoder")
        ),
        hybrid_cache=len(set(kinds)) > 1,
        kv_quant=any(hasattr(entry, "to_quantized") for entry in cache),
        bounded_kv=_honours_a_kv_bound(language_model),
        trimmable=every(lambda c: getattr(c, "is_trimmable", lambda: False)()),
        apc_exact=every(apc.apc_exact_eligible),
        apc_block=every(apc.apc_block_eligible),
        chunked_prefill=any(
            hasattr(obj, "chunked_prefill_policy") for obj in (language_model, model)
        ),
        speculative=_drafter_kind(arch),
    )


def representatives(all_caps):
    """One architecture per distinct signature.

    Architectures sharing a signature exercise the same set of code paths, so a
    sweep that runs every one of them mostly repeats itself. Forty architectures
    reduce to twenty signatures, and the reduction grows as more are added.

    Args:
        all_caps: the capabilities of every architecture under consideration.

    Returns:
        One representative per signature, in a stable order.
    """
    chosen: dict = {}
    for caps in sorted(all_caps, key=lambda c: c.arch):
        chosen.setdefault(caps.signature(), caps)
    return tuple(chosen[key] for key in sorted(chosen))
