# How the CI tests each component

One row per component: what code it guards, the workload that exercises it, the
configs (each hits a distinct branch), the metrics that catch a regression, which
architectures it reaches, and the gap worth closing. Workload constants live in
`ci/probe.py`: prompt 512 tokens, long prompt 4096, generation 128.

## apc — automatic prefix caching
- **Guards:** `apc.py`, `apc_adapters.py`, `apc_storage.py`.
- **Workload:** `shared_prefix_pair` — prime with a 512-token prefix, then measure a
  second request that reuses it. (A single request would report zero change.)
- **Configs:** off · exact (min 16) · block-16 · block-256 · guarded. Exact vs block
  modes, block-size scaling, the exact-prefix guard. All bite (512-token prefix reused).
- **Metrics:** token_hit_rate, matched_tokens, rejects_by_reason (functional, the
  primary signal) + prefill/decode/ttft/peak_mem.
- **Reaches:** archs with `apc_exact` or `apc_block`; signature includes `image_in`
  (APC hashes media payloads).
- **Gaps:**
  1. **No output-equivalence check.** APC is a cache, not a lossy transform: on-vs-off
     must produce identical tokens. `token_hit_rate` catches a cache that stops
     hitting, but not one that returns *wrong* cached tokens. Add `output_hash` and
     assert `apc-on == apc-off`.
  2. **No eviction/multi-turn test.** `apc_storage.py` evicts under pressure; a
     2-request pair never fills storage. A many-distinct-prefix scenario would exercise it.
  3. **No image prefix.** `image_in` is in scope but the prefix is text, so
     `hash_image_payload` is never run. A VLM with a shared image prefix would.

## kv_cache — cache construction & lifecycle
- **Guards:** `models/cache.py`.
- **Workload:** `single_generation` (cache grows to 512+128 = 640 entries).
- **Configs:** plain (unbounded) · bounded (512) · bounded-tight (128). 640 exceeds
  both bounds, so both rotate — the rotation path is exercised.
- **Metrics:** prefill/decode/ttft/peak_mem + **output_hash** (a bounded cache must not
  corrupt output within its window).
- **Reaches:** every arch (`requires: []`); signature `cache_kinds, hybrid_cache,
  kv_quant, bounded_kv, trimmable`.
- **Gaps:**
  1. **No plain-vs-bounded equivalence.** Output is compared head-vs-base per config,
     not across configs, so plain and bounded are never checked to agree on the tokens
     before the bound bites.
  2. **`trimmable` is a routing column but has no dedicated trim workload** (trim is
     exercised only incidentally via rotation).
  3. Caught real crashes already (this is the path that surfaced them), so coverage is
     working.

## kv_quant — key/value cache quantization
- **Guards:** `turboquant.py`, `generate/common.py`.
- **Workload:** `single_generation`.
- **Configs:** off · uniform-4 · uniform-3.5 (fractional-bit layer selection) ·
  turboquant-4 · late-start (`quantized_kv_start:1024`, conversion mid-generation) ·
  split-kv (asymmetric key/value bits). Enumerated to distinct branches, not the
  8-parameter cross product.
- **Metrics:** perf + output_hash.
- **Reaches:** archs with `kv_quant`.
- **Gaps:**
  1. **No quality bound vs off.** Quantized output *legitimately* differs from off, so
     `output_hash` only catches head-vs-base drift, not a quantization that tanks
     quality while keeping throughput. Add a greedy/KL agreement vs the off config.
  2. `quantized_kv_start:0` is the branch that surfaced the gemma2/phi3 crashes — the
     riskiest and correctly covered.

## chunked_prefill — chunked prefill in the AR path
- **Guards:** `generate/ar.py`, `generate/common.py`.
- **Workload:** `long_prompt` (4096 tokens, so chunking bites).
- **Configs:** off · chunk-512 (8 chunks) · chunk-2048 (2 chunks).
- **Metrics:** prefill/ttft/peak_mem + **output_hash** (no decode — chunking is a
  prefill-only concern).
- **Reaches:** archs with `chunked_prefill`.
- **Gaps:**
  1. **No chunk-vs-off equivalence.** Chunked prefill must be output-identical to
     unchunked; only head-vs-base is compared per config. This is where the `lfm2` conv
     crash surfaced, so crash-catching works — but a *silent* output divergence would
     not.

## speculative — draft-model speculative decoding
- **Status:** `enabled: false`.
- **Blocker:** the only meaningful metric is acceptance rate (throughput conflates
  draft quality with speed). The accepted count exists in
  `speculative/common.py` but `run_speculative_rounds` drops it through the
  mtp/eagle3/dflash yields. Enable once `GenerationResult` carries an accepted-tokens
  field and the probe reports it.
- **Configs (ready):** off · draft-3.

## Cross-cutting design gaps (apply to most components)
1. **Output-equivalence invariant.** Components that must preserve output
   (apc-on≈off, chunk≈off, plain≈bounded-within-window) are only checked head-vs-base
   per config. A bug that corrupts output identically in base and head passes. Add a
   per-component "must match the baseline config" assertion.
2. **"Did the config bite?" is not asserted.** If a scenario constant drifts (prompt
   shrinks below a chunk/bound), a config silently becomes a no-op and the CI reports
   green while testing nothing. Each config should assert it engaged (chunked >1 chunk,
   bounded rotated, apc cached the prefix).
3. **Functional metrics exist only for apc.** kv_quant/chunked/kv_cache lean on
   `output_hash` + perf. A quality metric (KL vs off) for lossy components and the
   equivalence checks above would close the correctness blind spot.
