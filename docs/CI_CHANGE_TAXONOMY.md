# Taxmonomy of Changes

Use this checklist to design and implement CI coverage for each class of change.

| Status | Change class | Examples | Required validation |
|---|---|---|---|
| - [ ] | Existing model | `models/qwen3_vl/**` | Synthetic unit tests, real load/generate, affected component cases |
| - [ ] | New model | New architecture directory/config | Require synthetic and checkpoint manifest data, then await maintainer approval before runtime testing |
| - [ ] | Removed/renamed model | Deleted directory or changed `model_type` | No dangling imports, registry cleanup, compatibility/error behavior |
| - [ ] | Shared component | Cache, attention, RoPE, sampling, APC, batching | Representatives for every distinct capability signature |
| - [ ] | Model registry/dispatch | Discovery, `utils.load`, generation dispatch | Registry enumeration plus small load smoke across architecture families |
| - [ ] | Media processing | Tokenizers, processors, prompt utilities, image/audio/video loading | Fixture tests plus one real model per affected modality |
| - [ ] | Weight lifecycle | Sanitization, conversion, quantization, LoRA/adapters | Raw-HF → converted → load → generate; quantized and unquantized controls |
| - [ ] | Generation engine | AR, diffusion, streaming, batching, speculative decoding | Mode-specific invariants and cross-mode equivalence where expected |
| - [ ] | API/CLI/configuration | OpenAI, Anthropic, Responses, realtime, schemas, CLI | Contract tests, default compatibility, invalid-input behavior, endpoint smoke |
| - [ ] | Tool/structured output | Tool parsers, constrained generation | Parser fixtures and end-to-end structured-output requests |
| - [ ] | Training | Trainer, LoRA/DoRA, datasets | Forward/backward, tiny optimization step, adapter save/reload |
| - [ ] | Dependencies/packaging | `requirements.txt`, `uv.lock`, `pyproject.toml` | Clean install, wheel install, full unit suite, dependency canary |
| - [ ] | CI/evaluation harness | Workflows, router, thresholds, benchmark code | Harness self-tests; protected from evaluating its own untrusted changes |
| - [ ] | Tests/docs only | Tests, README, docs | Cloud-only checks; no Apple runner unless behavior also changed |
