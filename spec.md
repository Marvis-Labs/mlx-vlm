# Design Plan — Benchmark and Verification CI

## 1. Overview

This document describes a continuous integration system for mlx-vlm that verifies correctness
and measures performance on real Apple Silicon hardware, per pull request. It exists because
the standard test suite is fully mocked: it runs in roughly three minutes, loads no weights,
and therefore cannot detect a performance regression or a numerical divergence in a model
implementation.

The system has two responsibilities, applied to different kinds of change:

- Verify that a newly added model matches its reference implementation numerically.
- Verify that a change to existing code does not regress performance on the paths it affects.

Both run on physical devices, because both measure properties that only exist on real
hardware. Neither can be approximated in a cloud runner.

The design principle throughout is that the system selects the smallest set of work that
covers the change. A pull request touching one cache class should not benchmark two hundred
architectures, and the mechanism that reduces it must be sound rather than a sampling
heuristic.

## 2. Scope

In scope: selecting work from a diff, dispatching it to devices, executing it reproducibly,
storing results, and reporting back to the pull request.

Out of scope: replacing the existing unit test suite, which remains the fast hermetic gate and
runs unchanged; training or conversion workflows; anything requiring hardware other than Apple
Silicon.

## 3. Change classification

Every pull request is classified by the paths it touches. A pull request spanning several
classes produces the union of their selections, deduplicated before dispatch.

| Class | Trigger | Gate | Selection |
| --- | --- | --- | --- |
| New model | A new directory under `mlx_vlm/models/` | Correctness | Parity against the reference implementation |
| Model | An existing file under `mlx_vlm/models/<arch>/` | Performance | That architecture, across every regime it supports |
| Component | A shared execution path (`apc.py`, `models/cache.py`, `turboquant.py`) | Performance | Every architecture reaching the component, reduced to one per capability signature |
| System | `server/`, request or response schemas, CLI entry points | Performance and protocol | A fixed, committed set of architectures |

Deduplication is not an optimization detail. A pull request touching two architectures and a
cache class will frequently select overlapping cells, because the architectures chosen to
represent a component are themselves architectures. Three touched areas do not imply three
independent runs.

## 4. Correctness gate for new models

A newly added model has no previous revision, so there is nothing to compare it against in
time. It is instead compared against its reference implementation in the `transformers`
library, on identical weights and identical inputs.

Two quantities are reported:

- Greedy agreement: the fraction of positions at which the two implementations select the same
  next token. This is the property users observe directly.
- Kullback-Leibler divergence: the divergence of the mlx next-token distribution from the
  reference distribution, reported as mean and maximum across positions. This detects
  degradation that greedy agreement hides, where the argmax happens to survive but the
  distribution has shifted.

Three constraints govern how this comparison is set up, and violating any of them makes the
result meaningless.

The comparison must use unquantized weights. Quantization noise is larger than the
implementation differences being looked for, and comparing a quantized mlx model against a
full precision reference measures the quantizer rather than the port.

The reference must run in float32 on CPU. Accelerated backends in `transformers` introduce
their own numerical differences, which would be attributed to the mlx implementation.

Both models must be resident simultaneously, so the device must have memory for two copies of
the model plus activations. This gate therefore runs only on high memory devices.

Thresholds are recorded per architecture rather than set globally, because acceptable
divergence varies with model depth and precision. A new model's thresholds are established
when it is added and are then treated as expected values: a later change that moves them is a
reported drift, not a silent pass.

## 5. Performance gate for existing code

A change to existing code is measured by running the same work on the revision before the
change and the revision after it, on the same device, back to back. The reported result is the
delta.

Measurements collected per run include prefill throughput, decode throughput, time to first
token, peak memory, and total wall time. The exact set is fixed per cell type so that results
remain comparable across revisions.

Both halves of a comparison are treated as a single indivisible unit of work. Splitting them
across devices would compare measurements taken on different hardware, which is meaningless
even when both devices are nominally identical.

## 6. Capability matrix

The matrix records what each architecture can be run with. It is generated by probing rather
than hand declaration: each architecture is instantiated small from its real `config.json`
with no weights, and the resulting model is inspected for the paths it supports. Probing is
the appropriate choice at this scale, since mlx-vlm contains roughly two hundred model
directories and hand maintained declarations for that many entries drift silently. Comparable
systems declare capabilities in code and accept that drift; this design avoids it.

The data is split across two files by maintenance model. The split should be preserved as the
matrix grows.

| File | Content | Maintained by |
| --- | --- | --- |
| `mlx_vlm/tests/models_registry.py` | Declared inputs: reference repository, config overrides, skip reasons | Hand edited, comments permitted |
| `mlx_vlm/tests/capabilities.json` | Probed facts: one row per architecture, one field per capability | Regenerated, never hand edited |

The matrix is committed to the repository rather than served from the orchestrator database.
Its second job, after routing, is drift detection: when a pull request changes what an
architecture supports, the change appears in the diff where a reviewer sees it. That property
is lost the moment the data moves behind a service. A file also corresponds to the code at any
revision by construction, which a single mutable database state cannot do for a branch cut
weeks earlier.

The file carries a provenance header recording the probe version, the mlx-vlm revision and the
generation date, and a test asserts the recorded probe version matches the current probe.
Without this, a change to the probe is indistinguishable from a change to the models, and the
two require opposite responses; the assertion converts a probe change into an instruction to
regenerate rather than several hundred phantom drifts.

## 7. Routing

Routing requires the inverse of the matrix: given a changed file, which capability column
governs it. This mapping cannot be derived from the code and is maintained explicitly
alongside it.

| Component path | Governing column |
| --- | --- |
| `apc.py`, `apc_adapters.py`, `apc_storage.py` | `apc_exact`, `apc_block` |
| `models/cache.py` | `cache_kinds` |
| `generate/ar.py`, `generate/common.py` | `chunked_prefill`, `speculative` |
| `turboquant.py` | requires a dedicated column |
| `sample_utils.py` | none; affects all architectures equally, uses the fixed set |
| `server/` | none; uses the fixed set |

Two known gaps must be closed before component routing is complete.

`turboquant.py` has no governing column. The existing `kv_quant` field records uniform
key/value quantization and does not distinguish the turboquant path, so a change to that file
selects nothing.

The `cache_kinds` field covers four of the sixteen classes in `models/cache.py`. The recorded
value reflects what an architecture constructs at rest, through `make_cache`. The remaining
classes — batched, quantized, chunked, buffered and prefix variants — are constructed later by
the generation path, when batching wraps a cache, quantization converts it, or prefix caching
substitutes it. Closing this means recording the cache class per architecture and per regime
rather than per architecture alone, which adds a dimension to the matrix. That decision should
be made before the remaining architectures are recorded, since widening afterwards requires
re-probing all of them.

Any path not covered by the map falls back to the fixed set. Unrouted changes are reported as
such rather than silently selecting nothing.

## 8. Cell selection

A cell is one measurable combination of architecture, regime and configuration, and is the
unit of work the system schedules.

Component changes select cells by capability signature, where a signature is the tuple of
every capability field. Architectures sharing a signature take an identical path through the
component, so measuring more than one of them measures the same code twice. The reduction is
substantial: across the architectures recorded so far, thirty-six construct a plain `KVCache`
but collapse to four distinct signatures, and eight construct a `RotatingKVCache` but collapse
to two. Selecting by signature is what makes component changes affordable, and it is sound
rather than a sample.

Model changes do not use signatures. A change to an architecture is measured across every
regime that architecture supports, since the objective is to detect a regression in that
specific implementation.

System changes use a fixed set committed to the repository. Random selection is not acceptable:
before and after comparison requires both halves to exercise identical work, and a set that
varies between runs converts the signal into noise. The fixed set must exercise generation and
not only response format, because `server/generation.py` maintains a generation path separate
from `generate/ar.py` and can move generation semantics independently.

## 8a. Router

The router converts a diff into a list of cells in two stages. Selection determines which
architectures are involved. Expansion determines what is actually executed for each of them.
The two change classes expand along different axes, and this asymmetry is the substance of the
design rather than an implementation detail.

For a model change the architecture is fixed and the components vary. The router expands that
one architecture across every path it supports — its cache variants, its quantization modes,
its prefix caching behaviour, its input modalities — because the objective is to detect a
regression anywhere in that specific implementation. The set of paths is read directly from
the architecture's row in the capability matrix.

For a component change the component is fixed while both the architectures and the component's
own arguments vary. Selecting one architecture per capability signature and running the
component once under its default arguments would leave most of the changed code unexercised: a
change to prefix caching that only manifests at a particular block size, or a change to
quantization that only manifests at fractional bit widths, would pass. The router therefore
expands each selected architecture across a declared set of configurations for that component.

The capability columns that a component declares serve two purposes. They determine which
architectures reach the component, and they define the scope of the signature used to
deduplicate them. Scoping is necessary rather than cosmetic: input modality says nothing about
how an architecture exercises the cache implementation, so including it when routing a cache
change multiplies representatives without adding coverage, while excluding it when routing a
prefix-caching change would omit real behaviour, since prefix caching hashes media payloads.
The same column is load-bearing for one component and noise for another, so each component
names its own scope.

Beyond the architectures and the component's configurations, a component change may also need
to run under more than one execution regime. The interaction between a component and the cache
implementation is already covered, because the cache field appears in every component's scope
and selecting one architecture per signature therefore exercises each cache class once.
Regimes that no capability column captures — single stream against batched execution, for
instance — are declared per component instead. This set is small and deliberate: expanding a
component change across every path its representative architectures support produces several
hundred cells, most of which exercise the component identically.

The risk here is asymmetric and the default should reflect it. Omitting a genuine interaction
produces a missed regression with no visible symptom, while including an unnecessary one costs
a few minutes of device time and is apparent in the run record. Regimes are therefore included
when their independence is uncertain, and an exclusion is recorded with its justification so
that it remains reviewable.

Device capacity enters at expansion, not at selection. The router knows each cell's memory
requirement, derived from model size, precision and cache configuration. For component changes
it prefers architectures whose representative models fit the available fleet, choosing an
alternative member of the same signature class where the default representative is too large.
Because members of a signature class take an identical path through the component, this
substitution costs no coverage.

The knob space for each component is declared in a file maintained alongside the routing map.
It names the paths that belong to the component, the capability columns that govern it, the
configurations to run, and the scenario under which the component becomes observable.

The scenario is not uniform across components and cannot be defaulted. Some components are
measurable in a single generation call. Others are not observable at all without a specific
interaction pattern: prefix caching only does work on a second request sharing a prefix with
the first, so a single-request cell measures nothing and would report a change of zero
regardless of what the code does. Batching requires concurrent sequences. Speculative decoding
requires enough tokens for acceptance behaviour to stabilise. Each component therefore declares
its scenario, and the cell body executes that scenario rather than a fixed one.

Where a component exposes functional counters, those are collected alongside timing and are
the primary signal. Prefix caching reports matched tokens, token hit rate and rejection
reasons; a change that leaves throughput untouched but drops the hit rate is a regression that
timing alone would miss. Timing is the fallback for components that expose nothing.

Configurations are enumerated explicitly rather than produced as a cross product of individual
knobs, for the same reason architectures are reduced by signature: an unconstrained product
grows past what the fleet can execute while a pull request is still open. Key/value cache
quantization alone exposes eight parameters, whose full product across the representative
architectures is several hundred cells for a single change.

The configuration set is resolved against both revisions. Configurations valid in both are
compared. A configuration introduced by the pull request has no counterpart in the base
revision and is measured once and reported as new rather than as an infinite improvement, and a
configuration the pull request removes is reported as withdrawn.

A cell is the contract between the router and the device, and carries everything required to
execute it without further interpretation: architecture, concrete model repository, regime,
component under test, the configuration of that component, the memory requirement, the two
revisions to compare, and the metrics to collect.


## 9. Orchestration

All communication is outbound from the machines that matter. No component of this system
accepts an inbound connection from the public internet, and no runner accepts an inbound
connection at all.

The orchestrator is a single process exposing a SQLite-backed HTTP API. It performs three
loops:

- It polls the GitHub API for new and updated pull requests, resolves each into a cell set, and
  records the cells as pending work.
- It serves the queue to devices, which claim work.
- It posts results back to the pull request through the GitHub API.

Polling rather than webhooks is a deliberate choice for the initial system. A webhook requires
public ingress to the orchestrator; polling requires nothing, and the latency it adds is
negligible against runs measured in minutes. Webhook delivery can be added later without
changing anything else.

Responsibility for deciding what runs is split. The orchestrator determines which cells a pull
request requires and what each cell needs in order to run, including a memory requirement
derived from model size and configuration. The device determines whether it can accept a given
cell. The orchestrator proposes and the device disposes: a device may decline or return a
claimed cell, which returns to the queue for another device.

This split matters because the orchestrator cannot know a device's live state — its thermal
condition, its free memory, whether a previous run left it dirty — and the device cannot know
what a pull request requires. Each side decides what only it can observe.

## 10. Device lifecycle

A device runs a small agent that loops through a fixed sequence. Every stage exists to protect
reproducibility, and a device that cannot reach a known-good state declines work rather than
producing a measurement that will be silently wrong.

1. Register, advertising chip, core counts, physical memory, usable memory fraction, and
   installed toolchain versions.
2. Verify readiness: confirm no other benchmark process is running, confirm system load is
   below threshold, confirm free memory and free disk exceed the requirement, confirm thermal
   state is nominal.
3. Claim a cell whose requirement fits, or wait.
4. Prepare: check out both revisions, build the environment against pinned dependencies, and
   warm the model cache.
5. Execute both revisions back to back, with warmup iterations preceding measured iterations.
6. Report results, including the environment fingerprint under which they were produced.
7. Clean up and return to readiness.

A device is single tenant. Concurrency is fixed at one, and this is a correctness requirement
rather than a tuning parameter: two jobs sharing a device contend for memory bandwidth and
GPU, and the resulting timings reflect neither change under test.

## 11. Reproducibility

The measurements this system produces are only useful if run to run variance on unchanged code
is small relative to the regressions it is meant to detect. Establishing that variance
empirically, on the actual hardware, is a prerequisite to reporting any result — a detector
tuned below the noise floor produces false regressions indefinitely.

The following are controlled explicitly.

| Factor | Control |
| --- | --- |
| Concurrent work | Single tenancy; readiness check refuses to start if load is elevated |
| Model cache state | Warmed before the pair, untouched during it, identical for both halves |
| Compiled kernel cache | Warmup iterations precede measured iterations |
| Dependency versions | Pinned and identical across both halves; recorded in the result |
| Thermal state | Cooldown between runs; thermal check before starting |
| Background system activity | Indexing and backup excluded from working directories |
| Power state | Mains power required; low power mode disabled |
| Process state | A fresh process per run; no state carried between revisions |

One control is commonly stated in a way that inverts its purpose. Clearing the model cache
between the two halves of a comparison is harmful, not helpful: the weights are identical for
both halves, so clearing between them causes the second half to re-download and measures
network throughput. The cache is cleared, if at all, before a pair begins, and is then warmed
so that both halves start from the same warm state. Cache clearing is a disk reclamation
policy, not an isolation mechanism.

Every result records the environment that produced it — device identity, dependency versions,
and the readiness measurements taken immediately before the run. A result that cannot be
attributed to a known environment cannot be compared against another, and results whose
fingerprints differ are not compared.

## 12. Data

Storage is split by what causes a value to change.

| Data | Changes when | Stored as |
| --- | --- | --- |
| Capability matrix, routing map, fixed sets, parity thresholds | The code changes | Files in the repository |
| Measurements, run history, device registry, work queue | Time passes | SQLite behind the orchestrator |

Declarative data that is only meaningful in the context of a particular revision belongs in
the repository, where revision correspondence is free. Observational data accumulates without
bound, is queried across revisions and devices, and cannot be committed. Placing the matrix in
the database would forfeit drift detection, review visibility and revision correspondence in
exchange for query performance that is not required at this scale.

The orchestrator schema covers devices, runs, cells, claims, results and reports. SQLite is
sufficient: the write rate is bounded by how fast physical hardware can produce measurements,
which is a few per minute at most.

## 13. Phases

Each phase produces something usable on its own, and later phases assume the earlier ones.

| Phase | Objective | Exit condition |
| --- | --- | --- |
| 1 | A single cell executes reproducibly on one device, invoked by hand | Repeated runs of an unchanged revision agree within a measured, documented tolerance |
| 2 | Orchestrator with schema, queue and claim protocol; one device agent | A manually created run reaches a device and returns a stored result |
| 3 | Routing from a diff to a cell set | A pull request produces the correct cell set, verified against hand-computed expectations |
| 4 | Reporting to the pull request | A comparison appears on a pull request automatically |
| 5 | Parity gate for new models | A newly added model reports agreement and divergence against the reference |
| 6 | Fleet | Additional devices register and claim without configuration changes |

Phase 1 is deliberately first. Every later phase is machinery for delivering work to a
measurement and returning its result; if the measurement itself is not reproducible, that
machinery reports noise at scale. The tolerance established in phase 1 sets the threshold at
which every subsequent phase can claim a regression.
