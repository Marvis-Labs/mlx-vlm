# Implementation Path — Benchmark and Verification CI

Ordered by dependency. Each task states what it produces and how you know it is finished.
Tasks in the same group are independent of each other and can proceed in parallel.

## Group 0 — Prerequisites

These block everything and are not engineering work.

| # | Task | Done when |
| --- | --- | --- |
| 0.1 | Resolve runner registration with the mlx-vlm maintainer. Self-hosted runners require repository or organisation admin, which the current push access does not grant. | Either admin is granted, or a decision is made to run the orchestrator ourselves |
| 0.2 | Decide the fork-PR execution policy. A self-hosted runner on a public repository executes contributor code on our hardware. | A trigger policy is written down: maintainer-applied label only, never raw pull_request from forks |
| 0.3 | Bring the intended devices onto the tailnet. | Each device appears in `tailscale status` and accepts key-based SSH |

## Group 1 — Device bootstrap

A device that cannot reach a known-good state must decline work rather than produce a
measurement that is quietly wrong.

| # | Task | Done when |
| --- | --- | --- |
| 1.1 | Write a single bootstrap script, idempotent, that provisions a bare macOS machine end to end. | Running it twice on a fresh device leaves the same state and the second run changes nothing |
| 1.2 | Key-based SSH; disable password authentication. | Unattended SSH succeeds with no password on the wire |
| 1.3 | Xcode Command Line Tools, installed headlessly. | `git --version` succeeds over a non-interactive SSH session |
| 1.4 | `uv`, a pinned Python, and a locked environment. | The lockfile resolves identically on two devices |
| 1.5 | Power and sleep policy: sleep, disk sleep and Power Nap disabled; Low Power Mode off; mains power required. | Device stays reachable overnight with no session attached |
| 1.6 | Spotlight and Time Machine exclusions for the cache and working directories. | Indexing does not touch the model cache during a run |
| 1.7 | Wired memory limit raised at boot via a launchd job. | The setting survives a reboot |
| 1.8 | Readiness check reporting load, free memory, free disk, thermal state and stray processes. | Returns a clear ready or not-ready verdict with the reason |
| 1.9 | Agent installed as a launchd service. | Survives reboot without anyone connecting to the machine |

## Group 2 — The measurement

This is the first thing to build and the only thing that matters until it is correct.
Everything in later groups is machinery for delivering work to this and returning its result.

| # | Task | Done when |
| --- | --- | --- |
| 2.1 | Define the cell schema: architecture, model repository, regime, component, configuration, memory requirement, both revisions, metrics. | Router and device agree on one serialised form and neither reinterprets it |
| 2.2 | Implement a cell body that executes one cell and emits a result with an environment fingerprint. | One cell runs by hand on a device and produces a result file |
| 2.3 | Implement warmup and measured iterations, separated. | Removing warmup visibly changes the first measured iteration |
| 2.4 | Implement the pair protocol: warm the cache once, run both revisions back to back, never clear between halves. | Both halves report identical cache state in their fingerprints |
| 2.5 | Characterise run-to-run variance by running an unchanged revision repeatedly. | A documented tolerance per metric, derived from the observed spread |
| 2.6 | Set regression thresholds above the measured noise floor. | Thresholds are justified by 2.5 rather than chosen |

Task 2.5 is the exit condition for the entire first phase. Until the noise floor is known, no
threshold can be defended and any regression report is unfalsifiable.

## Group 3 — Routing data

Three declarative files, all committed to the repository, all reviewable in a diff.

| # | Task | Done when |
| --- | --- | --- |
| 3.1 | Add a `_meta` provenance header to the capability matrix and a test asserting the recorded probe version matches the current probe. | A probe change reports "regenerate" instead of several hundred phantom drifts |
| 3.2 | Decide whether the matrix gains a regime dimension. | Decided and recorded, before further rows are added |
| 3.3 | Add a turboquant column, and any other column the routing map requires. | A change to `turboquant.py` selects a non-empty architecture set |
| 3.4 | Extend `cache_kinds` to cover the classes constructed by the generation path rather than by `make_cache`. | A change to any of the sixteen cache classes selects a non-empty set |
| 3.5 | Write the routing map from component paths to governing columns. | Every path under `mlx_vlm/` either maps to a column or is explicitly assigned to the fixed set |
| 3.6 | Write the knob space file: per component, its paths, its columns, and its enumerated configurations. | Each component declares a bounded configuration list, not a cross product |
| 3.7 | Add model metadata: concrete repository, parameter count, precision, and a memory estimate per cell. | The router can answer whether a given cell fits a given device |
| 3.8 | Define the fixed architecture set for system changes. | Committed, and exercising generation rather than only response format |
| 3.9 | Record the remaining architectures in the matrix. | Coverage is complete, or the uncovered set is explicitly justified |

Tasks 3.2 and 3.9 are ordered: widening the matrix after recording the rows means re-probing
all of them.

## Group 4 — The router

| # | Task | Done when |
| --- | --- | --- |
| 4.1 | Classify a diff into new-model, model, component and system changes. | A set of hand-written example diffs classifies correctly, including multi-class diffs |
| 4.2 | Selection stage: diff to architecture set, reduced by capability signature for component changes. | Selecting on a cache class returns representatives, not the full membership |
| 4.3 | Expansion stage, model path: one architecture across every path it supports. | Expansion matches the architecture's row in the matrix |
| 4.4 | Expansion stage, component path: selected architectures across the component's declared configurations. | Every configuration in the knob space appears in the output |
| 4.5 | Capacity filtering: substitute within a signature class when the default representative does not fit the fleet. | A signature is never dropped merely because its default representative is too large |
| 4.6 | Deduplicate the union across classes. | A diff touching two architectures and a shared component emits no duplicate cells |
| 4.7 | Report unrouted paths rather than silently selecting nothing. | A change to an unmapped file produces a visible warning |

## Group 5 — Orchestration

| # | Task | Done when |
| --- | --- | --- |
| 5.1 | Workflow triggered by the policy from 0.2, computing the cell set and emitting it as a job matrix. | A pull request produces the expected matrix |
| 5.2 | Runner labels encoding chip, memory and installed gates. | A cell requiring high memory cannot land on a device that lacks it |
| 5.3 | Job-start self-check calling the readiness check from 1.8, failing fast when not ready. | An unready device requeues rather than reporting a bad measurement |
| 5.4 | Both revisions executed within a single job so a pair never spans devices. | Result fingerprints confirm one device per pair |
| 5.5 | Result storage. | Results are queryable across revisions and devices |
| 5.6 | Reporting back to the pull request, comparing against the thresholds from 2.6. | A comparison appears on a pull request with no manual step |

## Group 6 — Correctness gate

| # | Task | Done when |
| --- | --- | --- |
| 6.1 | Detect a newly added model directory as a distinct change class. | A new model routes to the parity gate and not to the performance gate |
| 6.2 | Parity harness: unquantized weights, reference in float32 on CPU, both models resident. | Reported divergence is stable across repeated runs |
| 6.3 | Report greedy agreement and KL divergence, mean and maximum across positions. | Both quantities appear in the result |
| 6.4 | Record per-architecture thresholds as committed expected values. | A later change that moves them is reported as drift |
| 6.5 | Restrict the gate to devices with memory for two model copies. | The gate never lands on a device that cannot hold both |

## Group 7 — Fleet

| # | Task | Done when |
| --- | --- | --- |
| 7.1 | Bootstrap a second device from the same script with no manual steps. | Two devices differ only in their labels |
| 7.2 | Confirm cross-device comparability, or record the tolerance at which results from different devices may be compared. | Documented; pairs remain pinned to one device regardless |
| 7.3 | Health reporting for devices that stop claiming work. | A device that goes dark is visible without anyone checking manually |

## Critical path

Group 0 gates everything and is mostly other people's decisions, so start it immediately and
in parallel with technical work. Group 1 then Group 2 is the shortest path to a defensible
number. Group 3 can proceed alongside Groups 1 and 2 because it is declarative and needs no
hardware. Group 4 requires Group 3. Group 5 requires Groups 2 and 4. Groups 6 and 7 follow.

The single most important ordering constraint is that task 2.5 precedes anything that reports
a regression, and the second is that task 3.2 precedes task 3.9.
