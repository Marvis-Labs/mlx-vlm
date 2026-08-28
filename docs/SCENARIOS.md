# CI scenarios, edge cases, and designed solutions

A map of what the benchmark CI must handle, to gauge the design. Each line is a
scenario and its designed response. Status: **[ok]** handled today, **[fix]**
a change we made or should make, **[open]** a known limitation to decide on.

## 1. Trigger & authorization
1. Maintainer comments `/ci run` on a PR → run. **[ok]** `issue_comment` + author-association gate.
2. Random/outside user comments `/ci run` → nothing. **[ok]** gated on `OWNER/MEMBER/COLLABORATOR`; without it, a fork PR is RCE on self-hosted HW.
3. Fork PR from a contributor + a maintainer runs it → runs, but the maintainer read the diff first. **[ok]** the gate is a deliberate human step.
4. `/ci run` on a plain issue (not a PR) → ignored. **[ok]** `if github.event.issue.pull_request`.
5. Comment `/ci runaway later` (prefix match) → currently triggers. **[open]** `startsWith` matches; low risk now that only maintainers pass the gate, but tighten to a word boundary.
6. `/ci run` typed 5× rapidly → one run, one comment. **[ok]** concurrency `cancel-in-progress` kills the older runs; `upsert` edits one marker comment.
7. `workflow_dispatch` with a bad PR number → route fails fast on the API call. **[ok]** manual, maintainer's problem; a clear error.
8. Comment edited to add `/ci run` → does NOT fire. **[ok]** `types: [created]` only; edits don't re-trigger.

## 2. Diff → routing
9. Model file changed → that arch across every enabled component it reaches. **[ok]**
10. Component file changed → one smallest sized rep per signature class × its configs. **[fix]** was alphabetical (routed onto 600GB MoEs); now smallest-sized.
11. Both a model and a component file → union, de-duplicated. **[ok]** `dedup across model+component` test.
12. Test/README/`.md` inside a model dir → nothing. **[ok]** `behavioural()` filter.
13. A brand-new model (not in the matrix) declared in `parity_models.yaml` → the parity gate, not the perf path. **[ok]**
14. New model NOT declared → a note asking to declare it; no cells. **[ok]**
15. A file that is neither model nor component (e.g. `README.md`) → "nothing to run" comment. **[ok]**
16. `.github/workflows/*` changed → **refused**; only the owner may change the trigger/permissions, by committing to default. **[fix]**
17. `ci/` harness changed → runs, but the comment leads with a ⚠️ "modifies CI harness" banner. **[fix]**
18. `capabilities.json` (the matrix) changed → ⚠️ warned (it drives routing) even though it lives outside `ci/`. **[fix]**
19. A signature class whose only members are unsized → a "no sized model" note, not a silent gap. **[ok]**
20. A model in the matrix but missing metadata → noted "cannot size a cell". **[ok]**

## 3. Device sensing & capacity
21. 16GB mini vs 512GB Studio, same cell → each picks the largest precision it holds. **[ok]** zero-config; router ships all variants.
22. A cell needs mem-256, only a 128GB runner is online → the cell queues for a bigger runner. **[ok]** tier labels; **[open]** if none ever appears it sits pending (see 46).
23. A model too big for even the smallest variant on this device → declined "too small". **[fix]** declines cleanly, not a crash.
24. Adding a new machine → clone `ci-runner`, run it; it senses RAM→tiers and registers. **[fix]** no per-device config.
25. A device that also has ≥96GB → advertises `parity`, can run the correctness gate. **[ok]**
26. Heterogeneous fleet, one component run → big cells land only on the big runner, small on either. **[ok]** labels are the router's only knowledge of HW.
27. Disk-tight device with cached models → prune old caches to make room *before* the readiness gate rejects it. **[fix]** ordering bug fixed.

## 4. Measurement integrity
28. Background VM/exo load on a runner → decline rather than measure noisy numbers. **[ok]** readiness gate; **[open]** on a contended fleet almost everything declines.
29. Two revisions of the same package in one process → impossible; each measured in a fresh subprocess with `PYTHONPATH` at its tree. **[ok]**
30. Thermal/background drift loading onto the 2nd revision → interleaved, counterbalanced A/B cancels it. **[ok]**
31. Three noisy samples inflating stddev → median absolute deviation, not stddev, as the noise floor. **[ok]**
32. A change smaller than the noise → "inconclusive/noise", not a false regression. **[ok]** 2·SE + floor.
33. Peak memory estimate wrong (weights×multiplier) → replaced by the measured peak once a cell has run. **[ok]** store feedback loop.
34. Model download/load outruns the budget → clean "timed out", not a crash into the placeholder. **[fix]** `TimeoutExpired` caught.

## 5. Failures, declines & crashes
35. A cell's model code crashes (e.g. lfm2 chunked-prefill conv error) → ❌ with the exception in the note, status red. **[ok]** real bug caught this way.
36. Device busy → ⊘ "busy", counted apart from failures, re-runnable. **[fix]** was indistinguishable from a crash.
37. Gated model, no `HF_TOKEN` → declined early "gated model needs HF_TOKEN", not an opaque mid-run 401. **[fix]**
38. Some cells pass, some crash → headline "N cell(s) failed" (not hidden behind "no regression"), status red. **[fix]**
39. Runner dies mid-cell → no artifact; the cell stays ⏳; the report still renders the rest. **[ok]** partial, never a hang.
40. Every cell declined (fleet saturated) → "all devices busy — re-run when idle", green status (environmental). **[fix]**
41. Probe prints no result JSON → "bad output" error with captured stderr, not a silent pass. **[ok]**

## 6. Concurrency & scale
42. Two PRs run at once → separate concurrency groups, both proceed. **[ok]**
43. Same PR run twice → the second cancels the first. **[ok]**
44. Many concurrent cells hammer the runner-script fetch → 403 rate-limit. **[fix]** authenticated fetch (60→5000/hr); now the local clone, no fetch.
45. Two runs finish together and both push results-data → the loser was dropped. **[fix]** retry re-syncs and re-ingests.
46. A run with no matching runner → cells queue up to ~24h then expire; the comment shows ⏳ the whole time. **[open]** consider a "no runner for mem-N" note at route time.
47. A large component sweep on one runner → serial, ~9 min/cell. **[open]** batch a model's configs into one job (load once) and/or add runners.

## 7. Reporting & the comment
48. First thing a reviewer sees → title + one table, no dropdown, no ASCII chart. **[fix]** barebone by request.
49. Slow sweep → the comment fills in cell-by-cell, not one jump at the end. **[fix]** report job polls artifacts live.
50. Status invisible on the PR (issue_comment posts no check) → a commit status (pending→success/failure) on the head SHA. **[fix]**
51. A wide 11-column table scrolling off-screen → 4 speed columns; functional metrics still shape the status. **[fix]**
52. Regression in the median vs a single noisy cell → headline follows the median; per-cell marks show the individual verdict. **[ok]** (a possible confusion: cell 🔴 while headline "no regression").
53. Comment must survive a re-render overwriting it → notes (warning/refusal) uploaded and passed to the final render. **[fix]** were dropped.

## 8. Persistence & data
54. Results must outlive the ephemeral cloud job → SQLite pushed to a `results-data` branch. **[ok]**
55. Declarative data (matrix, thresholds) vs observational (measurements) → the first in the repo, the second in SQLite. **[ok]**
56. Peak-mem value stored as a summary dict instead of a scalar → `record()` extracts the median. **[ok]** verified live.
57. results.db grows unbounded (append-only) → fine at this scale; unlocks history/trend tracking. **[open]** prune or roll up eventually.
58. Historical trend of a metric across commits → data is there; not yet surfaced in the comment. **[open]** add a sparkline.

## 9. Runner lifecycle
59. Reboot → runner rejoins unattended (auto-login → RunAtLoad LaunchAgent → online). **[ok]** validated (177s).
60. Runner crashes → KeepAlive restarts it. **[ok]**
61. Network drop → reconnects (subsumed by the reboot test). **[ok]**
62. `ci-runner` is private → a bare `curl|bash` cannot auth. **[fix]** the developer clones it (the clone is the auth) and the runner self-updates via `git pull`.
63. Stale local device.sh after the repo moves → `git pull --ff-only` each run (best-effort); once a stale cached copy ran old logic. **[fix]**
64. Seeded clone without `.git` → detect the script by `bin/device.sh`, not `.git`. **[fix]**

## 10. Security & supply chain
65. Malware hidden in a workflow/CI file via a PR → refused (see 16); can't be blessed by a green run. **[fix]**
66. A PR weakening `protected_paths.yaml` itself → refused (the list is read from default, not the PR). **[fix]**
67. Contributor code on self-hosted HW → only after a maintainer reads the diff and runs it. **[ok]** the association gate.
68. Secrets (HF_TOKEN) → a repo secret, passed only to the measure step, never logged. **[ok]**
69. A runner repo made public exposes no secrets, but keep it private → developer-clone model, not public bootstrap. **[ok]** per design.

## The through-line of friction
Most complexity is the CI compensating for **contended, heterogeneous, sometimes-slow hosting**. A quiet dedicated runner (M5 Max, M3 Ultra) removes most of it: no declines, full sweeps, live parity. The residual design work is (a) batch configs per model to cut load count, (b) cap or flag big-only component signatures, (c) surface "no runner for this tier" and historical trends.
