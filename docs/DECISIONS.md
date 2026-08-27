# CI policy decisions

Recorded here so they are reviewable and changeable in one place.

## Trigger: on-demand `/ci run`

The benchmark runs only when a maintainer comments `/ci run` on a pull
request. Not on every push (14–30 cells per run on a serial fleet is too much
hardware time for every commit) and not only on approval (a regression should
be visible during review, not after). On-demand keeps the maintainer in
control of when hardware is spent, and it is the cheapest policy that still
catches regressions before merge. Re-run at will; the comment updates in place.

## Fork-PR execution: maintainer-gated, never automatic

A self-hosted runner executes the pull request's code on our own hardware, so
the trigger is a deliberate maintainer action and never an automatic reaction
to a fork push. `/ci run` is the gate: a maintainer reads the diff, then runs
it. The workflow reacts to `issue_comment`, not to `pull_request` from a fork,
so contributor code cannot run unreviewed. Repository fork-PR approval requires
approval for all external contributors as a second line.

## What is configured vs sensed

Nothing about the fleet is configured. A device senses its own memory and
advertises the tiers it can hold; the router ships every precision and each
device picks the largest it fits. Adding a machine needs no change here.

## What is a file vs a database

Declarative data (capability matrix, routing, thresholds, model metadata)
lives in the repository, versioned with the code that gives it meaning.
Observational data (measurements) accumulates and lives in SQLite, persisted to
the `results-data` branch so it survives an ephemeral cloud run.
