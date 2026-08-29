# CI Change Rules

The change detector maps a complete pull-request diff to independent CI components. Components receive structured matches and decide which jobs, approval gates, or blockers to emit; they do not parse Git diffs themselves.

Rules live in `ci/change-rules.yaml`:

```yaml
schema_version: 1
rules:
  model_path:
    component: model_path
    include:
      - mlx_vlm/models/{model}/**
    exclude: []
```

`*` matches within one path segment, `**` matches recursively, and `{name}` captures one segment for the component. Rules may also require a path to be present or absent in the immutable base or head tree, and a more specific rule may supersede another rule for the same captured subject.

Adding another path category to an existing CI behavior requires only another rule. A new CI behavior also requires a component with a unique `name` and a `plan(matches, context)` method; unregistered matched components are reported as blockers.

## New model paths

`new_model_path` applies when `mlx_vlm/models/{model}` is absent from the base tree and present in the pull-request head. It supersedes ordinary model-path handling for that model.

The pull request must change `ci/model_path.yaml` and add a matching entry with configured synthetic and Hugging Face checkpoint sections. CI validates that configuration without executing contributor code, then emits an `awaiting_maintainer_approval` gate bound to the exact head commit and a digest of the submitted configuration.

The trusted control workflow reports that gate through one pull-request comment and a protected `apple-silicon-ci` environment. Approval recomputes the current pull-request head, verifies the configuration digest, and publishes an immutable runner job manifest; it does not execute model code itself.
