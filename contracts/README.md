# Execution Contracts

`experiment_execution_capabilities.v1.json` is the canonical, versioned source for experiment execution capabilities. It records the current semantics of each `PlannedExperiment` field for every supported task/runner pair. The classifications mean:

- `executed`: the runner uses the field directly.
- `conditional`: the runner uses the field only when the recorded prerequisite is active.
- `metadata_only`: the field belongs to planner or control-plane provenance, not training semantics.
- `unsupported`: the runner does not execute the requested semantic.

The contract is intentionally descriptive in Fidelity PR 1. Scheduling and workers do not call the new normalizers yet, so adding the source of truth does not change experiment behavior.

After editing the canonical JSON, regenerate the embedded Go data and packaged Python data:

```sh
python3 scripts/generate_experiment_execution_capabilities.py
```

CI and local checks can detect drift without rewriting files:

```sh
python3 scripts/generate_experiment_execution_capabilities.py --check
```

The shared fixture file is consumed by Go and Python golden tests. The Go coverage test also reflects over `PlannedExperiment`, `Preprocessing`, and `AugmentationPolicyConfig`, so a newly added plan field must be classified in every profile before tests pass.
