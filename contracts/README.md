# Execution Contracts

`experiment_execution_capabilities.v1.json` is the canonical, versioned source for experiment execution capabilities. It records the current semantics of each `PlannedExperiment` field for every supported task/runner pair. The classifications mean:

- `executed`: the runner uses the field directly.
- `conditional`: the runner uses the field only when the recorded prerequisite is active.
- `metadata_only`: the field belongs to planner or control-plane provenance, not training semantics.
- `unsupported`: the runner does not execute the requested semantic.

Fidelity PR 2 uses the embedded Go representation to resolve a canonical `execution_spec_v1` when an experiment job is queued. The spec records the requested config, accepted semantic config, capability version, task, runner, requested-config hash, and accepted-spec hash. Legacy flat job keys remain in place for current workers.

Fidelity PR 5 snapshots that accepted spec independently from mutable job configuration and creates one execution record per assigned attempt. Workers may append authenticated `INITIALIZED` and `FINALIZED` observations through `POST /jobs/:id/execution-observations`; the server bounds and redacts evidence, derives the realized hash and fidelity verdict, and rejects callbacks from stale attempts. Records are available through `GET /jobs/:id/execution-record` and the bounded project-level execution-record list. The deterministic local runner explicitly finalizes as `SIMULATED`.

Fidelity PR 6 makes the Modal torchvision classifier consume the accepted semantic config through the packaged capability resolver. It reports initialized and finalized realization observations with exact optimizer, scheduler, loss, sampler, transform, normalization, transfer-learning, and framework-version evidence. Pretrained-weight and dataset-normalization failures no longer fall back to different semantics. Shadow mode records mismatches while runner enforcement stops before the training loop; legacy unversioned jobs remain runnable only in shadow mode.

New plan payloads are stored in a `planned_experiments.v1` envelope. Legacy array payloads remain readable and are reported as `legacy_unversioned` without rewriting stored records.

After editing the canonical JSON, regenerate the embedded Go data and packaged Python data:

```sh
python3 scripts/generate_experiment_execution_capabilities.py
```

CI and local checks can detect drift without rewriting files:

```sh
python3 scripts/generate_experiment_execution_capabilities.py --check
```

The shared fixture file is consumed by Go and Python golden tests. The Go coverage test also reflects over `PlannedExperiment`, `Preprocessing`, and `AugmentationPolicyConfig`, so a newly added plan field must be classified in every profile before tests pass.
