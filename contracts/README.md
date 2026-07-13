# Execution Contracts

`experiment_execution_capabilities.v1.json` is the canonical, versioned source for experiment execution capabilities. It records the current semantics of each `PlannedExperiment` field for every supported task/runner pair. The classifications mean:

- `executed`: the runner uses the field directly.
- `conditional`: the runner uses the field only when the recorded prerequisite is active.
- `metadata_only`: the field belongs to planner or control-plane provenance, not training semantics.
- `unsupported`: the runner does not execute the requested semantic.

Fidelity PR 2 uses the embedded Go representation to resolve a canonical `execution_spec_v1` when an experiment job is queued. The spec records the requested config, accepted semantic config, capability version, task, runner, requested-config hash, and accepted-spec hash. Legacy flat job keys remain in place for current workers.

Fidelity PR 5 snapshots that accepted spec independently from mutable job configuration and creates one execution record per assigned attempt. Workers may append authenticated `INITIALIZED` and `FINALIZED` observations through `POST /jobs/:id/execution-observations`; the server bounds and redacts evidence, derives the realized hash and fidelity verdict, and rejects callbacks from stale attempts. Records are available through `GET /jobs/:id/execution-record` and the bounded project-level execution-record list. The deterministic local runner explicitly finalizes as `SIMULATED`.

Fidelity PR 11 makes execution validation fail closed by default. Unsupported proposals are rejected before worker requirements or GPU jobs are created, and successful versioned real-training completion requires the same finalized, evidence-eligible verdict used by learning and champion selection. `MODEL_EXPRESS_EXECUTION_VALIDATION_MODE=shadow` is the one-switch compatibility rollback and never rewrites stored receipts. `GET /projects/:id/telemetry-summary` exposes compact execution-fidelity rollout counters for unsupported proposals, pending/not-realized attempts, verdicts, adjustments, legacy reads, and successful jobs missing final realization.

Fidelity PR 6 makes the Modal torchvision classifier consume the accepted semantic config through the packaged capability resolver. It reports initialized and finalized realization observations with exact optimizer, scheduler, loss, sampler, transform, normalization, transfer-learning, and framework-version evidence. Pretrained-weight and dataset-normalization failures no longer fall back to different semantics. Shadow mode records mismatches while runner enforcement stops before the training loop; legacy unversioned jobs remain runnable only in shadow mode.

Fidelity PR 7 makes the Modal Ultralytics detector consume the accepted detection spec through a pure kwargs translator. The Modal image pins Ultralytics 8.4.66, and the translator explicitly supplies the pinned native optimization, augmentation, determinism, and letterboxing semantics in addition to the five executable detection fields. The runner captures allowlisted trainer arguments at `on_pretrain_routine_end`, posts `INITIALIZED` and `FINALIZED` observations, and treats any framework-normalized divergence as a mismatch. Unsupported classifier augmentation, balancing, sampling, optimizer, scheduler, regularization, and fine-tuning requests remain typed validation findings rather than silently entering `YOLO.train`. Enforcement rejects mismatches before the epoch loop; shadow mode records them for rollout analysis.

Fidelity PR 8 assigns each identity to one lifecycle purpose. `requested_config_hash` remains request lineage, `accepted_spec_hash` is now the authoritative pre-scheduling duplicate identity, and `realized_effective_hash` is absent until a worker realization exists. Attempt IDs and Modal resource signatures remain operational identities and never participate in semantic hashes. Accepted duplicates are skipped before job creation in both validation modes. Approved batch recovery records the server-derived `batch_size_reduced_by_resource_recovery` adjustment reason while retaining the original accepted-spec lineage, so its distinct realized hash is not treated as a new planner mechanism. Historical unversioned jobs remain `UNVERIFIED` and do not receive fabricated accepted or realized hashes.

Fidelity PR 9 derives one evidence-eligibility decision from the latest attempt and uses it for planner outcomes, strategy scorecards, vector memory, project trajectory, and automatic champion selection. Finalized `MATCHED` runs are eligible. `APPROVED_ADJUSTMENT` is eligible only for the allowlisted `batch_size_reduced_by_resource_recovery` reason. `MISMATCH`, `SIMULATED`, pending, initialized-only, and unknown-versioned states remain visible but cannot teach the planner or become automatic champions. Historical `UNVERIFIED` runs are explicitly governed by `MODEL_EXPRESS_LEGACY_EXECUTION_EVIDENCE_POLICY`; compatibility defaults to `allow`, while `visible_only` removes legacy runs from both eligibility paths without rewriting them. Outcome and scorecard records retain requested mechanism, accepted identity, realized mechanism identity, realized hash, verdict, and adjustment reasons separately.

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
