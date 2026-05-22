# Model Express End-To-End Checklist

Use this checklist at the end of an agentic-upgrade pass. It verifies the roadmap while keeping the control boundary intact:

```text
LLMs propose structured JSON -> backend validates -> backend stores/schedules -> workers execute
```

## Roadmap Acceptance

- [x] PR 1: `datasets.profile` JSON is explicitly the active profile source of truth.
- [x] PR 1: `dataset_profiles` is documented as dormant; no code path reads it as canonical.
- [x] PR 1: preprocessing fields are aligned across backend validation, planner prompt/schema, worker config, and Mission Control display.
- [x] PR 2: worker tests cover transform construction, sampler/loss selection, profile artifact detection, and dataset normalization metadata.
- [x] PR 2: worker helper utilities parse split files, Pascal VOC XML, and annotation JSON without changing training behavior.
- [x] PR 2: dataset-computed normalization is bounded and applied only when requested.
- [ ] PR 2: explicit split-file training and bbox crop/full-image ablations are not wired into training yet.
- [x] PR 3: planner prompt/examples include `resolution_strategy`, `preprocessing`, `augmentation_policy`, `sampling_strategy`, class balancing, and JSON-only output.
- [x] PR 4: candidate ranking treats preprocessing/sampling/resolution changes as meaningful mechanisms.
- [x] PR 4: rejected options, scorecards, and candidate `score_components` remain compatible with new fields.
- [x] PR 5: Mission Control has section navigation, typed preprocessing display, candidate scores, export/demo controls, selected demo image next/random actions, and durable prediction-history rendering.
- [x] PR 6: low-risk reliability landed: positive epoch validation, export/demo execution events, and idempotent export request behavior.
- [ ] PR 6: job leases/requeue, SSE, async automation, and durable idempotency keys remain deferred.
- [x] PR 7: champion export records/API exist and are idempotent per project/champion/format.
- [x] PR 7: worker has export manifest/checkpoint helpers with guarded TorchScript/ONNX paths.
- [x] PR 7: backend persists demo prediction audit/history records and returns `RUNTIME_UNAVAILABLE` instead of fake predictions when runtime is absent.
- [x] PR 7: Mission Control shows export status, use-case fit, demo images, prediction status/error, true label, top-k, latency, and correctness fields when present.
- [ ] PR 7: worker-backed inference is not yet scheduled/executed through backend jobs.
- [x] PR 8: backend exposes capped visual exemplars from `datasets.profile` and passes evidence-only exemplar context to the planner.
- [x] PR 8: planner prompt context carries exemplar caps/audit and still requires JSON-only output.
- [x] PR 8: worker has deterministic class-balanced visual exemplar generation with downscale/compression and byte/image caps.
- [ ] PR 8: exemplar upload/persistence and multimodal image attachment remain deferred.
- [x] PR 9: `docs/me_ground_truth.md` exists and reflects the current architecture and safety boundary.

## Verification Commands

- [x] `go test ./...` from `services/orchestrator`
- [x] `python -m unittest discover -s tests -v` from `services/worker`
- [x] `python -m py_compile worker/datasets/profiler.py worker/datasets/exemplars.py worker/datasets/annotations.py worker/training/modal_app.py worker/training/local.py worker/exporting/metadata.py worker/exporting/artifacts.py worker/exporting/inference.py` from `services/worker`
- [x] `npm run build` from `apps/mission-control`

## Manual Demo Checklist

- [ ] Create or select a project with a profiled image-folder dataset.
- [ ] Confirm Mission Control shows dataset intelligence, split/artifact hints, and preprocessing recommendations.
- [ ] Execute a validated experiment plan and confirm workers report metrics, summaries, and evaluations.
- [ ] Select or confirm a champion.
- [ ] Open Export/Demo and request an ONNX export.
- [ ] Confirm duplicate ONNX requests update the existing export record rather than adding noisy duplicates.
- [ ] Confirm demo images load from `datasets.profile.demo_images` or `visual_exemplars`.
- [ ] Run a demo prediction and confirm the backend records a `RUNTIME_UNAVAILABLE` audit row until worker-backed inference is wired.
- [ ] Confirm planner decisions remain JSON-only and backend validation still rejects unsupported execution fields.

## Known Remaining Work

- Wire champion export/inference worker jobs end to end through backend validation and scheduling.
- Promote `dataset_profiles` only with a full write/read/backfill/test plan, or keep it permanently documented as dormant.
- Wire explicit split-file training, bbox crop/full-image ablations, and advanced augmentation object policies.
- Persist/upload generated visual exemplars and record planner invocation audit fields durably.
- Add job leases, stale job recovery, SSE/project event stream, async agent tasks, and stronger DB-backed idempotency.
