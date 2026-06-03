# YOLO And Real-Time Export Plan

## Goal

Make Model Express evaluate exported models like always-on live inference systems, then add true YOLO object-detection support for datasets that include YOLO files.

This plan intentionally separates the work into multiple PRs so classifier export testing can improve immediately while YOLO training/export lands behind a clear detection contract.

## Current State

- The training loop is image-classification first.
- Bounding boxes can guide preprocessing, but the worker does not yet train YOLO detectors.
- Champion testing is closer to per-image demo inference than a continuous live-stream simulation.
- Champion ranking now uses loss-heavy deployment readiness for classifier runs, but detection metrics need a separate readiness profile.

## PR 1: Real-Time Export Contract

Owner: backend and worker.

Add a model-agnostic live inference contract to export manifests.

Required fields:

- `model_kind`: `classification` or `detection`
- `runtime`: `onnx`, later `tensorrt`
- preprocessing contract
- postprocessing contract
- input shape
- class names
- confidence threshold defaults
- latency budget
- export self-test status

Acceptance:

- Every champion artifact exposes enough metadata for a frontend or worker to run it continuously.
- Classification exports work with the contract immediately.
- Detection fields are reserved cleanly for YOLO.

## PR 2: Export Self-Test And Parity Check

Owner: worker/export.

After export, run the exported model on heldout/test images and compare it against the framework-native model.

Measure:

- prediction parity
- probability or logit drift
- preprocessing contract version
- p50 and p95 latency
- throughput/FPS estimate
- failed inference count
- low-confidence rate

Acceptance:

- A champion can be marked `export_verified`.
- Bad exports, preprocessing mismatches, and slow exports are visible before user testing.
- Export-readiness metrics are persisted for champion ranking.

## PR 3: Always-On Slideshow Inference UI

Owner: frontend.

Replace click-per-image testing with a live test harness.

Behavior:

- Load champion export once.
- Warm it up.
- Feed heldout images as timed slideshow frames.
- Run inference continuously.
- Show prediction, confidence, latency, FPS, and recent prediction history.
- Allow pause, speed control, failure inspection, and feedback.
- Render classification labels now.
- Design the renderer so detection boxes can be added later.

Acceptance:

- The user can press "Start live test" and watch continuous inference.
- Inference uses the exported preprocessing contract.
- The UI records enough runtime stats for live-readiness scoring.

## PR 4: Live Readiness Scoring

Owner: orchestrator/planner.

Extend champion ranking so export behavior matters.

Ranking inputs:

- validation/test loss
- heldout metrics
- export parity
- p95 latency
- preprocessing correctness
- confidence stability
- low-confidence/error rate
- user feedback

Acceptance:

- A slightly more accurate model loses if export parity or p95 latency is bad.
- Planner feedback explicitly recommends smaller models, lower resolution, better preprocessing, threshold changes, or fuller fine-tuning based on live failures.

## PR 5: YOLO Dataset Detection And Catalog Options

Owner: dataset/orchestrator.

Detect YOLO files during profiling and expose YOLO pretrained detector candidates in planner context.

Supported evidence:

- `data.yaml`, `data.yml`, `dataset.yaml`, or `dataset.yml`
- `labels/*.txt` YOLO label files
- normalized YOLO rows: `class x_center y_center width height`

Catalog candidates:

- `yolo11n.pt`
- `yolo11s.pt`
- `yolo11m.pt`
- `yolo11l.pt`
- `yolo11x.pt`

Acceptance:

- YOLO datasets are profiled as `task_type: object_detection`.
- Profile includes `yolo_available`, `object_detection_available`, `bbox_count`, `bbox_per_class`, and `yolo_summary`.
- Planner model catalog includes YOLO11 detector options only when YOLO evidence is present.
- YOLO detector entries are marked as not schedulable until the detection worker lands.

## PR 6: YOLO Training Worker

Owner: detection/worker.

Add Ultralytics-based detector training.

Training requirements:

- Load YOLO dataset config.
- Train from COCO-pretrained detector weights.
- Preserve official train/val/test splits.
- Support nano/small first for live-stream use.
- Add medium/large as quality challengers.

Metrics:

- `mAP50_95`
- `mAP50`
- precision
- recall
- box loss
- cls loss
- DFL loss
- p50/p95 latency
- FPS

Acceptance:

- YOLO detector jobs train and persist detection metrics.
- Detection jobs cannot be confused with classifier jobs.
- Detector failures report actionable errors.

## PR 7: YOLO Export And Live Detection UI

Owner: worker/export and frontend.

Extend export and live test UI for detector outputs.

Behavior:

- Export ONNX first.
- Add TensorRT for NVIDIA deployment later.
- Decode detection outputs.
- Apply NMS.
- Draw boxes over slideshow frames.
- Show class/confidence per box.
- Support confidence threshold adjustment.
- Track per-frame detection count, misses, false positives, and postprocess latency.

Acceptance:

- A YOLO champion can be tested like a live camera stream using heldout slideshow frames.
- UI supports both classifier predictions and detection overlays from the same live-test shell.

## PR 8: Detection Champion Ranking

Owner: orchestrator/planner.

Add detection-specific deployment readiness.

Ranking should prioritize:

- validation/test detection losses
- `mAP50_95`
- `mAP50`
- recall for live safety
- precision when false positives are costly
- export parity
- p95 latency
- FPS
- confidence stability
- user feedback

Acceptance:

- A detector with better mAP but bad box/cls/DFL loss or unreliable export loses.
- Planner chooses follow-up experiments based on detection readiness, not classifier metrics.

## PR 9: Reliability And E2E Hardening

Owner: QA/integration.

Tests:

- YOLO dataset parser/profile tests
- model catalog gating tests
- export manifest contract tests
- ONNX parity tests
- live slideshow mocked-inference tests
- detection metric persistence tests
- champion ranking tests where slow/broken exports lose
- end-to-end smoke test for continuous inference

Acceptance:

- "Looks good in metrics but fails live" is hard to regress.
- Classifier and detector projects stay cleanly separated.

## Notes

Ultralytics documents YOLO11 detection weights as `yolo11n.pt`, `yolo11s.pt`, `yolo11m.pt`, `yolo11l.pt`, and `yolo11x.pt`, with train, validation, inference, and export support. See: https://docs.ultralytics.com/models/yolo11/
