# Mission Control Comprehension UX Backlog

Created: 2026-06-18

Purpose: capture the product/UX changes needed to make Mission Control understandable to a new user, especially a user with computer science knowledge but limited machine learning experience. This is a discussion backlog, not an implementation record.

## Problem Statement

Mission Control currently proves that the system is doing work, but it does not always make the work obvious. The UI exposes many correct technical details at once, which makes the product harder to understand in a demo or first run.

The default experience should answer these questions within a few seconds:

- What is Model Express doing?
- Why is it doing that?
- What changed because of it?
- Is the run still healthy?
- What should the user do next, if anything?

## Current Pain Points

- [ ] Too much text appears at equal priority.
- [ ] Long AI reasoning text is cut off inside small cards.
- [ ] ML terms such as `champion`, `Macro-F1`, `experiment`, `augmentation`, and `ONNX` appear before plain-English meaning.
- [ ] The Mission screen looks like a pipeline/status monitor, not a story of what Model Express is accomplishing.
- [ ] The Results screen needs stronger candidate comparison and better context for scores.
- [ ] Dataset/class examples are not visible enough, so the product feels abstract instead of grounded in the user's images.
- [ ] Developer-level details leak into the default user workflow.
- [ ] `Blocked` states are confusing when the backend/program is still running.
- [ ] The UI relies too heavily on prose where charts, tables, and visual comparisons would explain the state faster.

## Blocked State Issue

Observation from the current run: the app can show `BLOCKED` in the top status, Mission tab, Initial Experiments stage, Champion Selection stage, and AI thinking card while the engine is still ready and the program is still doing work or waiting on capacity.

Why this is confusing:

- `Blocked` reads like the whole run is broken.
- The UI simultaneously says `Engine READY`, `Experiment started`, `12/14 experiments complete`, and `BLOCKED`.
- A user cannot tell whether they should wait, fix something, open diagnostics, add capacity, or stop the run.
- The red styling spreads across multiple areas, making a partial capacity issue feel like total failure.
- `Needs your input` is misleading during autonomous runs because the user usually does not provide input while the run is in progress.

Desired behavior:

- Reserve `Blocked` for a true stop where the run cannot continue without intervention.
- Use softer, more precise state labels for recoverable or partial waits.
- Separate system health from workflow progress.
- Explain whether the user needs to act.
- Only say the user is needed when there is a concrete user-resolvable action.

Proposed status taxonomy:

- `Running`: active work is happening.
- `Waiting for training capacity`: jobs are queued or delayed, but the run is not broken.
- `Needs attention`: user action may help, but the project state is still recoverable.
- `Paused`: work was intentionally stopped by user/system policy.
- `Failed`: a terminal error occurred.
- `Blocked`: no forward progress is possible until a specific issue is fixed.

Potential UI copy:

- Instead of `BLOCKED`: `Waiting for capacity`
- Instead of `Needs your input`: `Waiting on training capacity`
- If user action is truly required: `Action required`
- Instead of `Pause new work until the blocker is resolved`: `Model Express is holding new trials until capacity is available. Completed results remain safe.`

Implementation notes:

- [x] Audit where `blocked` is derived in Mission Control state mapping.
- [x] Split global project status from individual stage status.
- [x] Do not paint the whole workflow red for a stage-specific blocker.
- [x] Add a clear action row: `What happened`, `Impact`, `What to do next`.
- [ ] Use `Open diagnostics` or `Review capacity` as the visible next action when useful.
- [x] Replace generic `Needs your input` language with either autonomous system status or a specific user action.
- [x] Add an explicit boolean or derived concept in the UI model: `requiresUserAction`.
- [x] If `requiresUserAction` is false, avoid labels like `needs your input`, `action required`, or `blocked`.

Backend impact estimate:

- Likely small or none for first pass if existing event/status payloads already distinguish capacity failures.
- Possible small backend improvement if the frontend cannot tell terminal failure from recoverable capacity wait.

## Plain-English Product Framing

Goal: make the first screen explain the product without requiring ML background.

Suggested default explanation:

> Model Express is testing multiple image models, comparing quality and deployment tradeoffs, then packaging the best one for use outside the app.

Suggested current-state explanation:

> Right now: 12 of 14 model trials are complete. The best model so far scores 90.9/100 across classes. Model Express is waiting for the remaining trials before choosing the final export.

Changes:

- [x] Add a top-level plain-English mission summary.
- [x] Add a `Right now` sentence that explains the current state.
- [x] Add a `Next decision` sentence.
- [x] Keep technical metrics visible, but secondary.

Backend impact estimate:

- None for first pass. Existing project/job/champion/run data can drive this in the frontend.
- Optional future backend endpoint: `GET /projects/:id/mission-brief`.

## Terminology Simplification

Use plain-English labels first, technical names second.

- [x] `Champion` -> `Best model so far`
- [x] `Champion Selection` -> `Choosing the best model`
- [x] `Macro-F1` -> `Balanced accuracy score`
- [x] `ONNX` -> `Portable model package`
- [x] `Experiment` -> `Model trial`
- [ ] `Augmentation` -> `Training image variations`
- [ ] `Regularization` -> `Anti-overfitting controls`
- [x] `Export` -> `Package for deployment`

Display pattern:

```text
Balanced accuracy score
Macro-F1: 0.909
```

Backend impact estimate:

- None.

## Text Density And Truncation

Problem: long paragraphs inside small cards make the UI feel broken and hide important reasoning.

Changes:

- [x] Replace paragraph cards with one-line summaries.
- [x] Add expand/collapse or drawer for full AI reasoning.
- [ ] Standardize long reasoning into four sections:
  - Observation
  - Decision
  - Why it matters
  - Technical details
- [x] Avoid clipped text for user-facing explanations.
- [x] Keep raw/verbose reasoning in Developer View.

Backend impact estimate:

- None unless we decide to persist separate short/long explanation fields.

## Charts And Tables First

Principle: the default user view should explain model progress through visual evidence first, then use short text only to interpret the chart or table. Text should annotate the evidence, not replace it.

Why this matters:

- Tables make candidate tradeoffs obvious.
- Charts make progress and plateau states visible without long explanations.
- Simple visual comparisons are easier to follow in a demo video.
- Users can understand image classification outcomes faster from images, bars, rankings, and confusion summaries than from AI reasoning paragraphs.

Changes:

- [x] Replace large explanation blocks with compact charts/tables wherever the data is structured.
- [x] Keep each visual anchored by a one-sentence takeaway.
- [x] Put detailed reasoning behind expanders or Developer View.
- [x] Use consistent visual language across Mission, Results, Activity, and Export.

Recommended visuals:

- [x] Mission progress chart: completed, running, waiting, failed model trials.
- [x] Model leaderboard table: model, balanced score, accuracy, speed, size, cost, status.
- [ ] Score-over-time chart: best score by completed trial or round.
- [ ] Training chart: validation score/loss over epochs for selected run.
- [ ] Class distribution chart: images per class.
- [x] Confusion matrix preview: where the best model gets confused.
- [x] Per-class performance table: class, precision, recall, F1, example misses.
- [ ] Candidate tradeoff scatter: quality versus latency or quality versus cost.
- [ ] Export readiness checklist: selected model, package ready, demo tested, downloadable.
- [x] Demo result bars: top predicted classes and confidence.

Screen-specific direction:

- Mission:
  - [x] Replace some stage-card density with a progress distribution bar.
  - [x] Show a small model-trial status table instead of repeated text cards.
  - [ ] Add a "what changed recently" table with time, event, impact.

- Results:
  - [x] Make the leaderboard the central object.
  - [x] Show why the leader is winning with numeric columns instead of paragraph explanation.
  - [ ] Add sparkline or trend for top candidates if data is available.

- Activity:
  - [x] Convert event cards into a timeline table by default.
  - [x] Use expandable rows for technical details.
  - [x] Show decisions as `decision`, `evidence`, `result`, `impact` columns.

- Export / Demo:
  - [x] Put image preview and top-class confidence bars first.
  - [ ] Show export readiness as a checklist/table, not separate prose panels.
  - [x] Move manifest/package internals below the demo.

Backend impact estimate:

- Mostly none for existing run, job, champion, and evaluation metrics.
- Small if the frontend needs cleaner per-class summaries or compact confusion-matrix payloads.
- Moderate only if missing data must be newly computed, persisted, or served as thumbnails/examples.

## Mission Screen Narrative

The Mission tab should behave like a live briefing, not a raw pipeline diagram.

Suggested first-screen beats:

- [ ] `Dataset understood`: show image count, class count, and readiness.
- [x] `Model trials running`: show completed/running/waiting trial counts.
- [x] `Best model so far`: show leading model and plain-English score.
- [x] `Next decision`: explain what Model Express will decide next.

Changes:

- [x] Rework timeline so it supports the story rather than dominating it.
- [x] Make the most important current state visually obvious.
- [x] Add an `Impact` line for state changes.
- [x] Show whether user action is required.

Backend impact estimate:

- None for first pass.
- Small if we want backend-provided workflow state reasons.

## Results Screen Comparison

Problem: a single score like `0.909 Macro-F1` is not enough context for new users.

Changes:

- [x] Show candidates as a ranked model comparison.
- [x] Compare quality, speed, size, cost, and deployment fit.
- [x] Explain why the leader is winning.
- [x] Show whether another candidate is close enough to wait for.
- [x] Add beginner-friendly metric explanation.

Example:

```text
90.9 / 100 balanced score
Higher is better. This measures performance across all classes, not just the easy ones.
```

Backend impact estimate:

- None for first pass. Existing training summaries/evaluations and champion records should be enough.
- Small if a clean backend candidate-comparison endpoint is desired later.

## Dataset And Class Examples

Problem: without actual images, the product feels abstract. For demos, users need to see that Model Express is working on their dataset.

Changes:

- [ ] Show class names and image counts early.
- [ ] Show representative thumbnails per class.
- [ ] Show any dataset imbalance or quality warnings in plain English.
- [x] Later, show held-out prediction examples after export/demo readiness.

Backend impact estimate:

- Unknown until current visual exemplar/demo image payloads are inspected.
- None or small if existing endpoints expose reliable `thumbnail_uri` or `preview_uri` values.
- Moderate if backend must generate, store, authorize, and serve thumbnails from raw dataset files.

Safe thumbnail requirements:

- [ ] UI receives browser-renderable thumbnail URLs.
- [ ] Backend avoids exposing arbitrary local filesystem paths.
- [ ] Backend prevents path traversal and unauthorized file reads.
- [ ] Thumbnails are small and stable across refreshes.
- [ ] Content types are correct.

## Export And Demo Comprehension

Problem: the Export page currently presents packaging, ONNX, validation, manifest, demo image, and handoff information as separate technical blocks. For an image-classification product, the demo should be the easiest part to understand: pick an image, run the best model, see the predicted class, confidence, and whether it matched the known label.

Current confusing points:

- [x] `Waiting for champion` explains the backend dependency, but not what export/demo will eventually let the user do.
- [x] `ONNX`, `technical manifest`, `preprocessing contract`, and `model card` appear before the user sees a simple image-classification demo.
- [x] The page says `Demo image` and `Held-out image`, but the empty state still reads like an internal artifact workflow.
- [x] `Prepare ONNX` sounds like a developer task, not "make this model usable."
- [x] Demo readiness is spread across `Format`, `Validation`, `Demo`, export records, and demo image state.

Desired user story:

> Model Express has picked the best model. Now test it on images. If the prediction looks right, download or package the model for use elsewhere.

Export page should answer:

- [x] What model won?
- [x] Can I test it on an image?
- [x] What did it predict?
- [x] How confident was it?
- [x] Was it correct on a held-out image?
- [x] Is the model ready to download/use?
- [x] What technical package was created?

Recommended layout direction:

- [x] Rename the page from `Export` to something like `Test & Export` or `Try Model`.
- [x] Put the image demo first once a champion exists.
- [x] Show a large image preview with a clear result panel:
  - `Prediction`
  - `Confidence`
  - `Known label`
  - `Correct / Missed / Unknown`
  - `Latency`
- [x] Show top-3 class probabilities as simple bars.
- [x] Show held-out image examples as thumbnails.
- [ ] Group held-out image examples by class when reliable class grouping is available.
- [x] Move ONNX/package/manifest details below the demo or behind `Technical package`.
- [x] Replace `Prepare ONNX` with `Prepare portable model` and show `ONNX` as the technical format beneath it.
- [x] Replace `Open technical manifest` with a quieter developer/advanced action.

Potential waiting-state copy:

```text
Waiting for the best model
Model Express is still comparing candidates. After it chooses the best model, this page will let you test images and prepare a portable model package.
```

Potential ready-state copy:

```text
Try the best model
Choose a held-out image or upload your own. Model Express will show the predicted class, confidence, and whether the result matches the known label.
```

Backend impact estimate:

- Mostly frontend if existing demo image and prediction payloads are reliable.
- Small backend work if the UI needs a single demo-readiness response instead of deriving readiness from exports, predictions, and champion fields.
- Moderate backend work if class-grouped held-out thumbnails or safe custom-image upload endpoints are missing.

## Gradio-Style Demo Feasibility

Question: can Model Express use something like Gradio for the demo experience?

Initial feasibility finding:

- [ ] Technically yes. Gradio supports image inputs and label/classification outputs, and its high-level `Interface` can wrap a Python prediction function with an image input and label output.
- [ ] Gradio can run a local web server through `launch()`.
- [ ] Gradio can also be mounted into a FastAPI app with `mount_gradio_app`, but the current orchestrator is Go/Gin, not FastAPI.
- [ ] The current worker environment does not include Gradio today, so this would add a new Python dependency and runtime surface.

Why Gradio could help:

- [ ] Very fast path to a recognizable ML demo: image upload on the left, prediction/confidence on the right.
- [ ] Built-in image upload/webcam/clipboard support could make demo interactions feel familiar.
- [ ] Built-in label confidence display could simplify the first demo prototype.
- [ ] Could be useful as an optional standalone "Open demo app" experience.

Why Gradio may be the wrong default inside Mission Control:

- [ ] It would add a second UI framework beside the existing Electron/React app.
- [ ] It would require launching and supervising another local server/process.
- [ ] Styling and navigation may feel disconnected from Mission Control.
- [ ] Sharing/public-link features need careful disabling or authentication because local model demos can expose data.
- [ ] Gradio file serving requires careful `allowed_paths` / `blocked_paths` handling to avoid exposing local files.
- [ ] It could duplicate existing browser ONNX and backend demo prediction flows instead of making them clearer.

Recommended position:

- [ ] Do not replace the Mission Control demo with Gradio by default.
- [x] First redesign the native React Export/Demo page around a Gradio-like interaction model: image input, prediction label, confidence bars, held-out examples, simple correctness.
- [ ] Consider an optional `Launch standalone demo` button later if we want a shareable local demo mode.
- [ ] If added, make it local-only by default, authenticated if exposed, and explicitly separated from the main app.

Possible implementation models:

1. Native React demo, recommended first pass.
   - Use existing ONNX/browser and backend prediction APIs.
   - Best visual consistency.
   - Least new infrastructure.

2. Optional local Gradio sidecar.
   - Electron starts a Python Gradio process on `127.0.0.1`.
   - UI embeds it in an iframe or opens it in a browser window.
   - Good for demo polish, but adds dependency, process management, auth, and file-path concerns.

3. Backend-mounted Gradio app.
   - Natural if the backend were FastAPI.
   - Less natural with the current Go/Gin orchestrator.
   - Would likely require a Python sidecar service anyway.

Backend impact estimate:

- Native React demo: none to small.
- Optional Gradio sidecar: moderate, because it needs dependency packaging, process lifecycle, port management, model/artifact loading, safe file access, and demo API wiring.
- Backend-mounted Gradio: moderate to high because it conflicts with the current Go orchestrator shape unless introduced as a separate Python service.

## Demo Inference Reliability Bug

Observed error:

```text
Local UI inference needs original image bytes or a local image URI. Falling back to backend demo inference.
```

Why this matters:

- This appears during the core demo moment, so it undermines trust in the entire product.
- The user does not care whether inference is local browser ONNX or backend worker fallback.
- If fallback is expected, it should feel seamless, not like the demo failed.
- If fallback also cannot run, the UI should explain the real missing requirement, not expose an implementation detail.

Likely current behavior:

- The UI tries browser/local ONNX inference first when a browser-safe export exists.
- Local inference requires original image bytes or a browser-loadable local/data URI.
- Held-out images may only expose `s3://`, worker-visible paths, thumbnails, or preview URLs.
- The local path throws `LOCAL_IMAGE_UNAVAILABLE`.
- The UI then attempts backend demo inference, but the user still sees the local fallback error.

Acceptance criteria:

- [x] A held-out demo image should produce a prediction without exposing local/browser/backend implementation details.
- [x] If browser inference cannot use the image safely, the UI should silently route to backend demo inference unless backend inference is unavailable.
- [ ] If backend inference is unavailable, show one clear action-oriented message such as `Demo worker unavailable` or `Original image unavailable for demo`.
- [x] Do not show fallback as an error when fallback is normal/expected.
- [x] The selected demo image should have a stable original image source that either the browser can load safely or the backend worker can read.
- [x] Held-out demo image ordering keeps known-correct examples visible before known training-time mistakes, while still keeping hard failures inspectable.
- [x] Prediction output should show image, predicted class, confidence bars, known label, and correctness.

Likely fix areas:

- [x] Frontend: distinguish local fallback notices from actual demo failures.
- [x] Frontend: avoid setting `localInferenceStatus` to `error` for expected fallback cases.
- [x] Frontend: only surface `localInferenceError` inside advanced/diagnostic detail unless prediction ultimately fails.
- [x] Backend/worker: confirm `POST /projects/:id/champion/demo-predictions` receives an original/source image URI for stored held-out demo inference.
- [x] Backend/worker: ensure held-out/demo image records treat original image URI and preview thumbnail URI as separate roles.
- [x] Tests: add coverage for held-out image with non-browser-loadable original URI falling back to backend prediction, and for thumbnail/preview-only stored records being rejected as display-only.

Backend impact estimate:

- Small if backend demo prediction already works and the UI only needs to hide expected fallback notices.
- Moderate if demo image records do not reliably include worker-readable original image URIs.
- Moderate if a safe image upload/bytes endpoint is needed for custom images.

## User View Versus Developer View

Problem: the default flow still exposes too much internal machinery.

Changes:

- [x] Default user view shows: what is happening, why, what changed, next action.
- [x] Developer View keeps: job IDs, raw events, payloads, telemetry, agent invocations, exact artifacts, and diagnostics.
- [x] Add clear affordances to inspect technical details without making them the default.

Backend impact estimate:

- None.

## Optional Backend Mission Brief Endpoint

Only add this if frontend aggregation becomes too messy or inconsistent.

Candidate endpoint:

```http
GET /projects/:id/mission-brief
```

Candidate response:

```json
{
  "headline": "Model Express is testing image models and preparing the best one for export.",
  "current_step": "Model trials running",
  "plain_status": "12 of 14 trials are complete. 2 are still running.",
  "best_model": {
    "name": "mobilenet_v3_small",
    "score_label": "Balanced accuracy score",
    "score_value": 0.909,
    "technical_metric": "macro_f1"
  },
  "next_decision": "Wait for remaining trials, then choose the export model.",
  "requires_user_action": false,
  "risk_summaries": [
    "The best model may still miss low-resolution class cues."
  ]
}
```

Backend impact estimate:

- Small if computed from existing store reads.
- Larger if summaries are LLM-generated, persisted, versioned, or audited.

## Suggested Implementation Phases

### Phase 1: UI-only comprehension pass

- [x] Fix blocked/waiting status language.
- [x] Add plain-English mission summary.
- [x] Rename technical labels in user-facing views.
- [x] Replace clipped long text with summaries and expandable details.
- [x] Replace prose-heavy explanations with charts/tables where data is structured.
- [x] Improve Results comparison layout.

Expected backend work: none.

### Phase 2: Dataset grounding

- [x] Audit visual exemplar/demo image endpoint payloads.
- [x] Use existing preview/thumbnail URLs if available.
- [ ] Add backend thumbnail endpoint only if needed.
- [x] Redesign Export as a `Try Model` flow before exposing package internals.
- [x] Fix demo inference fallback so local/browser limitations do not surface as user-facing demo failures.

Expected backend work: none to moderate, depending on current image payloads.

### Phase 3: Durable mission brief contract

- [ ] Add optional `/projects/:id/mission-brief`.
- [ ] Move fragile frontend inference into backend-owned response fields only where useful.
- [ ] Add contract tests for blocked/waiting/running/champion/export states.
- [ ] Decide whether a standalone Gradio-style demo is worth the process/dependency cost after the native demo redesign.

Expected backend work: small to moderate.

## Open Questions

- [x] Which statuses currently map to `blocked` in the Mission Control frontend?
- [ ] Does the backend distinguish capacity wait, worker failure, user intervention, and terminal failure cleanly?
- [ ] Do dataset visual exemplars already include browser-safe thumbnails?
- [x] Should `Macro-F1` remain visible in the default user view or only in expanded metric details?
- [ ] Should the Mission tab show all stages at once, or only the current stage plus expandable history?
- [ ] Should the video demo prioritize dataset images, model comparison, or export/demo inference?
- [x] Should the Export tab be renamed to `Try Model`, `Test & Export`, or another more user-facing label?
- [ ] Do we need custom image upload to work before export readiness, or only after a portable model exists?
- [ ] Is an optional Gradio sidecar worth adding after the native React demo is simplified?
- [ ] Are current held-out demo images guaranteed to include worker-readable original image URIs?
