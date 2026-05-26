# Visual Dataset Analysis Agent Planning Document

## 1. Summary

Add a separate **Visual Dataset Analysis Agent** that runs as its own expensive, rare, multimodal LLM stream after dataset profiling. It analyzes a bounded image sample pack, returns structured visual evidence and preprocessing hypotheses, and persists that evidence as durable dataset-level analysis. The existing Experiment Planner will only receive a compressed `PlannerVisualExemplarContext`, never raw images or the Visual Agent prompt history.

Core boundary:

```text
Profiler and backend choose samples/caps -> Visual Agent observes images -> backend validates output
-> backend persists DatasetVisualAnalysis -> Planner receives compressed evidence only
-> backend still validates any future experiment config before scheduling
```

The Visual Agent never creates jobs, plans, dataset mutations, labels, or executable preprocessing config by authority. It only produces evidence and hypotheses.

## Implementation Status

Status from the current implementation pass:

- **PR 1: Completed.** Durable `dataset_visual_analyses` storage, store interfaces, memory/Postgres persistence, migrations, list/latest/result/manual-run APIs, execution events, and invocation persistence are implemented.
- **PR 2: Completed.** Worker-side `analyze_dataset_visuals` job, deterministic bounded sample-pack generation, stable `image_id` manifest entries, image caps, and no-path manifest behavior are implemented and tested.
- **PR 3: Completed.** The Visual Dataset Analysis Agent now uses a separate worker LLM path with `MODEL_EXPRESS_VISUAL_LLM_*` config, evidence-only prompts, schema parsing, fake-client tests, and malformed-output rejection.
- **PR 4: Completed.** Training image preprocessing moved behind a tested registry for resize, crop, normalization, augmentation, and bbox-aware modes while preserving the existing Modal training path.
- **PR 5: Completed.** Backend-owned manual and initial-profile queueing, deficiency-trigger reanalysis, cooldown windows, per-profile-version max-run policy, output validation, accepted/rejected persistence, no-direct-scheduling validation, bbox-evidence checks, and manifest/telemetry leakage guards are implemented.
- **PR 6: Completed.** Experiment Planner input now fetches only the latest accepted analysis and converts it into capped `PlannerVisualExemplarContext`; raw images, raw Visual Agent output, and Visual Agent prompt history are excluded from planner prompts.
- **PR 7: Completed.** Go and Python unit/integration-style coverage proves the safe happy path, malformed/rejected output, unsupported/no-authority behavior, path leakage rejection, sampling caps, deficiency-trigger queueing, cooldown/max-run enforcement, and planner prompt exclusions without real LLM/Modal spend.
- **PR 8: Completed.** Mission Control displays latest/listed visual evidence, trigger/status, coverage, traits, classes to watch, hypotheses, cautions, limitations, validation errors, backend rerun policy, cooldown/max-run status, active-job blocking, and manual rerun controls that ensure a worker is available after a backend-approved rerun is queued.

Operational validation still to run outside this planning implementation:

- First real loop with a configured visual LLM provider and actual image dataset.
- Real Modal execution of `analyze_dataset_visuals`.
- Live Postgres migration smoke in the target deployment database.

Boundary status:

- Visual analysis runs in its own worker LLM stream and configuration namespace.
- Backend validation remains the source of truth for accepted evidence and scheduling authority.
- Durable accepted/rejected visual evidence is persisted separately from `datasets.profile`.
- Planner consumes compressed accepted evidence only.
- Visual Agent outputs cannot directly create jobs, plans, dataset mutations, labels, or executable authority.
- Workers execute only backend-created `analyze_dataset_visuals` jobs and backend-validated training plans.
- When visual analysis is enabled/configured, the initial experiment plan waits for the initial visual-analysis result; if visual analysis is disabled or the initial visual job fails, backend falls back to normal profile-only planning.

## 2. Architecture And Data Flow

```text
profile_dataset job
  -> datasets.profile updated
  -> backend decides whether visual analysis is allowed
  -> analyze_dataset_visuals job queued

Modal/Python worker
  -> downloads/extracts dataset
  -> builds deterministic bounded visual sample pack
  -> prepares resized/high-detail image inputs
  -> calls separate visual LLM using MODEL_EXPRESS_VISUAL_LLM_* settings
  -> posts raw JSON + sample manifest to backend

Backend
  -> validates schema, caps, enums, sample references, support status
  -> persists dataset_visual_analyses row
  -> records visual_dataset_analysis agent_invocation
  -> emits execution event
  -> exposes latest accepted compressed context to planner

Experiment Planner
  -> loads latest accepted analysis
  -> receives compact PlannerVisualExemplarContext inside planner_context_snapshot.visual_evidence
  -> proposes experiments normally
  -> backend validates mechanisms/preprocessing before scheduling
```

Default trigger behavior:

- `initial_profile`: run once after profile if visual analysis is enabled and no accepted analysis exists for the dataset/profile version.
- `deficiency_reanalysis`: allow rare rerun after severe failures, such as repeated no-improvement rounds, persistent top confusions, low macro-F1, worst-class recall failure, or contradicted visual hypotheses.
- `manual`: user-triggered rerun from API/UI.
- Enforce cooldowns and max reruns per dataset/profile version.

Recommended ownership choice: the **visual LLM call runs in the Python/Modal job**, not inside the existing Go planner path. This keeps image bytes near the worker, avoids bloating Go planner requests, and enables separate visual LLM API keys, model, cost caps, and timeouts.

## 3. Data Model And Interfaces

### Go Types

```go
type VisualReanalysisTrigger string

const (
	VisualTriggerInitialProfile      VisualReanalysisTrigger = "initial_profile"
	VisualTriggerDeficiencyReanalysis VisualReanalysisTrigger = "deficiency_reanalysis"
	VisualTriggerManual              VisualReanalysisTrigger = "manual"
)

type DatasetVisualAnalysis struct {
	ID                    string                  `json:"id"`
	ProjectID             string                  `json:"project_id"`
	DatasetID             string                  `json:"dataset_id"`
	DatasetName           string                  `json:"dataset_name"`
	SchemaVersion         string                  `json:"schema_version"`
	AnalysisVersion       int                     `json:"analysis_version"`
	PromptVersion         string                  `json:"prompt_version"`
	AgentName             string                  `json:"agent_name"`
	AgentVersion          string                  `json:"agent_version"`
	Provider              string                  `json:"provider"`
	Model                 string                  `json:"model"`
	TriggerReason         VisualReanalysisTrigger `json:"trigger_reason"`
	TriggerDetails        map[string]any          `json:"trigger_details,omitempty"`
	SourceJobID           string                  `json:"source_job_id,omitempty"`
	SourceInvocationID    string                  `json:"source_invocation_id,omitempty"`
	ProfileSchemaVersion  string                  `json:"profile_schema_version,omitempty"`
	ProfileFingerprint    string                  `json:"profile_fingerprint,omitempty"`
	TotalImages           int                     `json:"total_images"`
	ImagesAnalyzed        int                     `json:"images_analyzed"`
	CoverageReport        VisualCoverageReport    `json:"coverage_report"`
	ClassesToWatch        []ClassWatchItem        `json:"classes_to_watch"`
	Confidence            string                  `json:"confidence"`
	VisualTraits          []VisualTrait           `json:"visual_traits"`
	PreprocessingHypotheses []PreprocessingHypothesis `json:"preprocessing_hypotheses"`
	Cautions              []VisualCaution         `json:"cautions"`
	Limitations           []string                `json:"limitations"`
	ValidationStatus      string                  `json:"validation_status"`
	ValidationErrors      []string                `json:"validation_errors,omitempty"`
	CreatedAt             time.Time               `json:"created_at"`
	UpdatedAt             time.Time               `json:"updated_at"`
}

type VisualCoverageReport struct {
	SelectionStrategy       string         `json:"selection_strategy"`
	SelectionBasis          []string       `json:"selection_basis"`
	ImagesAvailable         int            `json:"images_available"`
	ImagesAnalyzed          int            `json:"images_analyzed"`
	ClassesTotal            int            `json:"classes_total"`
	ClassesCovered          int            `json:"classes_covered"`
	ClassCoverageRatio       float64        `json:"class_coverage_ratio"`
	PerClassCounts          map[string]int `json:"per_class_counts,omitempty"`
	HardExampleCount        int            `json:"hard_example_count"`
	EdgeCaseCount           int            `json:"edge_case_count"`
	HighDetailImageCount    int            `json:"high_detail_image_count"`
	Limitations             []string       `json:"limitations"`
}

type VisualTrait struct {
	Trait           string   `json:"trait"`
	Level           string   `json:"level"`
	Confidence      string   `json:"confidence"`
	Evidence         []string `json:"evidence"`
	ExampleImageIDs  []string `json:"example_image_ids,omitempty"`
	AffectedClasses  []string `json:"affected_classes,omitempty"`
	Notes            string   `json:"notes,omitempty"`
}

type ClassWatchItem struct {
	ClassName       string   `json:"class_name"`
	Reason          string   `json:"reason"`
	RelatedClasses  []string `json:"related_classes,omitempty"`
	Evidence         []string `json:"evidence"`
	ExampleImageIDs  []string `json:"example_image_ids,omitempty"`
	Confidence      string   `json:"confidence"`
}

type PreprocessingHypothesis struct {
	ID                         string                           `json:"id"`
	Mechanism                  string                           `json:"mechanism"`
	Summary                    string                           `json:"summary"`
	Evidence                   []string                         `json:"evidence"`
	SuggestedPreprocessing      *plans.Preprocessing             `json:"suggested_preprocessing,omitempty"`
	SuggestedImageSizes         []int                            `json:"suggested_image_sizes,omitempty"`
	SuggestedAugmentationPolicy string                           `json:"suggested_augmentation_policy,omitempty"`
	SuggestedAugmentationConfig *plans.AugmentationPolicyConfig  `json:"suggested_augmentation_policy_config,omitempty"`
	ExpectedEffect             string                           `json:"expected_effect"`
	Risk                       string                           `json:"risk,omitempty"`
	Confidence                 string                           `json:"confidence"`
	SupportStatus              string                           `json:"support_status"`
	UnsupportedReason          string                           `json:"unsupported_reason,omitempty"`
}

type VisualCaution struct {
	Operation       string   `json:"operation"`
	Reason          string   `json:"reason"`
	Severity        string   `json:"severity"`
	Confidence      string   `json:"confidence"`
	AffectedClasses  []string `json:"affected_classes,omitempty"`
	ExampleImageIDs  []string `json:"example_image_ids,omitempty"`
}

type PlannerVisualExemplarContext struct {
	Enabled              bool                       `json:"enabled"`
	EvidenceOnly         bool                       `json:"evidence_only"`
	AnalysisID           string                     `json:"analysis_id,omitempty"`
	AnalysisVersion      int                        `json:"analysis_version,omitempty"`
	TriggerReason        string                     `json:"trigger_reason,omitempty"`
	ImagesAnalyzed       int                        `json:"images_analyzed"`
	ClassCoverage         VisualCoverageReport       `json:"class_coverage"`
	Summary              string                     `json:"summary"`
	ObservedTraits       []string                   `json:"observed_traits"`
	ClassesToWatch       []ClassWatchItem           `json:"classes_to_watch,omitempty"`
	PreprocessingHypotheses []PreprocessingHypothesis `json:"preprocessing_hypotheses,omitempty"`
	Cautions             []VisualCaution            `json:"cautions,omitempty"`
	Limitations          []string                   `json:"limitations,omitempty"`
	RawImagesIncluded    bool                       `json:"raw_images_included"`
	PromptBudget         int                        `json:"prompt_budget"`
}
```

### Storage

Add a durable `dataset_visual_analyses` table rather than using dormant `dataset_profiles` or stuffing full analysis into `datasets.profile`.

Important columns:

- IDs: `id`, `project_id`, `dataset_id`, `source_job_id`, `source_invocation_id`
- versioning: `schema_version`, `analysis_version`, `profile_fingerprint`, `prompt_version`
- trigger: `trigger_reason`, `trigger_details`
- output JSONB: `coverage_report`, `classes_to_watch`, `visual_traits`, `preprocessing_hypotheses`, `cautions`, `limitations`
- validation: `validation_status`, `validation_errors`
- timestamps: `created_at`, `updated_at`

Add store methods:

- `CreateDatasetVisualAnalysis`
- `GetLatestAcceptedDatasetVisualAnalysis`
- `ListDatasetVisualAnalyses`
- `RejectDatasetVisualAnalysis`

Add API endpoints:

- `GET /datasets/:id/visual-analyses`
- `GET /datasets/:id/visual-analyses/latest`
- `POST /datasets/:id/visual-analysis-result`
- `POST /datasets/:id/visual-analyses/run` for manual trigger

## 4. Visual Agent Output JSON Schema

```json
{
  "schema_version": "dataset_visual_analysis_v1",
  "dataset_id": "dataset_123",
  "dataset_name": "flowers",
  "total_images": 12000,
  "images_analyzed": 48,
  "trigger_reason": "initial_profile",
  "confidence": "medium",
  "coverage_report": {
    "selection_strategy": "deterministic_risk_and_representative_sampling",
    "selection_basis": ["class_representative", "aspect_ratio_outlier", "blur_outlier"],
    "images_available": 12000,
    "images_analyzed": 48,
    "classes_total": 102,
    "classes_covered": 38,
    "class_coverage_ratio": 0.3725,
    "per_class_counts": {"daisy": 2},
    "hard_example_count": 0,
    "edge_case_count": 14,
    "high_detail_image_count": 4,
    "limitations": ["Sample is not class-complete."]
  },
  "visual_traits": [
    {
      "trait": "background_dominance",
      "level": "high",
      "confidence": "medium",
      "evidence": ["Several sampled images show small centered subjects with large background regions."],
      "example_image_ids": ["img_0007", "img_0019"],
      "affected_classes": ["daisy", "sunflower"],
      "notes": "May favor preserve-aspect or crop ablation."
    }
  ],
  "classes_to_watch": [
    {
      "class_name": "daisy",
      "reason": "Visually similar to sampled sunflower images under close crop.",
      "related_classes": ["sunflower"],
      "evidence": ["Petal color and central disk shape overlap in sampled examples."],
      "example_image_ids": ["img_0007", "img_0024"],
      "confidence": "medium"
    }
  ],
  "preprocessing_hypotheses": [
    {
      "id": "vh_001",
      "mechanism": "resolution_crop",
      "summary": "Compare preserve_aspect_pad at 256 against current squash resize.",
      "evidence": ["Aspect ratios vary and subjects are sometimes small."],
      "suggested_preprocessing": {
        "resize_strategy": "preserve_aspect_pad",
        "normalization": "imagenet",
        "crop_strategy": "none",
        "bbox_mode": "ignore"
      },
      "suggested_image_sizes": [256],
      "expected_effect": "Reduce shape distortion and preserve small-object detail.",
      "risk": "Higher latency than 224 squash.",
      "confidence": "medium",
      "support_status": "supported"
    }
  ],
  "cautions": [
    {
      "operation": "vertical_flip",
      "reason": "Several objects appear orientation-sensitive.",
      "severity": "medium",
      "confidence": "medium",
      "affected_classes": ["daisy"]
    }
  ],
  "limitations": [
    "Visual analysis is based on a bounded sentinel sample, not full dataset inspection.",
    "No experiment should be scheduled from this output without backend validation."
  ]
}
```

Allowed enum defaults:

- `confidence`: `low`, `medium`, `high`
- `trait`: `small_objects`, `large_objects`, `background_dominance`, `lighting_variation`, `blur`, `fine_grained_similarity`, `color_texture_signal`, `crop_bbox_useful`, `visual_ambiguity`, `orientation_sensitive`, `text_or_watermark`, `domain_shift_possible`
- `mechanism`: existing planner mechanisms, especially `resolution_crop`, `bbox_crop_ablation`, `augmentation_basic`, `augmentation_auto`, `regularization`, `label_noise_audit`, `hard_example_audit`
- `support_status`: `supported`, `unsupported`, `needs_backend_validation`

## 5. Prompt Design

System prompt essentials:

```text
You are the Visual Dataset Analysis Agent for Model Express.

You analyze a bounded sample of dataset images and metadata. You do not schedule experiments, create jobs, mutate datasets, relabel examples, delete files, or produce executable authority. Your output is visual evidence and preprocessing hypotheses only.

Use only the provided dataset metadata, sample manifest, and attached images. Refer to images by image_id only. Do not mention local file paths. Be explicit about uncertainty and coverage limitations. If the sample is not class-complete, say so.

Return JSON only matching dataset_visual_analysis_v1.
```

User prompt payload:

- dataset metadata: `dataset_id`, `dataset_name`, `classes`, class counts, profile stats, profile visual traits, artifacts, bbox availability
- sample manifest: `image_id`, class, width, height, selection basis, detail level, optional bbox metadata
- allowed operations: resize, crop, normalization, augmentation policy names, bbox modes
- trigger reason and budget: image count, high-detail count, byte/token estimate

Prompt rules:

- Prefer concise dataset-level observations.
- Mark hypotheses `unsupported` if they need operations outside the allowed set.
- Do not infer class-wide truth from one example; use `confidence: low`.
- Never output `proposed_experiments`, job configs, shell commands, dataset mutations, or labels to change.
- For augmentations, call out invalid/risky transforms such as vertical flip, rotation, color jitter, random erasing, MixUp/CutMix when visually risky.

## 6. Backend Validation Rules

Deterministic backend validation owns safety:

- Dataset identity must match the path parameter and current dataset.
- `schema_version` must be recognized.
- `images_analyzed` must be positive and no greater than the submitted sample manifest and configured cap.
- `coverage_report` counts must be internally consistent and bounded.
- Every `example_image_id` must exist in the submitted manifest.
- Enum fields must match allowed values.
- Text fields, array lengths, and JSON byte size must be capped.
- Local paths, absolute paths, S3 credentials, and arbitrary file references must be stripped or rejected.
- Output containing `jobs`, `proposed_experiments`, `commands`, `delete`, `relabel`, or direct execution authority is rejected.
- `bbox_crop_ablation` and bbox crop hypotheses require backend-profiled bbox/annotation evidence.
- Suggested preprocessing and augmentation fields must pass the same allowed-value validation as planned experiments.
- Unsupported suggestions may be persisted only as cautions/limitations, not exposed as runnable planner hypotheses.
- Accepted analyses are immutable; reruns create new versions.
- Persist raw LLM output in `agent_invocations`, but planner context uses only compressed validated fields.

## 7. Planner Consumption

The Experiment Planner remains separate:

- Do not include visual images, visual prompt messages, raw Visual Agent output, or full analysis JSON in planner prompts.
- Extend planner input building to fetch the latest `validation_status=accepted` analysis for the dataset.
- Convert it into a capped `PlannerVisualExemplarContext`:
  - max 8 traits
  - max 6 classes to watch
  - max 6 preprocessing hypotheses
  - max 6 cautions
  - max 8 limitations
- Keep `RawImagesIncluded=false` and `EvidenceOnly=true`.
- Planner may cite visual evidence when proposing `resolution_crop`, `bbox_crop_ablation`, augmentation, or audit mechanisms.
- Existing backend validation still decides whether a proposed experiment is valid and schedulable.

## 8. Modal/Python Image And Preprocessing Plan

### Visual Sampling And Preparation

Add Python modules for visual analysis:

- `worker/datasets/visual_sampling.py`
- `worker/visual_analysis/agent.py`
- `worker/visual_analysis/schema.py`
- `worker/visual_analysis/client.py`

Sampling policy:

- For small class counts, include one representative per class where budget allows.
- For large class counts, use risk coverage, not full class coverage:
  - representative samples
  - rare classes
  - dominant classes
  - aspect-ratio outliers
  - resolution outliers
  - blur/brightness/contrast outliers
  - bbox/object-scale outliers
  - hard examples/confusion examples when available
- Default cap: 32-64 images, with up to 4-8 high-detail images.
- Generate a sample manifest with stable `image_id`; do not send local paths to the LLM.
- Use low/medium resized JPEGs for broad scan; reserve high-detail images for tiny objects, text, defects, or fine-grained classes.

### Preprocessing Extensibility

Refactor current transform construction into a registry-style system:

```text
PreprocessingRegistry
  resize strategies:
    squash
    preserve_aspect_pad
    center_crop
    random_resized_crop

  crop strategies:
    none
    center_crop
    random_resized_crop
    bbox_crop_if_available
    bbox_crop_ablation

  normalization:
    imagenet
    dataset
    none

  augmentation policy mapping:
    none/light/moderate/strong/basic
    randaugment
    trivialaugment
    autoaugment
    mixup
    cutmix
```

Rules:

- Registry specs define name, required metadata, train/eval behavior, builder function, and validation function.
- Bbox-aware strategies require parsed bbox lookup and graceful fallback when boxes are unavailable.
- Unsupported LLM suggestions are marked unsupported by backend and never scheduled.
- Worker still fails clearly if an unsupported operation reaches execution.
- Unit tests cover every registry strategy.
- Integration tests prove a backend-accepted config generated from visual hypotheses can build transforms and run a minimal Modal/local training path.

## 9. Deterministic Logic Vs LLM Logic

Deterministic backend/Python logic:

- dataset profiling
- sample selection and caps
- image resizing/detail policy
- bbox availability detection
- LLM budget/cooldown/rerun eligibility
- JSON schema validation
- supported preprocessing validation
- persistence/versioning
- planner context compression
- experiment scheduling validation

LLM logic:

- qualitative visual observations
- ambiguity and class similarity notes
- classes to watch
- preprocessing hypotheses
- augmentation cautions
- confidence and limitations

## 10. PR Execution Roadmap

### PR 1: Dataset Visual Analysis Persistence And API

- **Status:** Completed in the current implementation pass.
- **Owner:** Backend/schema subagent
- **Goal:** Add durable storage and read/write APIs for validated visual analyses.
- **Likely modules:** `services/orchestrator/internal/datasets`, `store`, `api`, migration SQL.
- **Steps:** Add `DatasetVisualAnalysis` types, `dataset_visual_analyses` table, store methods, list/latest/result endpoints, execution events.
- **Tests:** Postgres and memory store CRUD, latest accepted version lookup, rejected validation status, API auth-free local route behavior consistent with existing APIs.
- **Dependencies:** None.
- **Acceptance:** Backend can persist/list latest accepted analysis without touching planner.
- **Risks:** Avoid activating dormant `dataset_profiles`; avoid bloating `datasets.profile`.

### PR 2: Visual Sample Pack Generation Job

- **Status:** Completed in the current implementation pass.
- **Owner:** Modal/Python preprocessing subagent
- **Goal:** Add deterministic bounded image sampling and preparation.
- **Likely modules:** `worker/jobs.py`, `worker/training/modal_provider.py`, `worker/datasets/visual_sampling.py`, `worker/visual_analysis/*`.
- **Steps:** Add `analyze_dataset_visuals` job shell, download/extract dataset, select representative/risk/hard samples, emit sample manifest, prepare resized/high-detail JPEG inputs.
- **Tests:** Sampling is deterministic, capped, class-aware, risk-aware, strips local paths, handles huge class counts.
- **Dependencies:** PR 1 endpoint shape for final callback, but sampling can be built independently.
- **Acceptance:** Worker can produce a sample manifest and image payload under configured caps.
- **Risks:** Cost creep from too many images; leaking local paths; sampling all classes on huge datasets.

### PR 3: Visual LLM Agent Prompt, Client, And Schema

- **Status:** Completed in the current implementation pass.
- **Owner:** LLM prompt/schema subagent
- **Goal:** Implement separate visual LLM invocation with strict JSON output.
- **Likely modules:** `worker/visual_analysis/agent.py`, `worker/visual_analysis/client.py`, backend validation helpers.
- **Steps:** Add `MODEL_EXPRESS_VISUAL_LLM_*` settings, prompt template, multimodal request builder, JSON schema parser, raw-output callback to backend.
- **Tests:** Golden prompt tests, fake LLM schema tests, malformed output rejection, unsupported-operation marking.
- **Dependencies:** PR 2 for sample manifest; PR 1 for persistence callback.
- **Acceptance:** Visual LLM call is fully separate from planner LLM config and prompt history.
- **Risks:** Provider-specific multimodal request shape; output verbosity; missing image support in local-compatible providers.

### PR 4: Preprocessing Registry Refactor

- **Status:** Completed in the current implementation pass.
- **Owner:** Modal/Python preprocessing subagent
- **Goal:** Make image preprocessing extensible and testable.
- **Likely modules:** `worker/training/modal_app.py`, new `worker/training/preprocessing_registry.py`, worker tests.
- **Steps:** Move resize/crop/normalization/augmentation builders into registry specs; preserve existing behavior; add bbox-aware strategy metadata.
- **Tests:** Unit tests for every resize, crop, normalization, augmentation, bbox mode; train-only augmentation behavior; clear unsupported errors.
- **Dependencies:** None, but should align enum names with backend.
- **Acceptance:** Current training tests pass and every supported operation has a registry test.
- **Risks:** Behavior drift in existing transforms; Modal dependency availability in tests.

### PR 5: Backend Validation And Trigger Orchestration

- **Status:** Completed in the current implementation pass. Manual/initial-profile queueing, deficiency-trigger reanalysis, result validation, accepted/rejected persistence, no-direct-scheduling guards, bbox evidence checks, leakage guards, cooldowns, and max-run guards are implemented.
- **Owner:** Backend/schema plus validation/testing subagent
- **Goal:** Queue visual analysis at the right times and validate outputs before persistence.
- **Likely modules:** `api/handlers.go`, `jobs/model.go`, store, tests.
- **Steps:** Add `TemplateAnalyzeDatasetVisuals`, initial-profile trigger, manual trigger endpoint, deficiency trigger evaluator, cooldown/max-rerun guards, analysis output validator.
- **Tests:** Initial run once, manual rerun creates new version, deficiency rerun only after thresholds, bbox hypotheses rejected without bbox evidence, unsupported operations not exposed as runnable.
- **Dependencies:** PR 1, PR 2, PR 3.
- **Acceptance:** Visual analysis can run end-to-end and cannot schedule experiments.
- **Risks:** Accidentally coupling to initial plan creation; duplicate expensive jobs; weak trigger thresholds.

### PR 6: Planner Compressed Context Integration

- **Status:** Completed in the current implementation pass.
- **Owner:** Planner integration subagent
- **Goal:** Let Experiment Planner consume only compressed accepted visual evidence.
- **Likely modules:** `experiment_planner_llm.go`, `api/handlers.go`, planner tests.
- **Steps:** Fetch latest accepted analysis in planner input, build capped `PlannerVisualExemplarContext`, update prompt wording to say visual evidence is advisory, preserve raw-context exclusions.
- **Tests:** Planner prompt includes compressed analysis, excludes raw images/raw visual output, caps arrays, cites evidence-only status, backend validation still blocks unsupported proposals.
- **Dependencies:** PR 1; can use fixture analysis before PR 5.
- **Acceptance:** Planner receives useful visual card without sharing Visual Agent context window.
- **Risks:** Prompt bloat; accidental copying of raw analysis.

### PR 7: End-To-End Validation Harness

- **Status:** Completed in the current implementation pass. Fake-LLM, backend, planner, worker sampling, schema, preprocessing, cooldown/max-run, deficiency-rerun, and no-direct-scheduling checks are implemented. Real visual LLM, real Modal, and live deployment Postgres smoke tests remain operational validation items for the first real loop.
- **Owner:** Validation/testing subagent
- **Goal:** Prove the complete chain works safely.
- **Likely modules:** backend integration tests, worker tests, test fixtures.
- **Steps:** Build fixture dataset, fake visual LLM response, accepted visual analysis, planner proposal using visual evidence, backend validation, worker transform build.
- **Tests:** E2E happy path, malformed LLM output, unsupported hypothesis, huge-class sampling cap, deficiency rerun, no direct job scheduling from Visual Agent.
- **Dependencies:** PR 1-6.
- **Acceptance:** One command can validate the visual-analysis path without real LLM or real Modal spend.
- **Risks:** Test brittleness around optional torchvision/Modal dependencies.

### PR 8: Mission Control Observability

- **Status:** Completed in the current implementation pass. Evidence/status display, backend rerun policy display, cooldown/max-run/active-job disabled states, and manual rerun controls are implemented.
- **Owner:** UI/observability subagent
- **Goal:** Make visual analysis status, evidence, cost risk, and rerun controls inspectable.
- **Likely modules:** `apps/mission-control/src/types.ts`, `App.tsx`, API client.
- **Steps:** Show latest analysis status, trigger reason, coverage, traits, hypotheses, cautions, limitations, validation errors, manual rerun button.
- **Tests:** Build, fixture rendering, empty/error states, rerun action disabled during cooldown/running job.
- **Dependencies:** PR 1 and PR 5.
- **Acceptance:** Operators can see why visual evidence exists and why it is limited.
- **Risks:** UI implying LLM hypotheses are approved actions; too much raw detail in planner-facing surfaces.

## 11. Assumptions And Defaults

- Visual LLM runs in the worker/Modal visual-analysis job using separate `MODEL_EXPRESS_VISUAL_LLM_*` config.
- Backend remains the source of truth and persists only validated analysis.
- `dataset_visual_analyses` is a new durable table; `dataset_profiles` remains dormant.
- Initial visual analysis is enabled only when visual LLM settings are configured and project/automation settings allow it.
- Planner receives compressed evidence only; no raw images, no raw Visual Agent output, no shared message stream.
- V1 supports existing backend preprocessing enums first; unsupported LLM ideas are preserved as cautions/limitations, not schedulable hypotheses.
