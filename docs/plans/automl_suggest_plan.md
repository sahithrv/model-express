# Roadmap: Backend-Validated AutoML Hyperparameter Suggestions

## Summary

AutoML should be stronger than a scalar-only tuner, but it must remain a bounded hyperparameter proposal engine for an experiment whose strategy has already been chosen by the LLM and accepted by backend validation.

The LLM remains responsible for strategic choices: model family/name, experiment mechanism, preprocessing, resolution/image-size intent, augmentation strategy/policy type, class-balancing strategy, fine-tuning intent, exploration vs exploitation, training budget intent, and the bounded search constraints. AutoML can only choose concrete hyperparameter values inside those LLM-approved and backend-validated constraints.

AutoML must not create experiment strategies, mutate datasets, choose model families independently, invent preprocessing, schedule jobs, bypass validation, or move sampling into workers. The orchestrator still owns validation and scheduling. Workers still consume concrete config.

## AutoML-Owned Hyperparameter Scope

AutoML v1 may tune these worker-supported hyperparameters when and only when the LLM/backend search space explicitly allows them:

- Core training: `learning_rate`, `weight_decay`, `batch_size`, `epochs`, `early_stopping_patience`, `dropout`, `label_smoothing`, `gradient_clip_norm`.
- Training algorithm choices: `optimizer`, `scheduler`, selected only from LLM-approved categorical options that are also backend-supported. Conditional algorithm knobs include `optimizer_momentum` only when the optimizer is SGD, plus `scheduler_step_size` and `scheduler_gamma` only when the scheduler is `step`.
- Structured augmentation parameters: `augmentation_policy_config.magnitude`, `num_ops`, `num_magnitude_bins`, `probability`, `alpha`, but only after the LLM has chosen the augmentation policy type and backend has validated it.
- Class-balancing parameters: `class_balancing_config.effective_number_beta`, but only when the LLM has chosen `effective_number_loss` or an equivalent supported effective-number strategy; `class_balancing_config.focal_loss_gamma`, but only when the LLM has chosen `focal_loss`.

AutoML v1 must not tune these strategy fields:

- `model`, `template`, model family, model capacity tier.
- `preprocessing`, `resolution_strategy`, `image_size`, bbox/crop/normalization choices.
- `augmentation_policy` or structured augmentation `policy_type`.
- `class_balancing` strategy, `sampling_strategy`.
- `pretrained`, `freeze_backbone`, `fine_tune_strategy`.
- Dataset split policy, labels, manifests, examples, visual-analysis inputs, worker/provider selection, scheduling knobs.

Future PRs may add heavier Bayesian/TPE/GP dependencies or richer acquisition policies after the lightweight adaptive sampler, persistence, and provenance loop are stable.

## PR 1: AutoML Domain Model And Interfaces

**Goal:** Define AutoML as a bounded hyperparameter suggestion layer, not a strategy planner.

**Files/modules:** `services/orchestrator/internal/automl/*`, `services/orchestrator/internal/plans/model.go`.

**New types/interfaces/data models:**

- `ExperimentIntent`: LLM-owned strategic summary and immutable strategy fields.
- `HyperparameterSearchSpace`: parameter specs for numeric, integer, categorical, and conditional parameters.
- `HyperparameterParameterSpec`: name, type, bounds/choices, scale, step, default, source, and dependency.
- `HyperparameterSuggestion`: concrete values and per-value provenance.
- `HyperparameterProvenance`: `llm`, `backend_default`, `user_manual`, `random_search`, `grid_search`, `bayesian_optimizer`, `other_sampler`.
- `OptimizerStudy`, `OptimizerTrial`, `OptimizerFeedbackSummary`.
- `Optimizer` interface with `Suggest(ctx, request)` and `Observe(ctx, observation)`.

**API/schema changes:** Add optional plan metadata for AutoML intent, validated search space, suggestion ID, and provenance. Metadata must separate LLM-owned strategy fields from AutoML-owned hyperparameters.

**Tests to add:** Search-space JSON round trip; provenance assignment; disabled AutoML no-op; rejection when search space includes strategy fields such as `model`, `preprocessing`, `augmentation_policy`, or `image_size`.

**Risks/tradeoffs:** Broader hyperparameter scope increases validation complexity. Keep the boundary explicit so AutoML cannot drift into planning.

**Acceptance criteria:** Types compile and no existing planner/scheduler behavior changes when AutoML is disabled.

## PR 2: Backend Capability Registry And Validation

**Goal:** Add authoritative backend validation for which hyperparameters AutoML may tune and under which constraints.

**Files/modules:** `services/orchestrator/internal/automl/*`, `services/orchestrator/internal/api/handlers.go`, `services/orchestrator/internal/agents/backend_gated_methods.go`.

**New types/interfaces/data models:**

- `HyperparameterCapabilityRegistry`: central registry of supported AutoML-owned knobs.
- `AutoMLSearchSpaceValidator`: validates search spaces before sampling.
- `AutoMLSuggestionValidator`: validates concrete suggestions before plan creation and again before job scheduling.

**Validation rules:**

- Reject any AutoML search space containing strategy-owned fields.
- Require every searchable parameter to be present in the backend capability registry.
- Require categorical choices to be a subset of backend-supported values, e.g. optimizers from `adamw`, `adam`, `sgd`; schedulers from `none`, `cosine`, `step`.
- Require numeric ranges to be inside stricter backend/worker bounds, e.g. positive lr within backend max, weight decay within `[0, 1]`, batch size within runtime-approved values.
- Require conditional parameters to match the LLM-selected strategy: `effective_number_beta` only with effective-number class balancing; MixUp/CutMix `alpha` only when the LLM selected that policy type; RandAugment magnitude/ops only when the LLM selected RandAugment.
- Run existing `validatePlannedExperiment` after AutoML applies suggestions.
- Store validation status and errors with the suggestion.

**API/schema changes:** Add `automl_enabled` automation setting, default false. Optionally expose a bounded `GET /automl/capabilities` or include capabilities in backend-curated planner context.

**Tests to add:** Reject strategy fields; reject unsupported params; reject out-of-bounds ranges; reject conditional params without matching LLM-owned strategy; accept valid broad search spaces.

**Risks/tradeoffs:** The validator becomes a critical safety layer. Keep it deterministic and independent of LLM text.

**Acceptance criteria:** Backend validation is authoritative and can prove AutoML is limited to approved hyperparameters.

## PR 3: Lightweight Pluggable Samplers

**Goal:** Implement first samplers for the broader validated search space without heavy dependencies.

**Files/modules:** `services/orchestrator/internal/automl/samplers/*`.

**New types/interfaces/data models:**

- `SeededRandomSampler`.
- `GridSampler` for small categorical/integer spaces.
- Optional simple exploitation sampler that perturbs around the best prior trial.

**API/schema changes:** None.

**Tests to add:** Deterministic with seed; never emits out-of-bound values; handles mixed numeric/integer/categorical/conditional spaces; handles empty history; gracefully rejects unsupported spaces before sampling.

**Risks/tradeoffs:** Seeded random/grid are simple and auditable; the lightweight adaptive Bayesian sampler exploits prior trials without a heavy dependency, so it is still less expressive than a full TPE/GP optimizer.

**Acceptance criteria:** Samplers propose valid concrete hyperparameter configs without LLM calls.

## PR 4: Persistence And Provenance

**Goal:** Persist studies, search spaces, suggestions, trial observations, validation results, and per-value provenance.

**Files/modules:** `services/orchestrator/internal/store/store.go`, `memory.go`, `postgres.go`, `migrations/002_automl.sql`.

**New types/interfaces/data models:** Store models and methods for studies, suggestions, and trials, linked to plan ID, experiment index, job ID, model selected by LLM, dataset, intent, sampler, seed, search space, concrete values, provenance, validation status, and observed metrics.

**API/schema changes:** Add `automl_studies`, `automl_suggestions`, and `automl_trials` tables. Suggestions should persist a compact snapshot of immutable LLM-owned strategy fields so audits can prove AutoML did not alter them.

**Tests to add:** Memory/Postgres parity; migration idempotency; provenance survives round trip; conditional values persist; trial observations link to suggestions and jobs.

**Risks/tradeoffs:** Store raw search/trial details in backend storage, but expose only bounded summaries to agents/UI by default.

**Acceptance criteria:** Every generated hyperparameter value can be traced to source, sampler, seed, bounds, and final job.

## PR 5: Planner Integration

**Goal:** Insert AutoML after LLM/backend validation and before follow-up plan creation.

**Files/modules:** `experiment_planner_llm.go`, `planner_information_tools.go`, `handlers.go`, `plans/model.go`.

**New types/interfaces/data models:** Planner recommendation fields for `experiment_intent`, immutable strategy snapshot, and per-experiment `hyperparameter_search_space`.

**API/schema changes:** Include AutoML metadata in stored planner decision payloads and follow-up plans.

**Behavior:**

- LLM chooses experiment strategy and which hyperparameters are allowed to vary.
- Backend validates the search space.
- AutoML fills only missing or sampler-owned concrete hyperparameter values.
- Backend validates the suggestion.
- Backend re-validates the final `PlannedExperiment`.
- Orchestrator schedules through the existing normal path.

**Tests to add:** Disabled path unchanged; AutoML preserves LLM-owned strategy fields; optimizer/scheduler categorical suggestions are bounded; conditional augmentation/class-balancing params are enforced; tool calls remain questions and cannot schedule jobs.

**Risks/tradeoffs:** Existing novelty checks may treat hyperparameter-only variants as minor. For AutoML studies, novelty should be attached to the LLM-approved experiment/study rather than treating each trial as a new strategy.

**Acceptance criteria:** Scheduled jobs receive concrete validated hyperparameters through existing orchestrator flow.

## PR 6: Optimizer Feedback To LLM

**Goal:** Summarize AutoML outcomes for LLM planning and training monitoring without raw trial dumps.

**Files/modules:** `experiment_planner_llm.go`, `training_monitor_llm.go`, `planner_information_tools.go`, `training_monitor_information_tools.go`, `handlers.go`.

**New types/interfaces/data models:** `OptimizerFeedbackSummary` with trial count, best hyperparameter set, best score, train/validation gap, failed-trial count, failed parameter patterns, trend, and recommended narrowed ranges.

**API/schema changes:** Add bounded `optimizer_feedback_summary` to planner and monitor context. Raw trial history remains available only through approved bounded tools if needed.

**Tests to add:** Summary size capped; raw trial dumps excluded by default; target metric honored; failed jobs update optimizer state; feedback cannot include scheduling authority.

**Risks/tradeoffs:** Feedback should guide the LLM's next strategic decision without letting AutoML choose the next strategy.

**Acceptance criteria:** LLM receives compact optimizer feedback, not raw trial history.

## PR 7: Worker Compatibility And New Knob Support

**Goal:** Ensure every AutoML-owned knob is executable by workers.

**Files/modules:** `services/worker/worker/training/local.py`, `modal_app.py`, worker tests, orchestrator validation tests.

**New types/interfaces/data models:** Shared supported-knob documentation/fixtures for local and modal workers.

**API/schema changes:** None unless exposing bounded capability metadata.

**Tests to add:** Worker accepts generated configs for all supported AutoML-owned knobs; unsupported optimizer-specific and scheduler-specific knobs are rejected before scheduling; dropout and regularization knobs are executable only through concrete backend-validated config.

**Risks/tradeoffs:** Local simulator and Modal trainer differ in depth. Backend capabilities should use the strict common subset unless provider-specific validation exists.

**Acceptance criteria:** AutoML cannot generate configs workers cannot execute.

## PR 8: API, UI, And Observability

**Goal:** Make AutoML decisions visible without exposing raw prompts or huge payloads.

**Files/modules:** `apps/mission-control/src/App.tsx`, `styles.css`, `types.ts`, orchestrator API handlers, diagnostics packages.

**New types/interfaces/data models:** UI/API summaries for enabled state, sampler, search-space summary, concrete suggestions, provenance per value, validation result, trial feedback summary, and immutable LLM-owned strategy snapshot.

**API/schema changes:** Add bounded AutoML summary endpoint or include compact AutoML metadata in existing plan/job responses.

**Tests to add:** UI build; API response shape tests; redaction tests for raw prompts, raw LLM outputs, images/base64, manifests, and full trial dumps.

**Risks/tradeoffs:** UI must not imply AutoML selected the model/preprocessing/augmentation strategy.

**Acceptance criteria:** User can understand which hyperparameters AutoML chose, why they were allowed, and which fields remained LLM/backend-owned.

## PR 9: Integration And Regression

**Goal:** Verify the full loop.

**Files/modules:** Cross-service integration tests and docs.

**Tests to run:**

- `go test ./...` from `services/orchestrator`.
- `python -m pytest` from `services/worker`.
- `npm run build` from `apps/mission-control`.

**Regression cases:** AutoML disabled; LLM planner path unchanged; backend rejects invalid spaces; AutoML cannot change strategy fields; samplers cannot schedule jobs; worker receives only concrete validated config; optimizer feedback is compact; provenance survives end to end.

**Risks/tradeoffs:** Broader search can consume more jobs. Enforce study-level trial caps, time budgets, and worker/runtime limits before scheduling.

**Acceptance criteria:** AutoML improves hyperparameter coverage while preserving the LLM-as-strategic-planner invariant.
