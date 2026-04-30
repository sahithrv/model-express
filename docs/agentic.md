# LLM Subagents + Autonomous ML Loop

## Summary
- Keep deterministic orchestration as the safety layer, but add real LLM subagents that produce structured recommendations.
- Default agent mode: `PROPOSE`, where agents can recommend actions but deterministic code decides what is safe to execute.
- Optional agent mode: `AUTONOMOUS`, where approved agent actions can schedule/execute work within budget, round, and policy limits.
- Support OpenAI and hosted open-source models through a provider interface; local inference can be added later without changing agent logic.

## Auto-Execution Worker Fix
- Keep the hybrid worker approach first:
  - Backend records durable `WorkerRequirement` and `ExecutionEvent` rows when auto execution queues jobs.
  - Mission Control watches pending requirements and satisfies them using existing Electron `ensureProjectWorker`.
  - Manual and automatic flows share the same worker lifecycle helper.
- Files involved:
  - `apps/mission-control/electron/main.cjs`: extract/reuse worker process lifecycle.
  - `apps/mission-control/src/App.tsx`: add background worker-requirement poller and status display.
  - `services/orchestrator/internal/api/handlers.go`: after auto execution queues jobs, create/update worker requirements and execution events.
  - `services/orchestrator/internal/store/*`: persist worker requirements and execution events.

## LLM Agent Runtime
- Add a new agent runtime layer:
  - `services/orchestrator/internal/llm`: provider interface, config, JSON generation, retry/validation.
  - `services/orchestrator/internal/agents`: specialized agent implementations.
  - `services/orchestrator/internal/memory`: shared memory models and retrieval helpers.
- LLM provider config:
  - `MODEL_EXPRESS_LLM_PROVIDER=openai | openai_compatible | local`
  - `MODEL_EXPRESS_LLM_BASE_URL`
  - `MODEL_EXPRESS_LLM_API_KEY`
  - `MODEL_EXPRESS_LLM_MODEL`
  - `MODEL_EXPRESS_AGENT_MODE=propose | autonomous`
- `openai_compatible` should support hosted Llama-style APIs if they expose chat/completions-like JSON behavior.
- All LLM outputs must validate against Go structs before being stored or acted on.

## Specialized Agents
- `DatasetAnalysisAgent`
  - Input: dataset profile.
  - Output: preprocessing, augmentation, metric, and risk recommendations.
  - Stores `dataset_analysis` and `preprocessing_recommendation` memory records.

- `TrainingMonitorAgent`
  - Input: epoch metrics, training run summary, plan, job config, relevant past memory.
  - Output: overfitting/underfitting/plateau/stability/cost findings and pruning or ranking recommendations.
  - Stores `training_evaluation` memory records.

- `ExperimentPlanningAgent`
  - Input: dataset analysis, training evaluations, reviewer decisions, prior successful/failed strategies.
  - Output: proposed experiments and rationale.
  - In `PROPOSE` mode, it only writes recommendations.
  - In `AUTONOMOUS` mode, deterministic guardrails may convert approved proposals into follow-up plans.

## Shared Memory
- Add `AgentMemoryRecord`:
  - `id`, `project_id`, `dataset_id`, `plan_id`, `job_id`
  - `agent_name`, `kind`, `summary`, `payload`, `tags`
  - `created_at`
- Memory retrieval v1:
  - Fetch recent project memory.
  - Fetch same-dataset memory.
  - Fetch similar dataset memories by simple tags: imbalance, small_dataset, many_classes, corrupt_images, model_family, metric.
- “Getting better” means agents receive prior outcomes, mistakes, and successful strategies as context before producing new recommendations.

## Action Safety
- Add `AgentActionProposal`:
  - `action_type`: `ADD_EXPERIMENTS`, `CHANGE_PREPROCESSING`, `PRUNE_RUN`, `RANK_MODELS`, `STOP_PROJECT`
  - `confidence`, `rationale`, `payload`, `requires_approval`
- Deterministic policy validates:
  - max follow-up rounds
  - cost/budget limits
  - valid experiment schema
  - duplicate prevention
  - allowed provider/GPU
  - autonomous mode enabled
- Even in `AUTONOMOUS`, LLMs propose; deterministic code executes.

## Minimal End-To-End Milestone
1. Auto reviewer creates a follow-up plan and queues jobs.
2. Backend creates worker requirements and execution events.
3. Mission Control starts required workers automatically.
4. Completed jobs trigger `TrainingMonitorAgent`.
5. Agent receives run summary, epoch metrics, and relevant memory.
6. Agent emits structured JSON recommendation.
7. Recommendation is validated and stored in shared memory.
8. Mission Control shows latest agent memory/recommendations.
9. `ExperimentPlanningAgent` can read those records in a later milestone to propose improved follow-up experiments.

## Tests
- `go test ./...` for orchestrator.
- Mission Control `npm run build`.
- Add backend tests for:
  - LLM output validation.
  - memory persistence/listing.
  - Training Monitor recommendation storage.
  - `PROPOSE` mode never schedules directly.
  - `AUTONOMOUS` mode still respects deterministic guardrails.
- Add frontend build coverage for worker requirement/event display.
