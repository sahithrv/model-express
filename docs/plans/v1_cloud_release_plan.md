# Model Express v1 Cloud Release Plan

## Release scope

v1 should ship as a **Cloud Agentic Demo / Modal + OpenAI required** product:

- Mission Control and the Go orchestrator run locally.
- Training, dataset profiling, and cloud worker execution use Modal.
- OpenAI powers the LLM agent/tool-loop path through the Responses API.
- Users provide Modal, OpenAI, and reachable S3-compatible storage credentials.
- Local training remains available for development, but it is not a v1 release blocker.
- The release succeeds only if a fresh cloud-backed run fails early with clear configuration errors.

Assumption: managed S3-compatible storage is the preferred v1 path. Local MinIO tunneling can remain as an explicit advanced/demo fallback, but should not be the happy path.

## P0 blockers

### 1. No cloud-first v1 env profile

Problem: `.env.v1.local.example` is safe-local oriented, while `.env.example` mixes cloud hints with local defaults.

Why it can stop a run: a fresh user can start with local provider defaults, missing OpenAI keys, default MinIO credentials, or no public Modal callback/storage URL.

Files/functions to inspect or change:

- `.env.example`
- `.env.v1.local.example`
- `services/orchestrator/internal/config/env.go`
- `apps/mission-control/electron/main.cjs` `loadRepoEnv`, `missionControlEnv`
- `services/orchestrator/internal/api/settings.go` `automationSettingsFromEnv`

Minimal fix:

- Add `.env.v1.cloud.example`.
- Document starting with `MODEL_EXPRESS_ENV_FILE=.env.v1.cloud`.
- Keep `.env.v1.local.example` but do not present it as v1 cloud release config.
- Set provider defaults to `modal`, LLM enabled, OpenAI Responses, and memory embeddings disabled.

Acceptance criteria:

- Fresh env file makes Mission Control settings show provider `modal`, GPU `T4`, LLM provider `openai`, LLM enabled.
- No default MinIO root credentials are used for Modal.
- The env file contains no secrets.

Tests/manual validation:

- `go test ./...` from `services/orchestrator`.
- `npm run build` from `apps/mission-control`.
- Manual: start with only `.env.v1.cloud` and confirm `/settings/automation` returns cloud defaults.

### 2. OpenAI key handling is easy to misconfigure

Problem: Go LLM config reads `MODEL_EXPRESS_LLM_API_KEY` and visual fallback keys, not `OPENAI_API_KEY`. Python visual LLM reads only `MODEL_EXPRESS_VISUAL_LLM_API_KEY`.

Why it can stop a run: many users will provide only `OPENAI_API_KEY`; the LLM path then fails during planner/reviewer work.

Files/functions to inspect or change:

- `services/orchestrator/internal/llm/model.go` `ConfigFromEnv`
- `services/worker/worker/visual_analysis/client.py` `VisualLLMConfig.from_env`
- `services/orchestrator/internal/llm/model_test.go`
- `services/worker/tests/test_visual_analysis_agent.py`

Minimal fix:

- Add a narrow fallback to `OPENAI_API_KEY` only when provider is `openai` and product-specific keys are empty.
- Preflight should report the resolved source as `MODEL_EXPRESS_LLM_API_KEY`, `MODEL_EXPRESS_VISUAL_LLM_API_KEY`, or `OPENAI_API_KEY`, without printing the value.
- Keep explicit product-specific keys in the cloud env example.

Acceptance criteria:

- `MODEL_EXPRESS_LLM_API_KEY` works.
- `OPENAI_API_KEY` alone works for OpenAI provider.
- Missing key fails preflight before upload/execution with a clear message.

Tests/manual validation:

- Add Go tests for `OPENAI_API_KEY` fallback.
- Add worker visual config test for `OPENAI_API_KEY` fallback.
- Manual: remove product-specific keys, set only `OPENAI_API_KEY`, run cloud preflight.

### 3. Cloud errors are discovered too late

Problem: dataset preflight only checks local folder/upload size. Modal/OpenAI/S3/callback errors surface after dataset upload, worker start, or Modal job claim.

Why it can stop a run: users wait through upload/profile/queueing only to hit missing Modal auth, unreachable storage, invalid OpenAI key, or public callback auth failure.

Files/functions to inspect or change:

- `apps/mission-control/electron/main.cjs` `preflightDatasetFolder`, `ensureProjectWorker`, `ensureRemoteTrainingSession`
- `apps/mission-control/electron/preload.cjs`
- `apps/mission-control/src/vite-env.d.ts`
- `apps/mission-control/src/App.tsx` `createProjectWithDataset`, `executePlan`, `scheduleFollowUpExperiments`
- `apps/mission-control/src/hooks/useWorkerSupervisor.ts`
- `services/orchestrator/internal/api/router.go`

Minimal fix:

- Add `GET /preflight/cloud` or `POST /preflight/cloud` in the orchestrator for backend-visible checks.
- Add Electron IPC `cloud:preflight` that aggregates backend, env, Modal, OpenAI, S3, and public URL checks.
- Gate dataset upload, plan execution, and Modal worker start on preflight success.

Acceptance criteria:

- With missing OpenAI key, upload is blocked before project creation.
- With default MinIO root credentials, Modal run is blocked before worker start.
- With wrong `MODEL_EXPRESS_API_TOKEN`, public orchestrator preflight fails before Modal submission.
- UI shows direct remediation text.

Tests/manual validation:

- Electron unit tests in `apps/mission-control/electron/main.test.cjs`.
- Go handler tests for preflight response shape.
- Manual: intentionally break each env var and confirm the failure happens before upload/execution.

### 4. Modal storage and callback contract needs pre-run proof

Problem: the code has good guards, but they are spread across Electron and worker paths. Modal storage refuses default MinIO root credentials in `modal_provider.py`, and callbacks use attempt tokens, but users only learn after job dispatch.

Why it can stop a run: Modal cannot read the dataset, cannot write artifacts, or cannot callback to protected orchestrator endpoints.

Files/functions to inspect or change:

- `services/worker/worker/training/modal_provider.py` `_modal_storage_payload_for_jobs`, `_modal_storage_credentials`, `_modal_orchestrator_url`
- `services/worker/worker/training/modal_callbacks.py`
- `services/worker/worker/orchestrator_client.py` `_api_headers`, `_callback_headers`
- `services/orchestrator/internal/api/router.go` `apiTokenMiddleware`
- `services/orchestrator/internal/api/jobs.go` `validateJobCallback`, `augmentPolledJob`
- `apps/mission-control/electron/main.cjs` `validateRemoteModalUrl`, `ensureRemoteTrainingSession`

Minimal fix:

- Reuse or mirror the Modal storage credential checks in cloud preflight.
- Check `MODAL_ORCHESTRATOR_URL` is HTTPS, public, and authenticated with `MODEL_EXPRESS_API_TOKEN`.
- Check `MODAL_S3_ENDPOINT_URL` is HTTPS/public and not MinIO console/Postgres/MLflow.
- Add a tiny S3 write/read/delete preflight object under `model-express/preflight/`.
- Add a live Modal no-GPU ping only if feasible in the 1-2 day window; otherwise check Modal import/auth config and make the real first Modal failure clear.

Acceptance criteria:

- Preflight rejects `model_express` / `model_express_password` unless explicitly allowed.
- Preflight rejects `:9001`, private IPs, and non-HTTPS Modal URLs.
- Public `/settings/automation` check succeeds only with the configured API token.
- Modal callbacks still require per-attempt callback tokens.

Tests/manual validation:

- Electron tests for URL and root credential failures.
- Existing Go callback-token tests remain green.
- Manual: run one Modal profile and confirm callbacks succeed with protected orchestrator URL.

### 5. Cloud execution settings are partly hidden

Problem: plan execution in `App.tsx` hardcodes `{ provider: "modal", gpu_type: "T4" }`, while profile/other worker paths use settings. Manual visual rerun can start a local worker even when default provider is Modal.

Why it can stop a run: UI settings can claim one provider/GPU while execution uses another; cloud jobs may be queued without the right worker.

Files/functions to inspect or change:

- `apps/mission-control/src/App.tsx` `createProjectWithDataset`, `executePlan`, `scheduleFollowUpExperiments`, `requestVisualAnalysisRerun`, `ensureChampionBackendWorker`
- `apps/mission-control/src/hooks/useWorkerSupervisor.ts`
- `services/orchestrator/internal/api/settings.go` `defaultExecuteExperimentPlanRequest`

Minimal fix:

- Use automation settings for provider/GPU everywhere.
- For cloud-v1, preflight fails unless provider is `modal`.
- Keep T4 as the default cloud-v1 GPU, but make it visible and editable through settings/env.
- Add honest UI copy: "Cloud Agentic Demo / Modal + OpenAI required."

Acceptance criteria:

- Changing default GPU from `T4` to `L4` updates plan execution request and worker requirement.
- Manual visual-analysis rerun starts a Modal dispatcher when provider is Modal.
- No hidden local fallback occurs in cloud-v1 mode.

Tests/manual validation:

- Frontend test if execution payload logic is extracted.
- Manual: set `MODEL_EXPRESS_DEFAULT_GPU_TYPE=L4`, execute plan, confirm job config and requirement use `L4`.

## P1 hardening

- Add `scripts/cloud_preflight.ps1` as a thin wrapper around the Electron/backend preflight for CLI users.
- Add a release doc `docs/plans/v1_cloud_release_checklist.md`.
- Add clearer activity events when preflight blocks a worker requirement.
- Add Modal call/preflight status display to existing Mission Control status areas; no new dashboard redesign.
- Keep export download restricted to existing safe `s3://` and validated local artifact paths.

## `.env.v1.cloud.example`

```env
# Model Express v1 cloud-first profile.
# Copy to .env.v1.cloud and start processes with:
# MODEL_EXPRESS_ENV_FILE=.env.v1.cloud

MODEL_EXPRESS_V1_PROFILE=cloud
MODEL_EXPRESS_EXECUTION_PROFILE=fast-remote

# Local orchestrator, protected when exposed through a public tunnel.
MODEL_EXPRESS_ORCHESTRATOR_ADDR=127.0.0.1:8080
MODEL_EXPRESS_ALLOW_LAN=true
MODEL_EXPRESS_ORCHESTRATOR_TUNNEL_MODE=true
MODEL_EXPRESS_API_TOKEN=replace-with-long-random-token
MODAL_ORCHESTRATOR_URL=https://your-public-orchestrator-tunnel.example.com

# Public S3-compatible storage reachable by both Mission Control and Modal.
S3_ENDPOINT_URL=https://your-s3-compatible-endpoint.example.com
MODAL_S3_ENDPOINT_URL=https://your-s3-compatible-endpoint.example.com
S3_BUCKET=model-express
MODEL_EXPRESS_ARTIFACT_BUCKET=model-express
MODEL_EXPRESS_ARTIFACT_PREFIX=model-express/artifacts
AWS_DEFAULT_REGION=us-east-1

# Scoped storage credentials. Do not use local MinIO root credentials for Modal.
AWS_ACCESS_KEY_ID=replace-with-scoped-upload-access-key
AWS_SECRET_ACCESS_KEY=replace-with-scoped-upload-secret-key
MODEL_EXPRESS_MODAL_AWS_ACCESS_KEY_ID=replace-with-scoped-modal-access-key
MODEL_EXPRESS_MODAL_AWS_SECRET_ACCESS_KEY=replace-with-scoped-modal-secret-key
MODEL_EXPRESS_ALLOW_MODAL_ROOT_STORAGE=false
MODEL_EXPRESS_MODAL_TUNNEL_S3=false

# Modal execution defaults.
MODEL_EXPRESS_DEFAULT_TRAINING_PROVIDER=modal
MODEL_EXPRESS_DEFAULT_GPU_TYPE=T4
MODEL_EXPRESS_MODAL_DEFAULT_GPU_TYPE=T4
MODAL_GPU_TYPE=T4
GPU_TYPE=modal
MODEL_EXPRESS_MAX_AUTO_WORKERS=1
MODEL_EXPRESS_MODAL_DISPATCHER_SLOTS=1
MODEL_EXPRESS_MODAL_DISPATCHER_IDLE_POLL_SLOTS=1
MODEL_EXPRESS_MODAL_BATCH_MAX_TRIALS=2

# OpenAI agent runtime.
MODEL_EXPRESS_LLM_ENABLED=true
MODEL_EXPRESS_LLM_PROVIDER=openai
MODEL_EXPRESS_LLM_MODEL=gpt-5.4-mini
MODEL_EXPRESS_LLM_API_STYLE=responses
MODEL_EXPRESS_LLM_STORED_RESPONSES=true
MODEL_EXPRESS_LLM_API_KEY=replace-with-openai-api-key

# Visual dataset analysis. Keep enabled only if preflight passes.
MODEL_EXPRESS_VISUAL_ANALYSIS_ENABLED=true
MODEL_EXPRESS_VISUAL_LLM_ENABLED=true
MODEL_EXPRESS_VISUAL_LLM_PROVIDER=openai
MODEL_EXPRESS_VISUAL_LLM_MODEL=gpt-5.4-mini
MODEL_EXPRESS_VISUAL_LLM_API_STYLE=responses
MODEL_EXPRESS_VISUAL_LLM_API_KEY=replace-with-openai-api-key

# Agent behavior: cloud-first but not runaway.
MODEL_EXPRESS_TRAINING_MONITOR_LLM_ENABLED=true
MODEL_EXPRESS_AGENT_MODE=propose
MODEL_EXPRESS_AUTO_REVIEW_EXPERIMENTS=true
MODEL_EXPRESS_AUTO_SCHEDULE_FOLLOWUPS=false
MODEL_EXPRESS_AUTO_EXECUTE_PLANS=false
MODEL_EXPRESS_MAX_FOLLOWUP_ROUNDS=1

# Memory remains runtime-generated only for v1.
MODEL_EXPRESS_MEMORY_RETRIEVAL_ENABLED=false
MODEL_EXPRESS_MEMORY_EMBEDDINGS_ENABLED=false
MODEL_EXPRESS_AUTOML_ENABLED=false
```

## Cloud preflight design

Backend endpoint:

- `POST /preflight/cloud`
- Request: `{ "stage": "dataset_upload" | "plan_execution" | "worker_start", "live": true }`
- Response: `{ "status": "ok" | "failed", "checks": [{ "id", "status", "message", "remediation" }] }`
- Register route after auth middleware so public/tunnel checks require API token.

Electron IPC:

- `window.missionControl.preflightCloud(options)`
- Implement in `apps/mission-control/electron/main.cjs`.
- Expose in `preload.cjs` and type in `vite-env.d.ts`.

Checks:

- Env profile is cloud-v1.
- Provider is `modal`.
- `MODEL_EXPRESS_API_TOKEN` present when public/tunnel URL is configured.
- `MODAL_ORCHESTRATOR_URL` is public HTTPS and authenticated.
- `S3_ENDPOINT_URL` and `MODAL_S3_ENDPOINT_URL` are not private, not console port `9001`, and point to the expected bucket.
- Scoped S3 credentials exist; default MinIO root credentials are rejected.
- Tiny S3 preflight object write/read/delete succeeds.
- Modal Python package is installed; optional live no-GPU Modal ping succeeds.
- LLM config uses `openai`, `responses`, `gpt-5.4-mini`, stored responses enabled, and key present.
- Optional live OpenAI Responses JSON check succeeds.
- Memory bootstrap is not required or enabled.

Trigger points:

- Before dataset upload in `createProjectWithDataset`.
- Before `/plans/:id/execute`.
- Before `ensureProjectWorker` when `gpuType === "modal"`.
- In `useWorkerSupervisor` before marking worker requirement `ACTIVE`.

User-facing failures:

- "Cloud Agentic Demo requires Modal + OpenAI. Set `MODEL_EXPRESS_LLM_API_KEY` and run preflight again."
- "Modal cannot reach the orchestrator. Set `MODAL_ORCHESTRATOR_URL` to a public HTTPS URL and keep `MODEL_EXPRESS_API_TOKEN` enabled."
- "Public orchestrator returned 401. Mission Control and the orchestrator are using different `MODEL_EXPRESS_API_TOKEN` values."
- "Modal storage refuses default MinIO root credentials. Use scoped S3 credentials."
- "`MODAL_S3_ENDPOINT_URL` points at the MinIO console port. Use the S3 API endpoint."
- "OpenAI Responses check failed for `gpt-5.4-mini`: <redacted provider error>."

## End-to-end cloud smoke test

1. Fresh DB: start Postgres from `infra/compose.yaml`; do not rely on local MinIO for the happy path.
2. Copy `.env.v1.cloud.example` to `.env.v1.cloud`, fill Modal/OpenAI/S3 values.
3. Start orchestrator with `MODEL_EXPRESS_ENV_FILE=.env.v1.cloud`.
4. Start Mission Control with the same env file.
5. Run cloud preflight; all required checks pass.
6. Create a new project from a small image-folder dataset.
7. Confirm upload produces `s3://.../datasets/<project_id>/...zip`.
8. Confirm Modal profile job runs and posts dataset profile.
9. If visual analysis is enabled, confirm Modal visual job and OpenAI visual Responses path complete or fail clearly.
10. Confirm initial plan appears.
11. Execute plan; confirm job config has `provider=modal`, expected GPU, and Modal callback/session metadata.
12. Confirm Modal training reports metrics, summary, evaluation, and artifact S3 URIs.
13. Confirm OpenAI planner/training monitor invocation records show `api_style=responses`.
14. Confirm champion is selected or reviewable.
15. Request export and save portable artifact ZIP.
16. Break one dependency at a time and confirm preflight blocks before upload/execution/worker start.

## Non-goals for v1

- No memory bootstrap.
- No local-only training optimization as a release blocker.
- No large Mission Control redesign.
- No hosted SaaS deployment.
- No production scheduler rewrite.
- No arbitrary HTTP artifact downloads.
- No weakening of API token or callback-token protection for public orchestrator URLs.
