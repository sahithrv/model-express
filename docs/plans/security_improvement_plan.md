# Security Audit Report

Audit date: 2026-06-10

Scope: Model Express local-first repository, including the Electron/Vite desktop app, Go orchestrator, Python worker, dataset/artifact handling, Docker/local config, database code, and LLM/agent integrations. This plan assumes V1 is primarily local-only, except for LLM provider calls and Modal training. Risks are ranked by practical release impact, especially what happens if a user enables LAN access, a Cloudflare/Modal tunnel, or a compromised renderer/process reaches the local control plane.

## Executive Summary

- Overall risk level: High until the local control plane is loopback-only by default and protected when exposed to Modal, LAN, or tunnels. Most issues become much less severe once unauthenticated remote access is closed.
- Top 5 security concerns:
  1. The orchestrator and Docker services can be exposed on all interfaces with no authentication while examples describe tunnel usage.
  2. Worker/job callback endpoints trust unauthenticated callers and accept some callbacks without a required training attempt binding.
  3. Electron IPC exposes filesystem and network-powered operations that trust renderer-provided paths, artifact URIs, base URLs, and upload endpoints.
  4. Artifact/checkpoint flows can trust unproven local paths or PyTorch checkpoints, including `torch.load` on `.pt`/`.pth` sources.
  5. Dataset archive extraction and upload preflight caps are not strict enough for hostile or malformed archives.
- Release-blocking issues:
  - Fix Findings 1 and 2 before V1 if any workflow uses Modal callbacks, Cloudflare tunnels, LAN access, or non-loopback Docker ports.
  - Fix the Electron IPC path and endpoint validation parts of Finding 3 before V1 desktop distribution.
  - Fix the arbitrary artifact URI and unsafe checkpoint trust parts of Finding 4 before allowing user-provided exports, job configs, or externally sourced artifacts.
- Safe-to-defer issues:
  - Dependency pinning and audit automation can be staged after the P0 exposure fixes, but should still be done before a public V1 tag.
  - Raw LLM trace redaction can be P1 if the API is loopback-only and authenticated before release.
  - JSON request body limits are P1/P2 for a purely local loopback app, but become P0/P1 if the orchestrator is tunnel-accessible.
- Positive guardrails already present:
  - `.env` and `.env.*` are ignored, while `.env.example` is allowed.
  - Electron has `contextIsolation: true` and `nodeIntegration: false`.
  - Go database access reviewed in this pass is mostly parameterized; no release-blocking SQL injection pattern was found.
  - LLM planner and monitor flows use typed parsing/validation and information-only tools in several places.
  - Worker/orchestrator diagnostics include redaction helpers and tests.
  - `npm audit --audit-level=high --json` in `apps/mission-control` reported zero known vulnerabilities at audit time.

## Findings

### Finding: Local Control Plane Can Be Exposed Without Authentication

- Severity: High
- Location: `services/orchestrator/cmd/orchestrator/main.go:46-55`, `services/orchestrator/internal/api/router.go:23-107`, `infra/compose.yaml:7-45`, `.env.example:1-4`, `configs/local.example.yaml:1-20`
- Category: Insecure local server binding, local network exposure, missing auth assumptions for local-only APIs, dangerous dev-only settings leaking into production
- What is wrong:
  - The orchestrator listens on `:8080`, which binds to all interfaces by default.
  - `infra/compose.yaml` publishes Postgres, MLflow, MinIO API, and MinIO console on host ports without a loopback host prefix.
  - Example config uses default local credentials for Postgres and MinIO, and `.env.example` includes tunnel-shaped examples for orchestrator and S3 access.
  - The API router exposes create, mutate, worker, export, telemetry, settings, and cancellation routes without an authentication middleware.
- Why it matters:
  - This is acceptable only if the services are truly reachable only by trusted local processes.
  - Once a user starts Docker on a LAN, opens a Cloudflare tunnel, or binds the orchestrator for Modal access, any reachable client can mutate local state, enqueue work, spoof worker results, access object storage, or inspect telemetry.
  - Default MinIO and Postgres credentials make accidental LAN exposure materially worse.
- Exploit scenario:
  - A user follows tunnel examples for Modal and exposes `8080` and `9000`.
  - An attacker who learns the tunnel URL posts to orchestrator mutation endpoints, queues jobs, marks jobs complete or failed, and uses default object-store credentials to read or overwrite local artifacts.
- Recommended fix:
  - Bind local services to loopback by default.
  - Require explicit opt-in and authentication for any non-loopback or tunnel mode.
  - Rotate example/default storage credentials into generated local secrets for real runs.
- Safe implementation plan:
  1. Change the orchestrator default address from all interfaces to `127.0.0.1:8080`.
  2. Add explicit environment/config controls such as `MODEL_EXPRESS_ORCHESTRATOR_ADDR` and `MODEL_EXPRESS_ALLOW_LAN=true`; refuse `0.0.0.0` or non-loopback binds unless the allow flag and an API token are present.
  3. Update `infra/compose.yaml` ports to loopback form, for example `127.0.0.1:5432:5432`, `127.0.0.1:5000:5000`, `127.0.0.1:9000:9000`, and `127.0.0.1:9001:9001`.
  4. Generate a per-install local API token and require it on mutating orchestrator endpoints, worker endpoints, and any tunnel-facing endpoints.
  5. Keep `.env.example` placeholders, but document that real tunnel mode requires generated secrets and token-protected callbacks.
- Verification steps:
  - Static check: `rg -n "addr := \":8080\"|0\\.0\\.0\\.0|5432:5432|5000:5000|9000:9000|9001:9001" services infra configs .env.example`
  - Local bind check after startup: `netstat -ano | findstr ":8080"` should show `127.0.0.1:8080`, not `0.0.0.0:8080`.
  - LAN check from another device: `Test-NetConnection <developer-machine-lan-ip> -Port 8080` should fail unless LAN mode was explicitly enabled.
  - Auth check: `Invoke-WebRequest http://127.0.0.1:8080/projects -Method POST -Body "{}" -ContentType "application/json"` should return `401` or `403` without the token.
  - Modal/tunnel check: repeat the real Modal callback path with the configured token and confirm it succeeds.
- Regression risk:
  - Medium. Modal and external worker flows may currently depend on unauthenticated reachability. Mitigate by introducing a documented tunnel mode instead of silently breaking remote callbacks.

### Finding: Worker Job and Callback Endpoints Trust Unauthenticated Callers

- Severity: High
- Location: `services/orchestrator/internal/api/router.go:23-107`, `services/orchestrator/internal/api/handlers.go:1101-1117`, `services/orchestrator/internal/api/handlers.go:1143-1363`, `services/orchestrator/internal/api/handlers.go:4184-4481`, `services/orchestrator/internal/api/handlers.go:4543-4555`, `services/orchestrator/internal/store/postgres.go:1150-1169`, `services/worker/worker/orchestrator_client.py:93-117`, `services/worker/worker/jobs.py:32-55`
- Category: Missing auth assumptions for local-only APIs, LLM/tool/action execution without validation, overly trusted agent/worker outputs
- What is wrong:
  - `POST /projects/:id/jobs` accepts a template and config from API callers and persists the job.
  - Worker callback endpoints accept metric, summary, evaluation, export, demo, complete, and fail callbacks without an auth token.
  - `ignoreStaleJobCallback` ignores mismatched `training_attempt_id`, but missing attempt IDs are accepted.
  - The worker executes known job templates, including export and demo jobs, based on queued job config.
- Why it matters:
  - If the orchestrator is reachable outside the trusted desktop app, an attacker can enqueue expensive or sensitive jobs, poison metrics, mark jobs as successful, or hide failures.
  - Missing attempt binding makes it easier for stale or spoofed callbacks to mutate current job state.
- Exploit scenario:
  - A malicious local process or tunnel client posts a job with an allowed template and crafted config.
  - The worker polls, runs the job, then attacker-controlled callbacks mark it complete with fake metrics or artifacts before the real worker finishes.
- Recommended fix:
  - Treat worker/job mutation endpoints as privileged internal APIs.
  - Require authentication and strict attempt identity on all worker callbacks.
  - Validate job templates and configs before enqueueing.
- Safe implementation plan:
  1. Add an internal bearer token or HMAC header for worker-to-orchestrator callbacks. Pass it to local workers and Modal workers through environment or secret injection.
  2. Require `training_attempt_id` or a job attempt nonce for every callback that mutates job/run state. Reject missing IDs with `400` and mismatches with `409`.
  3. Move job template/config validation into a central allowlist. Each template should have a minimal typed schema and reject unknown fields that imply file paths, artifact URIs, or execution mode changes.
  4. Keep UI-created job behavior compatible by having the UI/orchestrator create valid job configs through existing flows, not by exposing a broad arbitrary-job endpoint.
  5. Add tests for unauthenticated, wrong-token, missing-attempt, stale-attempt, unsupported-template, and valid-worker callback cases.
- Verification steps:
  - Static check: `rg -n "POST|completeJob|failJob|reportMetric|training_attempt_id|CreateJob" services/orchestrator/internal/api services/orchestrator/internal/store services/worker/worker`
  - Auth test: `Invoke-WebRequest http://127.0.0.1:8080/jobs/<job-id>/complete -Method POST -Body "{}" -ContentType "application/json"` should return `401` or `403` without the worker token.
  - Attempt test: a callback with no `training_attempt_id` should return `400`; a callback with a stale ID should return `409`; the active attempt with a valid token should succeed.
  - Template test: `POST /projects/<id>/jobs` with an unsupported template should return `400`.
  - Automated test command after implementation: `go test ./...`
- Regression risk:
  - Medium. Existing worker tests and Modal callback code will need token plumbing. Keep the token optional only in explicitly marked development test harnesses, not in normal app runtime.

### Finding: Electron IPC Can Proxy Network Requests and Read or Upload Renderer-Controlled Paths

- Severity: High
- Location: `apps/mission-control/electron/preload.cjs:3-12`, `apps/mission-control/electron/main.cjs:71-110`, `apps/mission-control/electron/main.cjs:150-164`, `apps/mission-control/electron/main.cjs:202-219`, `apps/mission-control/electron/main.cjs:258-353`, `apps/mission-control/electron/main.cjs:820-904`, `apps/mission-control/src/App.tsx:76`, `apps/mission-control/src/App.tsx:1199-1264`, `apps/mission-control/src/App.tsx:1846-1887`, `apps/mission-control/src/App.tsx:2175-2194`, `services/orchestrator/internal/api/handlers.go:1609-1679`
- Category: Electron security issues, unsafe IPC handlers, unvalidated renderer-to-main messages, unsanitized user-controlled paths, local network exposure, exported model/artifact leakage
- What is wrong:
  - The preload exposes broad APIs for orchestrator requests, dataset folder uploads, artifact model loading, and worker control.
  - `orchestrator:request` accepts renderer-provided `baseUrl`, path, method, and body, making the Electron main process a network proxy.
  - Dataset upload and preflight accept a renderer-provided `datasetPath` and S3-style endpoint/credentials.
  - Artifact model loading accepts local paths, `file://` URIs, relative paths, and external data file references without a strong app-owned root check.
  - Backend export creation can mark a user-provided artifact URI as ready, which the UI can later hand to Electron for local loading.
- Why it matters:
  - Electron main has filesystem and network privileges that the renderer should not be able to direct freely.
  - A compromised renderer, malicious dependency, or unsafe backend artifact URI could read local files, upload local folders to an arbitrary endpoint, or make requests to local network services from the main process.
  - `contextIsolation` and `nodeIntegration: false` are good, but they do not protect against an overly broad preload bridge.
- Exploit scenario:
  - An attacker gains renderer script execution through a frontend bug or compromised dependency.
  - The script calls `window.missionControl.uploadDatasetFolder` on a sensitive local directory with an attacker-controlled endpoint, or calls `loadModelArtifact` with a local path disguised as an artifact.
- Recommended fix:
  - Convert Electron IPC from broad pass-through methods into narrow validated commands.
  - Track user-selected filesystem paths in the main process and only operate on those selections.
  - Restrict network targets to loopback by default, with explicit allowlisting for documented tunnel/provider endpoints.
  - Restrict artifact loading to app-owned export/artifact roots or app-owned object-store artifacts with provenance.
- Safe implementation plan:
  1. Add schema validation for every IPC request in `main.cjs`; reject unknown fields, unsupported methods, absolute paths where not expected, and non-HTTP(S) URLs.
  2. Replace renderer-provided `datasetPath` with a main-process selection token. The renderer can request a native picker, receive an opaque token/display name, and pass only that token to preflight/upload.
  3. Restrict dataset upload endpoints to local configured MinIO by default. Require an explicit saved allowlist entry before sending files to any non-loopback endpoint.
  4. Restrict `loadModelArtifact` to canonical paths under known export/artifact directories, or to object-store URIs that resolve through the orchestrator with an app-issued artifact ID.
  5. Validate the Engine URL in the main process, not only in React state. Default to `http://127.0.0.1:8080`; require explicit confirmation and token support for non-loopback URLs.
  6. Add negative IPC tests that attempt to read `C:\Windows\win.ini`, upload a parent home directory, use `file://` artifact URIs, and proxy to a non-allowlisted host.
- Verification steps:
  - Static check: `rg -n "ipcMain.handle|baseUrl|datasetPath|artifactUri|file://|readFileSync|createReadStream" apps/mission-control/electron apps/mission-control/src services/orchestrator/internal/api`
  - Manual renderer console test: `window.missionControl.loadModelArtifact({ artifactUri: "C:\\Windows\\win.ini" })` should reject with a validation error.
  - Manual upload test: attempting to upload `C:\Users` or send to `http://example.com:9000` should reject unless the path was selected and endpoint explicitly allowlisted.
  - Manual network proxy test: setting Engine URL to a non-loopback host should require explicit allowlisting and fail without auth.
  - Automated test command after adding Electron tests: `npm test -- --runInBand` or the repo's chosen Electron test command.
- Regression risk:
  - Medium to High. Dataset import and exported model preview rely on these IPC paths. Implement validation behind small helper functions and keep the normal picker-driven happy path unchanged.

### Finding: Artifact URI and PyTorch Checkpoint Loading Trust Unproven Sources

- Severity: High when API or job configs are externally reachable; Medium in a strictly local trusted-only workflow
- Location: `services/orchestrator/internal/api/handlers.go:1609-1679`, `services/worker/worker/champion_jobs.py:313-336`, `services/worker/worker/champion_jobs.py:402-420`, `services/worker/worker/champion_jobs.py:494-533`, `services/worker/worker/exporting/inference.py:239-255`, `services/worker/worker/exporting/artifacts.py:87-163`
- Category: Python worker execution risks, exported model/artifact leakage, unsafe deserialization, unsanitized user-controlled paths
- What is wrong:
  - Backend export creation accepts a request artifact URI and can mark the export ready without proving it was produced by the current project or worker.
  - Worker export/demo paths accept local file paths, `file://` URIs, S3 URIs, and config-provided checkpoint paths with mostly extension-based checks.
  - PyTorch checkpoint loading uses `torch.load` on `.pt`/`.pth` sources, including a path with `weights_only=False`.
  - Artifact copying can copy a source file into an export directory if the config points at a matching file.
- Why it matters:
  - PyTorch native checkpoints are pickle-based and can execute code when loaded if the file is malicious.
  - A crafted job config or export request can point the worker or Electron app at local files that were not produced by Model Express.
  - This risk is much higher when combined with unauthenticated job creation or broad Electron artifact loading.
- Exploit scenario:
  - An attacker uploads or points to a malicious `.pt` checkpoint, then triggers export or demo inference.
  - The worker loads it with `torch.load`, executing code under the user's local account or Modal worker context.
- Recommended fix:
  - Only load artifacts that have app-owned provenance, a project/run association, and an expected manifest.
  - Avoid loading untrusted PyTorch pickle checkpoints for V1; prefer ONNX or safetensors-style formats for external artifacts.
  - Use safer PyTorch loading modes where native checkpoints remain necessary.
- Safe implementation plan:
  1. Change export creation so a caller cannot mark an arbitrary `artifact_uri` as ready. Require an existing app artifact ID, project ID, run ID, and export manifest generated by the worker.
  2. Add an artifact provenance table or manifest containing project ID, run ID, artifact type, model format, expected files, size, and checksum.
  3. Reject local absolute paths and `file://` URIs in job configs unless they canonicalize under a known worker cache/export root and match the manifest.
  4. For PyTorch, try `torch.load(..., weights_only=True, map_location="cpu")` first and reject files requiring unsafe unpickling unless they were generated by the same trusted worker flow.
  5. Add a format allowlist for V1 previews and demos: ONNX for renderer inference, app-generated PyTorch checkpoints only for internal conversion.
  6. Log provenance validation failures without printing full local paths or secrets.
- Verification steps:
  - Static check: `rg -n "torch.load|weights_only|artifact_uri|source_artifact|checkpoint_path|file://" services/worker services/orchestrator`
  - Negative API test: creating a champion export with `artifact_uri` pointing to an arbitrary local `.onnx` path should return `400`.
  - Negative worker test: a job config with `checkpoint_path` outside the worker artifact/cache root should be rejected before any file read.
  - Deserialization test: a malicious pickle checkpoint fixture should be rejected and should not create a marker file or execute side effects.
  - Happy-path test: a normal app-generated checkpoint still exports to ONNX and can be previewed.
- Regression risk:
  - Medium. Some current convenience paths may depend on direct local artifact URIs. Preserve app-generated paths by migrating them to manifest-backed artifact IDs.

### Finding: Dataset ZIP Extraction and Upload Caps Are Too Trusting

- Severity: Medium
- Location: `services/worker/worker/datasets/cache.py:45-74`, `services/worker/worker/datasets/cache.py:181-188`, `services/worker/worker/datasets/shards.py:362-393`, `apps/mission-control/electron/zip-stream.cjs:94-156`, `apps/mission-control/electron/main.cjs:840-848`
- Category: Path traversal risks in file/artifact handling, unsafe file overwrite behavior, unsanitized user-controlled paths, local disk exhaustion
- What is wrong:
  - Dataset cache extraction uses archive `extractall` paths without an explicit shared safe-extraction guard.
  - Shard tar extraction has stronger member validation, but ZIP extraction does not appear to use the same explicit canonical-path pattern.
  - Dataset upload preflight has warning thresholds, but hard file-count and byte caps default to disabled unless configured.
- Why it matters:
  - Users may import third-party datasets. A malicious archive can attempt path traversal, unexpected overwrite, or disk exhaustion.
  - Even if Python's ZIP handling normalizes some paths, relying on library behavior is weaker than an explicit app policy with file count, total uncompressed size, compression ratio, and canonical destination checks.
- Exploit scenario:
  - A user imports a ZIP dataset containing `../` members, absolute paths, or extremely large compressed content.
  - The worker extracts files outside the intended cache or fills local disk before the run fails.
- Recommended fix:
  - Implement one safe ZIP extraction helper and use it everywhere.
  - Add default hard upload and extraction limits appropriate for V1 local machines.
- Safe implementation plan:
  1. Create `safe_extract_zip(archive, destination, limits)` in the worker dataset package.
  2. For each member, reject absolute paths, drive-qualified paths, `..` traversal, symlinks, directories that escape the destination, duplicate normalized names, and unsupported metadata.
  3. Enforce hard defaults such as maximum file count, maximum total uncompressed bytes, maximum single-file bytes, and maximum compression ratio. Make them configurable with safe defaults.
  4. Use the same canonical child-path helper style already present in shard extraction.
  5. In Electron preflight, set nonzero hard defaults for file count and bytes; keep warnings separate from hard stops.
- Verification steps:
  - Static check: `rg -n "extractall|maxFileCount|maxBytes|safe_extract|ZipFile" services/worker apps/mission-control/electron`
  - Add tests with ZIP entries named `../escape.txt`, `/absolute.txt`, `C:\absolute.txt`, duplicate normalized names, and a high-ratio compressed file. All should fail before extraction.
  - Add an import test for a normal small dataset ZIP; it should still materialize and profile.
  - Manual disk-cap test: try an import larger than the configured cap and confirm the app displays a controlled validation error.
- Regression risk:
  - Low to Medium. Some unusual datasets may need clearer errors or documented cap overrides, but normal image/tabular dataset imports should continue working.

### Finding: Raw Agent Invocation Traces Are Stored and Returned by Normal APIs

- Severity: Medium; High if the orchestrator is exposed outside loopback
- Location: `services/orchestrator/internal/memory/model.go:44-64`, `services/orchestrator/internal/api/handlers.go:3948-3990`, `services/orchestrator/internal/api/handlers.go:5758-5788`, `services/orchestrator/internal/llm/experiment_planner_llm.go:811-960`, `services/worker/worker/datasets/visual_analysis/agent.py:96-224`
- Category: LLM prompt injection risks, overly trusted agent outputs, logging sensitive data, exported model/artifact leakage
- What is wrong:
  - Agent invocation records include input messages, input context, raw model output, and parsed output.
  - Regular project and telemetry endpoints can return invocation records broadly.
  - Existing agent prompt/output validation is useful, but raw traces may still contain project goals, dataset summaries, storage references, local paths, or accidental secrets if upstream sanitization misses something.
- Why it matters:
  - Raw LLM traces are valuable for debugging but should be treated as sensitive diagnostic data.
  - If the local API is exposed, these endpoints become a project data and prompt leakage surface.
  - Even local users may accidentally export or share telemetry containing raw prompts and outputs.
- Exploit scenario:
  - A tunnel or local process calls the telemetry endpoint and retrieves raw planner prompts and outputs for a project, including dataset-derived context and artifact references.
- Recommended fix:
  - Make raw agent traces an explicit debug-only capability.
  - Return redacted summaries by default.
- Safe implementation plan:
  1. Split agent invocation APIs into a normal summary endpoint and a raw debug endpoint.
  2. Default the summary endpoint to fields such as agent name, status, timing, token usage, validation result, and short redacted error summary.
  3. Gate raw input/output fields behind both authentication and an explicit dev flag such as `MODEL_EXPRESS_ENABLE_RAW_AGENT_AUDIT=true`.
  4. Reuse existing diagnostics redaction helpers for keys and text fragments that look like tokens, credentials, signed URLs, or local secret paths.
  5. Add UI copy or export labels that make raw trace export an intentional debug action.
- Verification steps:
  - Static check: `rg -n "input_messages|input_context|raw_output|agent_invocations|telemetry" services/orchestrator`
  - API check: `GET /projects/<id>/agent-invocations` should not include `input_messages`, `input_context`, or `raw_output` by default.
  - Debug gate check: the same raw fields should appear only with auth and `MODEL_EXPRESS_ENABLE_RAW_AGENT_AUDIT=true`.
  - Redaction test: create an invocation containing fake `sk-test` and `AWS_SECRET_ACCESS_KEY` values; default responses must redact them.
- Regression risk:
  - Low to Medium. Debugging gets slightly less convenient, but the raw path remains available behind an intentional flag.

### Finding: JSON Request Body Size Limits Are Not Centralized

- Severity: Medium if exposed over LAN/tunnels; Low in a trusted loopback-only app
- Location: `services/orchestrator/internal/api/handlers.go:14506-14512`, `services/orchestrator/internal/api/router.go:23-107`
- Category: Unsafe JSON parsing assumptions, local network exposure
- What is wrong:
  - `bindJSON` calls JSON binding without a central request size limit.
  - Many API routes accept JSON payloads before route-specific size checks can protect memory and CPU.
- Why it matters:
  - A reachable client can send very large JSON bodies to consume memory or slow down the local control plane.
  - This is not a primary V1 issue for loopback-only use, but it becomes practical if tunnels or LAN binding remain possible.
- Exploit scenario:
  - A local or tunnel client repeatedly posts huge JSON bodies to metadata, settings, callback, or job endpoints and makes the app unresponsive.
- Recommended fix:
  - Add a centralized request body limit middleware with route-specific overrides for endpoints that legitimately need larger payloads.
- Safe implementation plan:
  1. Wrap request bodies with `http.MaxBytesReader` before binding JSON.
  2. Set a conservative default JSON cap, for example 1 MB or 2 MB.
  3. Add explicit larger limits only for known endpoints that need them, with comments and tests.
  4. Return `413 Payload Too Large` with a clear error body.
- Verification steps:
  - Static check: `rg -n "ShouldBindJSON|MaxBytesReader|bindJSON" services/orchestrator/internal/api`
  - Negative test: send a JSON body larger than the default cap and confirm `413`.
  - Happy-path test: normal project creation, settings update, and worker callback requests still pass.
  - Automated test command after implementation: `go test ./...`
- Regression risk:
  - Low. The main risk is accidentally setting the cap below legitimate metadata payload sizes. Use route-specific overrides where needed.

### Finding: Release Builds Depend On Floating Packages and Images

- Severity: Medium for public V1 supply chain; Low for private local development
- Location: `apps/mission-control/package.json`, `services/worker/pyproject.toml`, `infra/compose.yaml`, `services/orchestrator/go.mod`
- Category: Dependency vulnerabilities or risky packages, dangerous dev-only settings leaking into production
- What is wrong:
  - Several Node dependencies use `latest` or broad ranges.
  - Python worker dependencies are not pinned by a committed lock file in the reviewed tree.
  - Docker Compose uses mutable image tags such as `latest`.
  - Go dependencies are pinned in `go.mod`, which is the strongest of the dependency areas reviewed.
- Why it matters:
  - V1 builds should be reproducible. Floating packages and images can change behavior or pull new vulnerabilities without a code change.
  - This is especially relevant for Electron and ML/worker dependencies, which have large transitive dependency surfaces.
- Exploit scenario:
  - A fresh V1 install resolves a newer Electron, MLflow, MinIO, or Python package version with a regression or vulnerability that was not tested in the release candidate.
- Recommended fix:
  - Pin release dependencies and add lightweight audit checks to the release process.
- Safe implementation plan:
  1. Replace `latest` with tested exact or compatible pinned versions in `apps/mission-control/package.json`.
  2. Commit the package lock already used by the app and verify clean install from lock.
  3. Add a Python lock strategy, such as `uv.lock`, `requirements-lock.txt`, or another repo-standard lock file generated from `pyproject.toml`.
  4. Pin Docker images to explicit versions, and consider digests for release builds.
  5. Add release checks for `npm audit --audit-level=high`, `go test ./...`, and a Python dependency audit if the team standardizes on `pip-audit` or equivalent.
- Verification steps:
  - Static check: `rg -n "\"latest\"|:latest|\"\\^" apps/mission-control/package.json services/worker/pyproject.toml infra/compose.yaml`
  - Node audit: `cd apps/mission-control; npm audit --audit-level=high`
  - Go verification: `go test ./...`
  - Python audit after choosing a tool: `pip-audit -r requirements-lock.txt` or the equivalent lock-file command.
- Regression risk:
  - Low to Medium. Pinning can reveal dependency conflicts. Do it after P0 security fixes but before the release candidate is cut.

### Finding: Sensitive Tunnel and Environment-Derived Values Can Still Reach Console Logs

- Severity: Low; Medium if users share logs while using public tunnels
- Location: `apps/mission-control/electron/main.cjs:450-578`, `services/worker/worker/diagnostics.py:17-37`, `services/orchestrator/internal/diagnostics/diagnostics.go:22-24`
- Category: Secrets/API keys accidentally logged, logging sensitive data, unsafe `.env` handling
- What is wrong:
  - The repo has useful redaction helpers, but some Electron worker-start logs still print Modal tunnel URL values and environment-derived settings to the console.
  - `.env.local` is ignored, which is good, but any logging of env-derived tunnel URLs can still leak operational access details in copied logs or screenshots.
- Why it matters:
  - Tunnel URLs are not credentials by themselves, but in this app they can become access paths to unauthenticated local APIs or object storage.
  - Once auth is added, logs should still avoid printing full tokens, signed URLs, object-store secrets, or public tunnel hostnames.
- Exploit scenario:
  - A user posts a debug log while troubleshooting Modal. The log includes a live tunnel URL that reaches their local orchestrator or MinIO during the session.
- Recommended fix:
  - Route all environment and tunnel logging through a single redaction helper.
  - Log presence, mode, and host class rather than full sensitive values.
- Safe implementation plan:
  1. Replace direct console logs for Modal URLs and env-derived values with redacted diagnostic entries.
  2. Treat keys containing `TOKEN`, `SECRET`, `PASSWORD`, `KEY`, `URL`, `ENDPOINT`, `AUTH`, and `COOKIE` as sensitive by default unless explicitly allowlisted.
  3. Add tests that fake tunnel URLs and secrets are redacted in Electron, Go, and Python diagnostic paths.
  4. Keep enough information for support, such as `configured=true`, `scheme=https`, and `host_kind=cloudflare_tunnel`, without full host/path/token.
- Verification steps:
  - Static check: `rg -n "console\\.log|print\\(|log\\.|MODAL_.*URL|SECRET|TOKEN|PASSWORD" apps/mission-control services`
  - Manual run: start the worker with fake tunnel env vars and confirm logs do not contain the full tunnel hostname or fake secret.
  - Unit test: diagnostics should redact fake values like `sk-test-value`, `model_express_password`, and `https://example.trycloudflare.com`.
- Regression risk:
  - Low. This should only change diagnostic text, not runtime behavior.

## V1 Security Fix Plan

- P0: Must fix before release
  - Bind orchestrator and Docker services to loopback by default.
  - Add authenticated tunnel/LAN mode for orchestrator and worker callbacks.
  - Require worker callback attempt IDs or nonces and reject missing/stale callbacks.
  - Validate job template/config creation before enqueueing worker jobs.
  - Harden Electron IPC for renderer-controlled paths, artifact URIs, upload endpoints, and Engine URL.
  - Stop accepting arbitrary ready artifact URIs for champion exports; require app-owned artifact provenance.
  - Prevent unsafe PyTorch checkpoint loading for untrusted or config-provided artifacts.

- P1: Should fix soon
  - Implement safe ZIP extraction and default hard dataset upload/extraction caps.
  - Redact or gate raw agent invocation traces behind auth and a debug flag.
  - Add central JSON body size limits with route-specific overrides.
  - Pin Node, Python, and Docker release dependencies and add audit checks.
  - Add tests for path validation, artifact provenance, worker auth, and callback attempt matching.

- P2: Can defer
  - Improve support diagnostics to show redacted tunnel/provider status instead of full values.
  - Add optional local firewall or first-run checks that warn when services are reachable from LAN.
  - Add a formal security checklist to the release template.
  - Add periodic dependency scanning once the release lock files are stable.

## V1 Automatic Modal Connectivity Design

The current development workflow uses manual `cloudflared` tunnels so Modal can reach the local orchestrator and S3-compatible object storage. V1 should make this an explicit "remote training session" flow, not a permanent public exposure of local services.

- Recommended default:
  - Keep the orchestrator, MinIO, MLflow, and Postgres bound to `127.0.0.1`.
  - When the user starts a Modal run, create a short-lived remote training session with a generated `run_id`, `training_attempt_id`, callback token, and storage access scope.
  - Start required connectivity automatically from the Electron main process or orchestrator supervisor.
  - Pass only the session URLs and short-lived credentials/tokens to Modal.
  - Stop the tunnels and revoke session credentials when the run finishes, fails, is cancelled, or the app exits.

- Safer architecture:
  - Prefer tunneling only the orchestrator callback API, not the full local API surface.
  - Avoid exposing raw MinIO credentials to Modal when possible. Use presigned object URLs or a scoped per-run object credential limited to one bucket/prefix.
  - If MinIO must be tunnel-accessible, expose only the MinIO API port, never the console, and require per-run scoped credentials or presigned URLs.
  - Do not expose Postgres or MLflow through tunnels.

- Tunnel manager implementation:
  1. Add a `RemoteTrainingSession` record in the orchestrator with fields such as `id`, `project_id`, `training_attempt_id`, `status`, `callback_token_hash`, `orchestrator_public_url`, `storage_public_url`, `storage_prefix`, `created_at`, `expires_at`, and `closed_at`.
  2. Add a tunnel manager in Electron main or a small local supervisor that can start, monitor, and stop `cloudflared` child processes.
  3. Parse the tunnel URLs from process output, but never print full public URLs to normal logs.
  4. Health-check the public callback endpoint before submitting the Modal job.
  5. Submit the Modal job with `MODEL_EXPRESS_CALLBACK_URL`, `MODEL_EXPRESS_CALLBACK_TOKEN`, `MODEL_EXPRESS_TRAINING_ATTEMPT_ID`, and storage access information.
  6. Require Modal callbacks to include the token and attempt ID. Reject missing, expired, stale, or mismatched callbacks.
  7. Close the session and stop tunnels automatically on completion, cancellation, app shutdown, or timeout.

- Storage options:
  - Best V1 option: use a managed cloud bucket such as S3 or R2 with presigned URLs or scoped credentials. This avoids exposing the user's local MinIO over the internet.
  - Acceptable local-first fallback: automatically tunnel local MinIO only for the duration of one Modal run, with a per-run bucket/prefix and presigned URLs or scoped credentials.
  - Avoid: long-lived tunnels to local MinIO with reusable root credentials from `.env`.

- Security checks for the automatic flow:
  - Modal callback without token returns `401`.
  - Modal callback with the wrong `training_attempt_id` returns `409`.
  - Expired session callback returns `401` or `410`.
  - Tunnel is not started until the user begins a Modal run.
  - Tunnel is stopped after run completion/cancellation.
  - MinIO console, Postgres, and MLflow are not tunnel-routable.
  - Logs show only redacted tunnel status, for example `configured=true`, `host_kind=cloudflare_tunnel`, and `expires_at`, not the full URL or token.

- Verification steps:
  - Unit test: session token hashing, expiry, stale attempt rejection, and callback auth.
  - Integration test: start a fake tunnel process, parse a fake URL, pass it into a fake Modal submission, and stop the process on run completion.
  - Manual test: start a Modal run from a clean app install with no tunnel env vars. The app should create connectivity automatically, complete the run, then close the tunnel.
  - Negative manual test: while a session is active, call the public callback URL without the token and confirm it fails.
  - Static check: `rg -n "cloudflared|MODEL_EXPRESS_CALLBACK_TOKEN|training_attempt_id|RemoteTrainingSession|MODAL_ORCHESTRATOR_URL|MODAL_S3_ENDPOINT_URL" apps services`

## Security Guardrails to Add

- `env.example` cleanup:
  - Keep placeholders only. Do not include reusable real credentials beyond clearly documented local development defaults.
  - Add comments that tunnel mode requires generated tokens and should not expose default MinIO or orchestrator endpoints unauthenticated.
  - Verification: `git check-ignore -v .env .env.local .env.example` should show real env files ignored and `.env.example` allowed.

- Secret scanning:
  - Add a release check with a tool such as Gitleaks or TruffleHog.
  - Use redaction in CI output.
  - Verification: `gitleaks detect --source . --redact` or the selected equivalent command should pass before V1 tagging.

- Path validation utility:
  - Add shared helpers for canonical child-path checks in Electron, Go, and Python.
  - Use the helper for dataset uploads, archive extraction, artifact loading, export copying, and any file deletion/overwrite path.
  - Verification: tests should reject absolute paths, `..`, drive-qualified paths, `file://` URIs where disallowed, symlinks, and paths outside configured roots.

- Safe subprocess wrapper:
  - Keep subprocess calls array-based and shell-free.
  - Centralize executable path validation for Python worker startup.
  - Verification: `rg -n "shell=True|exec\\(|spawn\\(|subprocess" services apps` should show shell execution is absent or justified.

- Validated worker action schema:
  - Define allowed job templates and typed config schemas.
  - Reject unknown fields and untrusted file/artifact path fields.
  - Verification: API tests for unsupported templates and malicious config fields return `400`.

- Electron IPC validation:
  - Treat preload APIs as a security boundary.
  - Validate request shape, URL scheme/host, method, path, artifact URI, dataset selection token, and upload endpoint in main process.
  - Verification: renderer-console negative tests for arbitrary file read, arbitrary folder upload, and non-allowlisted network proxy all fail.

- Validated agent action schema:
  - Keep LLM tool outputs information-only unless a future feature explicitly needs mutations.
  - For any future agent action, require a typed schema, explicit allowlist, project/run ownership checks, and human-visible confirmation for filesystem/network effects.
  - Verification: malformed, unknown, or prompt-injected actions should be rejected by tests before reaching side-effect code.

- Production config checklist:
  - Confirm loopback bind by default.
  - Confirm auth token required for non-loopback and tunnel mode.
  - Confirm Docker ports are loopback-bound.
  - Confirm default credentials are rotated or local-only.
  - Confirm raw agent traces are gated.
  - Confirm dependency locks and audits are current.
  - Confirm Electron IPC negative tests pass.

- Release verification command set:
  - `git check-ignore -v .env .env.local .env.example`
  - `rg -n "\"latest\"|:latest|\"\\^" apps/mission-control/package.json services/worker/pyproject.toml infra/compose.yaml`
  - `rg -n "addr := \":8080\"|0\\.0\\.0\\.0|5432:5432|5000:5000|9000:9000|9001:9001" services infra configs`
  - `rg -n "torch.load|extractall|file://|readFileSync|datasetPath|artifactUri|ShouldBindJSON" services apps`
  - `cd apps/mission-control; npm audit --audit-level=high`
  - `go test ./...`
  - Python worker tests for safe extraction, artifact provenance, and checkpoint rejection once those tests are added.
