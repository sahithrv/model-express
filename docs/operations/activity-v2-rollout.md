# Activity V2 Rollout And Retirement Policy

Status: Active
Owner: Mission Control and orchestrator maintainers
Effective: 2026-07-13

## Supported-version matrix

| Mission Control | Orchestrator | Worker | Expected live behavior | Support status |
| --- | --- | --- | --- | --- |
| Current | Current | Current | Compact snapshot plus cursor-ordered v2 events; targeted invalidations only while healthy | Primary |
| Previous | Current | Previous or current | Deprecated synthesized `/activity-stream` plus legacy reads | Compatibility window |
| Current | Previous | Previous or current | Automatic unsupported-endpoint fallback to bounded ten/thirty-second polling | Compatibility window |
| Current | Current | Previous | Backend lifecycle snapshots and metric compatibility; pre-epoch worker detail may be unavailable | Compatibility window |
| Previous | Previous | Current | Progress callback treats 404/405/501 as non-fatal and disables further progress reports for that job | Compatibility window |

The primary supported deployment is the current three-component set. One previous Mission Control/orchestrator compatibility pair remains supported for one tagged release. The current UI does not consume the synthesized activity SSE: it uses v2 when healthy and constructs a bounded fallback view from normal project reads when an older backend or a disconnect requires polling.

The orchestrator retains `GET /projects/:id/activity-stream` only for the previous UI during this window. Responses carry `Deprecation: true` and a warning naming `live-state` plus `events/stream/v2` as the replacement. The unversioned raw `GET /projects/:id/events/stream` has no supported consumer and is removed.

## Deterministic release gate

Run:

```bash
cd apps/mission-control
npm run rollout:activity
```

The fixed active scenario models six epoch boundaries in one minute. Its current budget is eleven requests: one initial snapshot, one probe, one stream connection, two compact snapshots, and six targeted metric reads. The PR 1 active baseline is 63 broad requests per minute, so the deterministic reduction is greater than 80%. Connected idle state performs only its initial snapshot, probe, and stream connection, with no broad refresh.

The release gate is green only when all of these are true:

1. Active request reduction is at least 80% against the 63-request PR 1 baseline.
2. Connected active and idle broad-GET counts are zero.
3. The orchestrator burst/reconnect soak delivers every project cursor exactly once, including bursts larger than the bounded catch-up page budget and global sequence gaps caused by other projects.
4. Mixed-version tests cover unsupported v2 snapshots/probes, deprecated previous-UI activity access, old-worker lifecycle/metric behavior, and non-fatal new-worker progress callbacks to old backends.
5. V2 transition reads call only `ListProjectExecutionEventsAfter`; no job, invocation, or decision history participates in v2 streaming.

## Runtime observation window

Observe at least one tagged compatibility release and seven consecutive days containing both active and idle sessions before removing the deprecated synthesized endpoint. Deliberate rollback drills and old-backend compatibility tests are excluded from adoption percentages but retained as separate reason codes.

Use bounded JSONL diagnostics only; no payload content or resource identity is needed:

| Signal | Diagnostic | Go threshold |
| --- | --- | --- |
| V2/fallback adoption | `mission_control_incremental_live` plus request reason/category counts | At least 95% of current-version sessions reach healthy v2; unsupported fallback is zero for current-version pairs |
| Request volume and bytes | `mission_control_request_metrics` | At least 80% active request reduction; zero connected broad GETs; no material response-byte regression |
| Reconnect/cursor safety | `mission_control_incremental_live` and `execution_event_stream_v2_connection` | Zero out-of-order or exhausted cursor recoveries; every reconnect probe resumes from the last applied cursor |
| Stream queries and bytes | `execution_event_stream_v2_batch` | Store/query calls equal bounded pages; response bytes and delivered counts remain bounded per batch |
| Snapshot queries and bytes | `project_live_state_read` | One store call, six-query PostgreSQL budget, response at or below 16 KiB |
| Event visibility | `activity_visibility_latency` with `source_code=execution_event_v2` | No sustained average regression from the PR 1 visibility baseline; investigate repeated maximum latency above one heartbeat interval |

No-go conditions are any missed transition, any current-version unsupported fallback, request reduction below 80%, snapshot response above 16 KiB, or repeated cursor-recovery failure. A no-go keeps the deprecated endpoint and rollout flags in place.

## Rollback and retirement decisions

During the observation window:

- `MODEL_EXPRESS_ACTIVITY_STREAM_V2_ENABLED=false` disables the backend v2 probe and stream without data repair. V2 is enabled by default.
- `MODEL_EXPRESS_MISSION_CONTROL_LIVE_V2_ROLLBACK=true` restores legacy polling. The enable and shadow flags remain available for compatibility diagnostics.
- `MODEL_EXPRESS_PROGRESS_REPORTING_ENABLED=false` disables worker progress observations without changing terminal callbacks or stored snapshots/events.

Activity PR 15 retires only paths whose replacement is already proven:

- The current UI's synthesized activity EventSource and activity-triggered broad refresh are removed.
- The unversioned raw execution-event stream is removed.
- Detailed Modal phase-event dual-write and `MODEL_EXPRESS_REMOTE_GPU_STAGE_TELEMETRY` are removed. Historical `stage_telemetry` fields remain readable, and the bounded resource summary is retained because run-detail consumers still use it.
- Automatic broad polling remains for current-version disconnect recovery and the previous backend.
- The deprecated synthesized endpoint and manual rollout flags remain until the observation window passes. Their deletion condition is the supported-version matrix dropping the previous UI/backend pair and every runtime threshold above remaining green for the full window.

Rollback changes flags only and never rewrites `execution_events` or `job_progress`.
