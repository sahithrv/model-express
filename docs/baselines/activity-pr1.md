# Activity PR 1 Baseline

This baseline captures the live path before any polling or refresh behavior is changed. The deterministic harness can be rerun with:

```bash
cd apps/mission-control
npm run baseline:activity
```

## Fixed request counts

| Scenario | UI refresh interval | Broad GETs/minute | Activity ticks/minute | Logical activity store calls/tick | PostgreSQL reads/tick |
| --- | ---: | ---: | ---: | ---: | ---: |
| Active work | 10 seconds | 63 | 12 | 4 | 8 |
| Idle | 30 seconds | 22 | 6 | 4 | 8 |

The UI count models the current fast refresh fan-out, a selected job, and the existing 15-second execution-event cache. It does not add hypothetical SSE-triggered refreshes or stream reconnects, so those appear separately at runtime under the `activity_event`, `stream_initial`, and `stream_reconnect` reason codes. Each v1 activity store method first verifies the project in PostgreSQL and then performs its list query, which is why four logical store calls currently mean eight SQL reads.

## Runtime measurements

The orchestrator writes one `activity_stream_tick` diagnostic per v1 tick with duration, synthesized/visible event counts, exact response-byte delta, logical store-call count, source-record count, reconnect/error counts, and controlled reason/scenario codes. Mission Control writes `mission_control_request_metrics` from the actual network boundary, after cache hits and in-flight request deduplication, with rolling request, broad-GET, error, byte, duration, endpoint-category, and reason-code totals. Native EventSource opens/reconnects report through a narrow scalar IPC so they are included even though they bypass the normal request bridge. Request aggregates are flushed at most once every five seconds to avoid synchronous per-response disk I/O; the rolling window uses documented one-second buckets.

Event-to-visible-state latency is measured after React commits the activity state. Pending samples are capped at 32 and emitted as scalar summaries under `activity_visibility_latency`; prompts, event messages, payloads, resource IDs, paths, and storage URIs are not recorded.

The runtime measurements live in the normal bounded, rotated JSONL diagnostics (`orchestrator.jsonl` and `mission-control.jsonl`). They are intended for before/after comparisons with identical active and idle workloads.
