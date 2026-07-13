import { resourcesForExecutionEvent, type LiveResource } from "./liveInvalidations";

export const ACTIVE_LEGACY_POLL_INTERVAL_MS = 10_000;
export const IDLE_LEGACY_POLL_INTERVAL_MS = 30_000;
export const UNSUPPORTED_LIVE_ENDPOINT_STATUSES = [404, 405, 501] as const;

export type LiveDataMode = "legacy" | "shadow" | "incremental" | "fallback" | "rollback";
export type LiveEndpointSupport = "unknown" | "supported" | "unsupported";
export type LiveStreamStatus = "idle" | "connecting" | "connected" | "disconnected" | "error";
export type LiveCursorStatus = "unknown" | "consistent" | "recovering" | "failed";

export type LiveDataFeatureFlags = {
  incremental_v2_enabled: boolean;
  shadow_mode: boolean;
  rollback_to_legacy: boolean;
};

export type IncrementalLiveHealth = {
  endpoint_support: LiveEndpointSupport;
  snapshot_ready: boolean;
  stream_status: LiveStreamStatus;
  cursor_status: LiveCursorStatus;
};

export type LivePollingReasonCode =
  | "feature_disabled"
  | "rollback_enabled"
  | "endpoint_unsupported"
  | "shadow_mode"
  | "snapshot_not_ready"
  | "stream_disconnected"
  | "cursor_inconsistent"
  | "cursor_recovery_failed"
  | "healthy_incremental";

export type LivePollingPolicy = {
  mode: LiveDataMode;
  reason_code: LivePollingReasonCode;
  start_incremental_layer: boolean;
  use_incremental_presentation: boolean;
  run_legacy_polling: boolean;
  broad_refresh_interval_ms: number | null;
  manual_refresh_enabled: true;
  detail_tabs_on_demand: true;
};

export type ResolveLivePollingPolicyInput = {
  flags: LiveDataFeatureFlags;
  health: IncrementalLiveHealth;
  has_open_work: boolean;
};

/**
 * Broad polling is disabled only after all four health gates are satisfied.
 * Startup, disconnect, cursor recovery, unsupported endpoints, and rollback
 * therefore retain the conservative legacy interval.
 */
export function resolveLivePollingPolicy(input: ResolveLivePollingPolicyInput): LivePollingPolicy {
  const { flags, health, has_open_work: hasOpenWork } = input;
  if (flags.rollback_to_legacy) {
    return legacyPolicy("rollback", "rollback_enabled", hasOpenWork, false);
  }
  if (!flags.incremental_v2_enabled) {
    return legacyPolicy("legacy", "feature_disabled", hasOpenWork, false);
  }
  if (health.endpoint_support === "unsupported") {
    return legacyPolicy("fallback", "endpoint_unsupported", hasOpenWork, false);
  }
  if (flags.shadow_mode) {
    return legacyPolicy("shadow", "shadow_mode", hasOpenWork, true);
  }
  if (health.cursor_status === "failed") {
    return legacyPolicy("fallback", "cursor_recovery_failed", hasOpenWork, true);
  }
  if (!health.snapshot_ready) {
    return legacyPolicy("fallback", "snapshot_not_ready", hasOpenWork, true);
  }
  if (health.stream_status !== "connected") {
    return legacyPolicy("fallback", "stream_disconnected", hasOpenWork, true);
  }
  if (health.cursor_status !== "consistent") {
    return legacyPolicy("fallback", "cursor_inconsistent", hasOpenWork, true);
  }
  return {
    mode: "incremental",
    reason_code: "healthy_incremental",
    start_incremental_layer: true,
    use_incremental_presentation: true,
    run_legacy_polling: false,
    broad_refresh_interval_ms: null,
    manual_refresh_enabled: true,
    detail_tabs_on_demand: true,
  };
}

export function isUnsupportedLiveEndpointStatus(status: number): boolean {
  return (UNSUPPORTED_LIVE_ENDPOINT_STATUSES as readonly number[]).includes(status);
}

function legacyPolicy(
  mode: Exclude<LiveDataMode, "incremental">,
  reasonCode: Exclude<LivePollingReasonCode, "healthy_incremental">,
  hasOpenWork: boolean,
  startIncrementalLayer: boolean,
): LivePollingPolicy {
  return {
    mode,
    reason_code: reasonCode,
    start_incremental_layer: startIncrementalLayer,
    use_incremental_presentation: false,
    run_legacy_polling: true,
    broad_refresh_interval_ms: hasOpenWork ? ACTIVE_LEGACY_POLL_INTERVAL_MS : IDLE_LEGACY_POLL_INTERVAL_MS,
    manual_refresh_enabled: true,
    detail_tabs_on_demand: true,
  };
}

export type LegacyPollingProfile = {
  active_interval_ms: number;
  idle_interval_ms: number;
  uncached_requests_per_tick: number;
  cached_request_ttls_ms: readonly number[];
};

/**
 * The current refreshLive path issues ten uncached GETs and one execution-event
 * GET cached for 15 seconds. Keeping the profile explicit makes request-volume
 * assertions deterministic and exposes future changes to the baseline.
 */
export const CURRENT_LEGACY_POLLING_PROFILE: LegacyPollingProfile = {
  active_interval_ms: ACTIVE_LEGACY_POLL_INTERVAL_MS,
  idle_interval_ms: IDLE_LEGACY_POLL_INTERVAL_MS,
  uncached_requests_per_tick: 10,
  cached_request_ttls_ms: [15_000],
};

export type LegacyPollingVolume = {
  polling_ticks: number;
  broad_requests: number;
  uncached_requests: number;
  cached_requests: number;
};

export function simulateLegacyPollingVolume(
  durationMs: number,
  hasOpenWork: boolean,
  profile: LegacyPollingProfile = CURRENT_LEGACY_POLLING_PROFILE,
): LegacyPollingVolume {
  const duration = nonNegativeSafeInteger(durationMs, "durationMs");
  validatePollingProfile(profile);
  const interval = hasOpenWork ? profile.active_interval_ms : profile.idle_interval_ms;
  const pollingTicks = Math.floor(duration / interval);
  const uncachedRequests = pollingTicks * profile.uncached_requests_per_tick;
  let cachedRequests = 0;
  const lastRequestedAt = profile.cached_request_ttls_ms.map(() => Number.NEGATIVE_INFINITY);
  for (let tick = 1; tick <= pollingTicks; tick += 1) {
    const at = tick * interval;
    profile.cached_request_ttls_ms.forEach((ttl, index) => {
      if (at - lastRequestedAt[index] >= ttl) {
        cachedRequests += 1;
        lastRequestedAt[index] = at;
      }
    });
  }
  return {
    polling_ticks: pollingTicks,
    broad_requests: uncachedRequests + cachedRequests,
    uncached_requests: uncachedRequests,
    cached_requests: cachedRequests,
  };
}

export type LiveEventBurst = {
  at_ms: number;
  event_types: readonly string[];
};

export type LiveRequestVolumeScenario = {
  duration_ms: number;
  has_open_work: boolean;
  policy: LivePollingPolicy;
  event_bursts?: readonly LiveEventBurst[];
  manual_refresh_requests?: number;
  detail_tab_requests?: number;
  /** Logical HTTP cost for one manual refresh; zero keeps comparisons focused on passive live behavior. */
  manual_refresh_request_cost?: number;
  compact_snapshot_interval_ms?: number;
  legacy_profile?: LegacyPollingProfile;
};

export type LiveRequestVolume = {
  broad_refreshes: number;
  broad_requests: number;
  live_state_requests: number;
  compact_snapshot_requests: number;
  stream_probes: number;
  stream_connections: number;
  targeted_requests: number;
  targeted_by_resource: Readonly<Partial<Record<LiveResource, number>>>;
  manual_refreshes: number;
  manual_requests: number;
  detail_tab_requests: number;
  total_requests: number;
};

/**
 * Pure request-count harness. Events at the same timestamp form one burst, and
 * each resource is fetched at most once per burst. It deliberately contains no
 * wall-clock waits or randomness.
 */
export function simulateLiveRequestVolume(scenario: LiveRequestVolumeScenario): LiveRequestVolume {
  const duration = nonNegativeSafeInteger(scenario.duration_ms, "duration_ms");
  const manualRefreshes = nonNegativeSafeInteger(scenario.manual_refresh_requests ?? 0, "manual_refresh_requests");
  const detailTabRequests = nonNegativeSafeInteger(scenario.detail_tab_requests ?? 0, "detail_tab_requests");
  const manualRequestCost = nonNegativeSafeInteger(scenario.manual_refresh_request_cost ?? 0, "manual_refresh_request_cost");
  const compactSnapshotInterval = positiveSafeInteger(
    scenario.compact_snapshot_interval_ms ?? 30_000,
    "compact_snapshot_interval_ms",
  );
  const legacy = scenario.policy.run_legacy_polling
    ? simulateLegacyPollingVolume(duration, scenario.has_open_work, scenario.legacy_profile)
    : { polling_ticks: 0, broad_requests: 0, uncached_requests: 0, cached_requests: 0 };
  const targetedByResource: Partial<Record<LiveResource, number>> = {};

  if (scenario.policy.use_incremental_presentation) {
    const bursts = coalesceBursts(scenario.event_bursts ?? [], duration);
    for (const eventTypes of bursts.values()) {
      const resources = new Set<LiveResource>();
      for (const eventType of eventTypes) {
        for (const resource of resourcesForExecutionEvent(eventType)) resources.add(resource);
      }
      for (const resource of resources) {
        targetedByResource[resource] = (targetedByResource[resource] ?? 0) + 1;
      }
    }
  }

  const targetedRequests = Object.values(targetedByResource).reduce((sum, count) => sum + (count ?? 0), 0);
  const compactSnapshotRequests =
    scenario.has_open_work && ["incremental", "shadow"].includes(scenario.policy.mode)
      ? Math.floor(duration / compactSnapshotInterval)
      : 0;
  const liveStateRequests = scenario.policy.start_incremental_layer ? 1 + compactSnapshotRequests : 0;
  const streamProbes = scenario.policy.start_incremental_layer ? 1 : 0;
  const streamConnections = scenario.policy.start_incremental_layer ? 1 : 0;
  const manualRequests = manualRefreshes * manualRequestCost;
  return {
    broad_refreshes: legacy.polling_ticks,
    broad_requests: legacy.broad_requests,
    live_state_requests: liveStateRequests,
    compact_snapshot_requests: compactSnapshotRequests,
    stream_probes: streamProbes,
    stream_connections: streamConnections,
    targeted_requests: targetedRequests,
    targeted_by_resource: targetedByResource,
    manual_refreshes: manualRefreshes,
    manual_requests: manualRequests,
    detail_tab_requests: detailTabRequests,
    total_requests:
      legacy.broad_requests +
      liveStateRequests +
      streamProbes +
      streamConnections +
      targetedRequests +
      manualRequests +
      detailTabRequests,
  };
}

export function requestReductionPercent(baselineRequests: number, currentRequests: number): number {
  const baseline = nonNegativeSafeInteger(baselineRequests, "baselineRequests");
  const current = nonNegativeSafeInteger(currentRequests, "currentRequests");
  // Diagnostics must remain finite even when no baseline sample is available.
  if (baseline === 0) return 0;
  return ((baseline - current) / baseline) * 100;
}

function coalesceBursts(bursts: readonly LiveEventBurst[], durationMs: number): Map<number, string[]> {
  const output = new Map<number, string[]>();
  for (const burst of bursts) {
    const at = nonNegativeSafeInteger(burst.at_ms, "event burst at_ms");
    if (at > durationMs) continue;
    const events = output.get(at) ?? [];
    for (const eventType of burst.event_types) {
      if (typeof eventType !== "string") throw new TypeError("event_types must contain strings.");
      events.push(eventType);
    }
    output.set(at, events);
  }
  return output;
}

function validatePollingProfile(profile: LegacyPollingProfile): void {
  const active = positiveSafeInteger(profile.active_interval_ms, "active_interval_ms");
  const idle = positiveSafeInteger(profile.idle_interval_ms, "idle_interval_ms");
  nonNegativeSafeInteger(profile.uncached_requests_per_tick, "uncached_requests_per_tick");
  profile.cached_request_ttls_ms.forEach((ttl, index) => positiveSafeInteger(ttl, `cached_request_ttls_ms[${index}]`));
  if (!active || !idle) throw new TypeError("Polling intervals must be positive.");
}

function positiveSafeInteger(value: number, name: string): number {
  const parsed = nonNegativeSafeInteger(value, name);
  if (parsed === 0) throw new TypeError(`${name} must be positive.`);
  return parsed;
}

function nonNegativeSafeInteger(value: number, name: string): number {
  if (!Number.isSafeInteger(value) || value < 0) throw new TypeError(`${name} must be a non-negative safe integer.`);
  return value;
}
