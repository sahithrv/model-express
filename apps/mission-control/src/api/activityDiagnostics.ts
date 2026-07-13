export const maxPendingActivityVisibilitySamples = 32;

export type ActivityVisibilityReasonCode =
  | "clock_skew"
  | "initial_catch_up"
  | "invalid_timestamp"
  | "live"
  | "reconnect_catch_up";

export type ActivityVisibilitySample = {
  createdAtMs: number;
  receivedAtMs: number;
  streamOpenedAtMs: number;
  catchUpReason: "initial_catch_up" | "reconnect_catch_up";
};

export type ActivityVisibilitySummary = {
  reason_code: ActivityVisibilityReasonCode;
  sample_count: number;
  latency_sample_count: number;
  invalid_sample_count: number;
  dropped_count: number;
  latency_total_ms: number;
  latency_min_ms: number;
  latency_max_ms: number;
  latency_average_ms: number;
  commit_delay_total_ms: number;
  commit_delay_average_ms: number;
};

export function appendActivityVisibilitySample(
  current: ActivityVisibilitySample[],
  sample: ActivityVisibilitySample,
): { samples: ActivityVisibilitySample[]; dropped: number } {
  const next = [...current, sample];
  const dropped = Math.max(0, next.length - maxPendingActivityVisibilitySamples);
  return {
    samples: dropped > 0 ? next.slice(dropped) : next,
    dropped,
  };
}

export function summarizeActivityVisibility(
  samples: ActivityVisibilitySample[],
  visibleAtMs: number,
  droppedCount = 0,
): ActivityVisibilitySummary[] {
  const bounded = samples.slice(-maxPendingActivityVisibilitySamples);
  const groups = new Map<ActivityVisibilityReasonCode, { latencies: number[]; commitDelays: number[]; invalid: number }>();

  for (const sample of bounded) {
    let reason: ActivityVisibilityReasonCode;
    let latency: number | null = null;
    if (!Number.isFinite(sample.createdAtMs)) {
      reason = "invalid_timestamp";
    } else if (sample.createdAtMs > visibleAtMs) {
      reason = "clock_skew";
    } else {
      reason = sample.createdAtMs < sample.streamOpenedAtMs ? sample.catchUpReason : "live";
      latency = nonNegativeInteger(visibleAtMs - sample.createdAtMs);
    }
    const group = groups.get(reason) ?? { latencies: [], commitDelays: [], invalid: 0 };
    if (latency === null) {
      group.invalid += 1;
    } else {
      group.latencies.push(latency);
    }
    group.commitDelays.push(nonNegativeInteger(visibleAtMs - sample.receivedAtMs));
    groups.set(reason, group);
  }

  return Array.from(groups.entries()).map(([reasonCode, group], index) => {
    const latencyTotal = sum(group.latencies);
    const commitDelayTotal = sum(group.commitDelays);
    return {
      reason_code: reasonCode,
      sample_count: group.latencies.length + group.invalid,
      latency_sample_count: group.latencies.length,
      invalid_sample_count: group.invalid,
      dropped_count: index === 0 ? nonNegativeInteger(droppedCount) : 0,
      latency_total_ms: latencyTotal,
      latency_min_ms: group.latencies.length > 0 ? Math.min(...group.latencies) : 0,
      latency_max_ms: group.latencies.length > 0 ? Math.max(...group.latencies) : 0,
      latency_average_ms: group.latencies.length > 0 ? Math.round(latencyTotal / group.latencies.length) : 0,
      commit_delay_total_ms: commitDelayTotal,
      commit_delay_average_ms: group.commitDelays.length > 0 ? Math.round(commitDelayTotal / group.commitDelays.length) : 0,
    };
  });
}

function sum(values: number[]) {
  return values.reduce((total, value) => total + value, 0);
}

function nonNegativeInteger(value: number) {
  return Number.isFinite(value) && value > 0 ? Math.round(value) : 0;
}
