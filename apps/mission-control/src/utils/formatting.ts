export function formatBytes(value: number) {
  if (!value) return "-";
  const units = ["B", "KB", "MB", "GB"];
  let size = value;
  let index = 0;
  while (size >= 1024 && index < units.length - 1) {
    size /= 1024;
    index += 1;
  }
  return `${size.toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}

export function formatRelativeTime(value: string) {
  const timestamp = Date.parse(value);
  if (!Number.isFinite(timestamp)) return value;
  const elapsedSeconds = Math.max(0, Math.round((Date.now() - timestamp) / 1000));
  if (elapsedSeconds < 60) return `${elapsedSeconds}s ago`;
  const elapsedMinutes = Math.round(elapsedSeconds / 60);
  if (elapsedMinutes < 60) return `${elapsedMinutes}m ago`;
  const elapsedHours = Math.round(elapsedMinutes / 60);
  if (elapsedHours < 24) return `${elapsedHours}h ago`;
  return new Date(timestamp).toLocaleString();
}

export function formatTimestamp(value: string) {
  const timestamp = Date.parse(value);
  if (!Number.isFinite(timestamp)) return value;
  return new Date(timestamp).toLocaleString();
}

export function formatCompactNumber(value: number) {
  if (!Number.isFinite(value)) return "-";
  const abs = Math.abs(value);
  if (abs < 1000) {
    return String(Math.round(value));
  }
  return new Intl.NumberFormat("en-US", {
    notation: "compact",
    maximumFractionDigits: 1,
  }).format(value);
}

export function formatTelemetryTokenPair(exactTokens: number, approxTokens: number) {
  const hasExact = Number.isFinite(exactTokens) && exactTokens > 0;
  const hasApprox = Number.isFinite(approxTokens) && approxTokens > 0;
  if (hasExact && hasApprox) {
    return `${formatCompactNumber(exactTokens)} exact + ~${formatCompactNumber(approxTokens)} approx`;
  }
  if (hasExact) {
    return `${formatCompactNumber(exactTokens)} exact`;
  }
  if (hasApprox) {
    return `~${formatCompactNumber(approxTokens)}`;
  }
  return "0";
}

export function formatTopKScore(value: unknown) {
  const numeric = typeof value === "number" ? value : typeof value === "string" ? Number(value) : 0;
  if (!numeric || !Number.isFinite(numeric)) return "";
  return numeric <= 1 ? `${Math.round(numeric * 100)}%` : numeric.toFixed(3);
}

export function formatCurrency(value: number) {
  return `$${value.toFixed(value < 1 ? 4 : 2)}`;
}

export function formatSeconds(value: number) {
  if (!value) return "0s";
  if (value < 60) return `${Math.round(value)}s`;
  const minutes = Math.floor(value / 60);
  const seconds = Math.round(value % 60);
  return `${minutes}m ${seconds}s`;
}

export function formatMaybeMetric(value: number) {
  if (!value) return "-";
  return value.toFixed(3);
}

export function formatDecisionQualityCount(value: number | null) {
  return value === null || !Number.isFinite(value) ? "-" : String(value);
}

export function formatDecisionQualityMetric(value: number | null, signed: boolean) {
  if (value === null || !Number.isFinite(value)) return "-";
  const sign = signed && value > 0 ? "+" : "";
  return `${sign}${value.toFixed(3)}`;
}

export function formatPercent(value: number) {
  if (!Number.isFinite(value)) return "-";
  return `${Math.round(value * 100)}%`;
}

export function formatLossGap(value: number | null) {
  if (value === null || !Number.isFinite(value)) return "-";
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(3)}`;
}

export function formatSeedVariance(value: number | null, runCount: number) {
  if (value === null || runCount < 2 || !Number.isFinite(value)) return "-";
  return `${value.toFixed(5)} (${runCount})`;
}

export function formatMetricNumber(value: number | null) {
  if (value === null || !Number.isFinite(value)) return "-";
  return value.toFixed(3);
}

export function formatLatency(value: unknown) {
  const numeric = typeof value === "number" ? value : typeof value === "string" ? Number(value) : 0;
  if (!numeric || !Number.isFinite(numeric)) return "-";
  return `${numeric.toFixed(numeric < 10 ? 2 : 1)}ms`;
}

export function formatChartValue(value: number) {
  if (Math.abs(value) >= 10) return value.toFixed(2);
  return value.toFixed(4);
}

export function formatUnknownValue(value: unknown) {
  if (typeof value === "number") return formatMetricNumber(value);
  if (typeof value === "string") return value;
  if (typeof value === "boolean") return value ? "true" : "false";
  if (value === null || value === undefined) return "";
  return JSON.stringify(value);
}
