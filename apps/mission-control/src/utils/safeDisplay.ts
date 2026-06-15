export function shortValue(value: unknown): string {
  if (Array.isArray(value)) return value.map((item) => shortValue(item)).join(", ");
  if (value && typeof value === "object") return JSON.stringify(value);
  return String(value);
}

export function errorMessage(error: unknown) {
  return error instanceof Error ? error.message : String(error);
}

export function isUnsupportedEndpointError(error: unknown) {
  const message = errorMessage(error).toLowerCase();
  return (
    message.includes("404") ||
    message.includes("not found") ||
    message.includes("cannot get") ||
    message.includes("unexpected non-whitespace")
  );
}
