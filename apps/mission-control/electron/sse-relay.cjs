"use strict";

const { TextDecoder } = require("node:util");

const DEFAULT_LIMITS = Object.freeze({
  maxEventDataBytes: 64 * 1024,
  maxLineBytes: 64 * 1024,
  maxBufferedBytes: 128 * 1024,
  maxChunkBytes: 128 * 1024,
  maxDataLines: 1024,
  maxEventNameBytes: 128,
  maxIdBytes: 512,
});

const HARD_LIMITS = Object.freeze({
  maxEventDataBytes: 1024 * 1024,
  maxLineBytes: 1024 * 1024,
  maxBufferedBytes: 2 * 1024 * 1024,
  maxChunkBytes: 2 * 1024 * 1024,
  maxDataLines: 4096,
  maxEventNameBytes: 1024,
  maxIdBytes: 4096,
});

const INVALID_LINE_CONTROL = /[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]/;

class SSERelayError extends Error {
  constructor(code, message) {
    super(message);
    this.name = "SSERelayError";
    this.code = code;
  }
}

class IncrementalSSEParser {
  constructor(options = {}) {
    this.limits = normalizeLimits(options);
    this.decoder = new TextDecoder("utf-8", { fatal: true });
    this.buffer = "";
    this.dataLines = [];
    this.dataBytes = 0;
    this.eventName = "";
    this.currentLastEventId = "";
    this.started = false;
    this.closed = false;
  }

  get lastEventId() {
    return this.currentLastEventId;
  }

  push(chunk) {
    this.#assertOpen();
    const bytes = normalizeChunk(chunk);
    if (bytes.byteLength > this.limits.maxChunkBytes) {
      this.closed = true;
      throw new SSERelayError("chunk_too_large", "SSE chunk exceeds the configured byte limit.");
    }

    try {
      const decoded = this.decoder.decode(bytes, { stream: true });
      return this.#appendDecoded(decoded, false);
    } catch (error) {
      this.closed = true;
      if (error instanceof SSERelayError) throw error;
      throw new SSERelayError("invalid_utf8", "SSE stream contains invalid UTF-8.");
    }
  }

  finish() {
    this.#assertOpen();
    try {
      const decoded = this.decoder.decode();
      const events = this.#appendDecoded(decoded, true);
      if (this.dataLines.length > 0 || this.eventName !== "") {
        throw new SSERelayError("incomplete_event", "SSE stream ended before the current event was terminated.");
      }
      this.closed = true;
      return events;
    } catch (error) {
      this.closed = true;
      if (error instanceof SSERelayError) throw error;
      throw new SSERelayError("invalid_utf8", "SSE stream contains invalid UTF-8.");
    }
  }

  #assertOpen() {
    if (this.closed) {
      throw new SSERelayError("parser_closed", "SSE parser is already closed.");
    }
  }

  #appendDecoded(decoded, final) {
    let text = decoded;
    if (!this.started && text.length > 0) {
      this.started = true;
      if (text.charCodeAt(0) === 0xfeff) text = text.slice(1);
    }
    this.buffer += text;
    const events = this.#drainLines(final);
    if (Buffer.byteLength(this.buffer, "utf8") > this.limits.maxBufferedBytes) {
      throw new SSERelayError("buffer_too_large", "SSE parser buffer exceeds the configured byte limit.");
    }
    return events;
  }

  #drainLines(final) {
    const events = [];
    while (this.buffer.length > 0) {
      const delimiterIndex = this.buffer.search(/[\r\n]/);
      if (delimiterIndex < 0) {
        if (final) {
          const line = this.buffer;
          this.buffer = "";
          this.#processLine(line, events);
        }
        break;
      }

      const delimiter = this.buffer[delimiterIndex];
      if (!final && delimiter === "\r" && delimiterIndex === this.buffer.length - 1) {
        break;
      }
      let delimiterLength = 1;
      if (delimiter === "\r" && this.buffer[delimiterIndex + 1] === "\n") {
        delimiterLength = 2;
      }
      const line = this.buffer.slice(0, delimiterIndex);
      this.buffer = this.buffer.slice(delimiterIndex + delimiterLength);
      this.#processLine(line, events);
    }
    return events;
  }

  #processLine(line, events) {
    if (Buffer.byteLength(line, "utf8") > this.limits.maxLineBytes) {
      throw new SSERelayError("line_too_large", "SSE line exceeds the configured byte limit.");
    }
    if (INVALID_LINE_CONTROL.test(line)) {
      throw new SSERelayError("malformed_line", "SSE line contains an unsupported control character.");
    }
    if (line === "") {
      const event = this.#dispatchEvent();
      if (event) events.push(event);
      return;
    }
    if (line.startsWith(":")) return;

    const separatorIndex = line.indexOf(":");
    const field = separatorIndex < 0 ? line : line.slice(0, separatorIndex);
    let value = separatorIndex < 0 ? "" : line.slice(separatorIndex + 1);
    if (value.startsWith(" ")) value = value.slice(1);

    switch (field) {
      case "data":
        this.#appendData(value);
        break;
      case "event":
        if (Buffer.byteLength(value, "utf8") > this.limits.maxEventNameBytes) {
          throw new SSERelayError("event_name_too_large", "SSE event name exceeds the configured byte limit.");
        }
        this.eventName = value;
        break;
      case "id":
        if (Buffer.byteLength(value, "utf8") > this.limits.maxIdBytes) {
          throw new SSERelayError("id_too_large", "SSE event id exceeds the configured byte limit.");
        }
        this.currentLastEventId = value;
        break;
      default:
        // `retry` and extension fields do not carry application data.
        break;
    }
  }

  #appendData(value) {
    if (this.dataLines.length >= this.limits.maxDataLines) {
      throw new SSERelayError("too_many_data_lines", "SSE event contains too many data lines.");
    }
    const separatorBytes = this.dataLines.length > 0 ? 1 : 0;
    const nextBytes = this.dataBytes + separatorBytes + Buffer.byteLength(value, "utf8");
    if (nextBytes > this.limits.maxEventDataBytes) {
      throw new SSERelayError("event_data_too_large", "SSE event data exceeds the configured byte limit.");
    }
    this.dataLines.push(value);
    this.dataBytes = nextBytes;
  }

  #dispatchEvent() {
    if (this.dataLines.length === 0) {
      this.eventName = "";
      return null;
    }
    const event = {
      event: this.eventName || "message",
      lastEventId: this.currentLastEventId,
      data: this.dataLines.join("\n"),
    };
    this.dataLines = [];
    this.dataBytes = 0;
    this.eventName = "";
    return event;
  }
}

function createSSEParser(options = {}) {
  return new IncrementalSSEParser(options);
}

async function readSSEBody(responseOrBody, options = {}) {
  const { signal, onEvent, ...parserOptions } = options;
  if (typeof onEvent !== "function") {
    throw new TypeError("SSE response reader requires an onEvent callback.");
  }
  validateAbortSignal(signal);
  throwIfAborted(signal);

  const body = responseOrBody && responseOrBody.body ? responseOrBody.body : responseOrBody;
  if (!body || typeof body.getReader !== "function") {
    throw new SSERelayError("invalid_body", "SSE response does not expose a readable body.");
  }

  const parser = createSSEParser(parserOptions);
  const reader = body.getReader();
  let completed = false;
  let aborted = false;
  let eventCount = 0;
  const abortReader = () => {
    aborted = true;
    try {
      Promise.resolve(reader.cancel()).catch(() => {});
    } catch {
      // Synchronous cancellation errors are intentionally ignored and never logged.
    }
  };
  signal?.addEventListener("abort", abortReader, { once: true });

  try {
    while (true) {
      throwIfAborted(signal);
      const result = await reader.read();
      throwIfAborted(signal);
      if (!result || result.done) {
        completed = true;
        break;
      }
      const events = parser.push(result.value);
      for (const event of events) {
        throwIfAborted(signal);
        await waitForCallback(onEvent(event), signal);
        eventCount += 1;
      }
    }

    for (const event of parser.finish()) {
      throwIfAborted(signal);
      await waitForCallback(onEvent(event), signal);
      eventCount += 1;
    }
    return { eventCount, lastEventId: parser.lastEventId };
  } catch (error) {
    if (aborted || signal?.aborted || error?.name === "AbortError") {
      throw createAbortError();
    }
    throw error;
  } finally {
    signal?.removeEventListener("abort", abortReader);
    if (!completed) {
      try {
        await reader.cancel();
      } catch {
        // Cancellation errors are intentionally ignored and never logged.
      }
    }
    try {
      reader.releaseLock();
    } catch {
      // Some synthetic readers do not require or support lock release.
    }
  }
}

function normalizeChunk(chunk) {
  if (typeof chunk === "string") return Buffer.from(chunk, "utf8");
  if (chunk instanceof Uint8Array) return chunk;
  if (chunk instanceof ArrayBuffer) return new Uint8Array(chunk);
  if (ArrayBuffer.isView(chunk)) return new Uint8Array(chunk.buffer, chunk.byteOffset, chunk.byteLength);
  throw new TypeError("SSE parser chunks must be strings or byte arrays.");
}

function normalizeLimits(options) {
  const limits = {};
  for (const [name, fallback] of Object.entries(DEFAULT_LIMITS)) {
    const value = options[name] ?? fallback;
    if (!Number.isSafeInteger(value) || value < 1 || value > HARD_LIMITS[name]) {
      throw new TypeError(`Invalid bounded SSE parser option: ${name}.`);
    }
    limits[name] = value;
  }
  return Object.freeze(limits);
}

function validateAbortSignal(signal) {
  if (signal === undefined || signal === null) return;
  if (
    typeof signal.aborted !== "boolean" ||
    typeof signal.addEventListener !== "function" ||
    typeof signal.removeEventListener !== "function"
  ) {
    throw new TypeError("SSE response reader signal must be an AbortSignal.");
  }
}

function throwIfAborted(signal) {
  if (signal?.aborted) throw createAbortError();
}

function createAbortError() {
  const error = new Error("SSE relay aborted.");
  error.name = "AbortError";
  error.code = "ABORT_ERR";
  return error;
}

function waitForCallback(value, signal) {
  if (!signal) return Promise.resolve(value);
  throwIfAborted(signal);
  return new Promise((resolve, reject) => {
    let settled = false;
    const onAbort = () => {
      if (settled) return;
      settled = true;
      reject(createAbortError());
    };
    signal.addEventListener("abort", onAbort, { once: true });
    Promise.resolve(value).then(
      (result) => {
        if (settled) return;
        settled = true;
        signal.removeEventListener("abort", onAbort);
        resolve(result);
      },
      (error) => {
        if (settled) return;
        settled = true;
        signal.removeEventListener("abort", onAbort);
        reject(error);
      },
    );
  });
}

module.exports = {
  DEFAULT_LIMITS,
  IncrementalSSEParser,
  SSERelayError,
  createAbortError,
  createSSEParser,
  readSSEBody,
};
