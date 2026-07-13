"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");

const {
  SSERelayError,
  createSSEParser,
  readSSEBody,
} = require("./sse-relay.cjs");

test("incremental SSE parser preserves split CRLF events, multiline data, and last id", () => {
  const parser = createSSEParser();

  assert.deepEqual(parser.push(Buffer.from("id: 41\r")), []);
  assert.deepEqual(parser.push(Buffer.from("\nevent: execution_event_v2\r\ndata: {\"part\":\r\n")), []);
  assert.deepEqual(parser.push(Buffer.from("data: 1}\r\n\r\n")), [
    {
      event: "execution_event_v2",
      lastEventId: "41",
      data: '{"part":\n1}',
    },
  ]);
  assert.deepEqual(parser.push("data: next\n\n"), [
    { event: "message", lastEventId: "41", data: "next" },
  ]);
  assert.deepEqual(parser.finish(), []);
});

test("incremental SSE parser ignores comments, keepalives, retries, and empty blocks", () => {
  const parser = createSSEParser();
  const events = parser.push(
    ": keepalive\n\nretry: 2000\n\n: another comment\nevent: ignored_without_data\n\n" +
      "id: 9\nevent: execution_event_v2\ndata: safe\n\n",
  );

  assert.deepEqual(events, [
    { event: "execution_event_v2", lastEventId: "9", data: "safe" },
  ]);
  assert.deepEqual(parser.finish(), []);
});

test("incremental SSE parser rejects malformed, incomplete, and oversized input without echoing data", () => {
  const invalidUtf8 = createSSEParser();
  assert.throws(
    () => invalidUtf8.push(Uint8Array.from([0xff])),
    (error) => error instanceof SSERelayError && error.code === "invalid_utf8" && !error.message.includes("ff"),
  );

  const malformed = createSSEParser();
  assert.throws(
    () => malformed.push("data: hidden\u0000payload\n\n"),
    (error) => error instanceof SSERelayError && error.code === "malformed_line" && !error.message.includes("hidden"),
  );

  const oversized = createSSEParser({ maxEventDataBytes: 5 });
  assert.throws(
    () => oversized.push("data: private-payload\n\n"),
    (error) => error instanceof SSERelayError && error.code === "event_data_too_large" && !error.message.includes("private"),
  );

  const incomplete = createSSEParser();
  assert.deepEqual(incomplete.push("data: partial"), []);
  assert.throws(
    () => incomplete.finish(),
    (error) => error instanceof SSERelayError && error.code === "incomplete_event",
  );
});

test("async SSE body reader handles split byte chunks and awaits delivery", async () => {
  const encoder = new TextEncoder();
  const body = new ReadableStream({
    start(controller) {
      controller.enqueue(encoder.encode("id: 7\nevent: execution_"));
      controller.enqueue(encoder.encode("event_v2\ndata: {\"ok\":true}\n\n"));
      controller.close();
    },
  });
  const delivered = [];

  const result = await readSSEBody({ body }, {
    onEvent: async (event) => {
      await Promise.resolve();
      delivered.push(event);
    },
  });

  assert.deepEqual(delivered, [
    { event: "execution_event_v2", lastEventId: "7", data: '{"ok":true}' },
  ]);
  assert.deepEqual(result, { eventCount: 1, lastEventId: "7" });
});

test("async SSE body reader honors an AbortSignal and cancels its reader", async () => {
  let resolveRead;
  let cancelCount = 0;
  let releaseCount = 0;
  const reader = {
    read() {
      return new Promise((resolve) => {
        resolveRead = resolve;
      });
    },
    cancel() {
      cancelCount += 1;
      resolveRead?.({ done: true });
      return Promise.resolve();
    },
    releaseLock() {
      releaseCount += 1;
    },
  };
  const controller = new AbortController();
  const pending = readSSEBody({ body: { getReader: () => reader } }, {
    signal: controller.signal,
    onEvent: () => {
      throw new Error("no event expected");
    },
  });

  controller.abort("private abort reason");
  await assert.rejects(
    pending,
    (error) => error?.name === "AbortError" && error?.code === "ABORT_ERR" && !error.message.includes("private"),
  );
  assert.equal(cancelCount >= 1, true);
  assert.equal(releaseCount, 1);
});

test("async SSE body reader rejects a signal that was already aborted", async () => {
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(
    readSSEBody({ body: { getReader: () => assert.fail("reader must not be acquired") } }, {
      signal: controller.signal,
      onEvent: () => {},
    }),
    (error) => error?.name === "AbortError" && error?.code === "ABORT_ERR",
  );
});
