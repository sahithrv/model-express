const assert = require("node:assert/strict");
const test = require("node:test");

const { buildDatasetUploadPreflight, planZipArchive } = require("./zip-stream.cjs");

test("dataset upload preflight reports warnings and caps", () => {
  const entries = [
    {
      absolutePath: "one.jpg",
      zipPath: "one.jpg",
      size: 10,
      modifiedAt: new Date("2026-01-01T00:00:00Z"),
    },
    {
      absolutePath: "two.jpg",
      zipPath: "two.jpg",
      size: 20,
      modifiedAt: new Date("2026-01-01T00:00:00Z"),
    },
  ];
  const plan = planZipArchive(entries);
  const preflight = buildDatasetUploadPreflight(entries, plan, {
    warnFileCount: 1,
    warnBytes: 15,
    maxFileCount: 1,
    maxBytes: 25,
  });

  assert.equal(preflight.file_count, 2);
  assert.equal(preflight.uncompressed_size_bytes, 30);
  assert.equal(preflight.largest_file.path, "two.jpg");
  assert.deepEqual(
    preflight.warnings.map((warning) => warning.code),
    ["dataset_upload_file_count_warning", "dataset_upload_size_warning"],
  );
  assert.deepEqual(
    preflight.errors.map((error) => error.code),
    ["dataset_upload_file_count_cap", "dataset_upload_size_cap"],
  );
});
