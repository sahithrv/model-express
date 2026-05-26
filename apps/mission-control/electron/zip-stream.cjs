const fs = require("fs");
const path = require("path");
const { Readable } = require("stream");

const ZIP32_MAX = 0xffffffff;
const ZIP64_VERSION = 45;
const ZIP32_VERSION = 20;
const DATA_DESCRIPTOR_FLAG = 0x08;
const UTF8_FILE_NAME_FLAG = 0x0800;
const GENERAL_PURPOSE_FLAGS = DATA_DESCRIPTOR_FLAG | UTF8_FILE_NAME_FLAG;

async function collectDatasetFiles(rootPath) {
  const root = path.resolve(rootPath);
  const entries = [];

  async function walk(currentPath, relativeParts) {
    const directory = await fs.promises.opendir(currentPath);
    for await (const dirent of directory) {
      const absolutePath = path.join(currentPath, dirent.name);
      const nextParts = [...relativeParts, dirent.name];

      if (dirent.isDirectory()) {
        await walk(absolutePath, nextParts);
        continue;
      }

      if (!dirent.isFile()) {
        continue;
      }

      const stat = await fs.promises.stat(absolutePath);
      entries.push({
        absolutePath,
        zipPath: nextParts.join("/"),
        size: stat.size,
        modifiedAt: stat.mtime,
      });
    }
  }

  await walk(root, []);
  entries.sort((left, right) => left.zipPath.localeCompare(right.zipPath));
  return entries;
}

function planZipArchive(entries) {
  let offset = 0;
  const plannedEntries = entries.map((entry) => {
    const fileName = Buffer.from(entry.zipPath, "utf8");
    const localZip64 = entry.size >= ZIP32_MAX;
    const localExtraLength = localZip64 ? 20 : 0;
    const dataDescriptorLength = localZip64 ? 24 : 16;
    const localHeaderOffset = offset;
    const localHeaderLength = 30 + fileName.length + localExtraLength;

    offset += localHeaderLength + entry.size + dataDescriptorLength;

    return {
      ...entry,
      fileName,
      localZip64,
      localHeaderOffset,
      localExtraLength,
      dataDescriptorLength,
    };
  });

  const centralDirectoryOffset = offset;
  let centralDirectorySize = 0;
  for (const entry of plannedEntries) {
    entry.centralZip64 = entry.localZip64 || entry.localHeaderOffset >= ZIP32_MAX;
    entry.centralExtraLength = entry.centralZip64 ? 28 : 0;
    centralDirectorySize += 46 + entry.fileName.length + entry.centralExtraLength;
  }

  const needsZip64End =
    plannedEntries.length >= 0xffff ||
    centralDirectoryOffset >= ZIP32_MAX ||
    centralDirectorySize >= ZIP32_MAX;
  const zip64EndLength = needsZip64End ? 76 : 0;
  const archiveSize = centralDirectoryOffset + centralDirectorySize + zip64EndLength + 22;

  return {
    entries: plannedEntries,
    centralDirectoryOffset,
    centralDirectorySize,
    needsZip64End,
    archiveSize,
  };
}

function createZipArchiveStream(plan) {
  return Readable.from(generateZipArchive(plan));
}

async function* generateZipArchive(plan) {
  for (const entry of plan.entries) {
    yield localFileHeader(entry);

    let crc = 0xffffffff;
    for await (const chunk of fs.createReadStream(entry.absolutePath)) {
      crc = updateCrc32(crc, chunk);
      yield chunk;
    }

    entry.crc32 = (crc ^ 0xffffffff) >>> 0;
    yield dataDescriptor(entry);
  }

  for (const entry of plan.entries) {
    yield centralDirectoryHeader(entry);
  }

  if (plan.needsZip64End) {
    const zip64EndOffset = plan.centralDirectoryOffset + plan.centralDirectorySize;
    yield zip64EndOfCentralDirectory(plan);
    yield zip64EndOfCentralDirectoryLocator(zip64EndOffset);
  }

  yield endOfCentralDirectory(plan);
}

function localFileHeader(entry) {
  const versionNeeded = entry.localZip64 ? ZIP64_VERSION : ZIP32_VERSION;
  const extra = entry.localZip64
    ? zip64ExtraField([entry.size, entry.size])
    : Buffer.alloc(0);
  const header = Buffer.alloc(30);
  const dos = dosDateTime(entry.modifiedAt);

  header.writeUInt32LE(0x04034b50, 0);
  header.writeUInt16LE(versionNeeded, 4);
  header.writeUInt16LE(GENERAL_PURPOSE_FLAGS, 6);
  header.writeUInt16LE(0, 8);
  header.writeUInt16LE(dos.time, 10);
  header.writeUInt16LE(dos.date, 12);
  header.writeUInt32LE(0, 14);
  header.writeUInt32LE(entry.localZip64 ? ZIP32_MAX : 0, 18);
  header.writeUInt32LE(entry.localZip64 ? ZIP32_MAX : 0, 22);
  header.writeUInt16LE(entry.fileName.length, 26);
  header.writeUInt16LE(extra.length, 28);

  return Buffer.concat([header, entry.fileName, extra]);
}

function dataDescriptor(entry) {
  const descriptor = Buffer.alloc(entry.localZip64 ? 24 : 16);
  descriptor.writeUInt32LE(0x08074b50, 0);
  descriptor.writeUInt32LE(entry.crc32, 4);
  if (entry.localZip64) {
    writeUInt64LE(descriptor, entry.size, 8);
    writeUInt64LE(descriptor, entry.size, 16);
  } else {
    descriptor.writeUInt32LE(entry.size, 8);
    descriptor.writeUInt32LE(entry.size, 12);
  }
  return descriptor;
}

function centralDirectoryHeader(entry) {
  const version = entry.centralZip64 || entry.localZip64 ? ZIP64_VERSION : ZIP32_VERSION;
  const extra = entry.centralZip64
    ? zip64ExtraField([entry.size, entry.size, entry.localHeaderOffset])
    : Buffer.alloc(0);
  const header = Buffer.alloc(46);
  const dos = dosDateTime(entry.modifiedAt);

  header.writeUInt32LE(0x02014b50, 0);
  header.writeUInt16LE(version, 4);
  header.writeUInt16LE(version, 6);
  header.writeUInt16LE(GENERAL_PURPOSE_FLAGS, 8);
  header.writeUInt16LE(0, 10);
  header.writeUInt16LE(dos.time, 12);
  header.writeUInt16LE(dos.date, 14);
  header.writeUInt32LE(entry.crc32, 16);
  header.writeUInt32LE(entry.centralZip64 ? ZIP32_MAX : entry.size, 20);
  header.writeUInt32LE(entry.centralZip64 ? ZIP32_MAX : entry.size, 24);
  header.writeUInt16LE(entry.fileName.length, 28);
  header.writeUInt16LE(extra.length, 30);
  header.writeUInt16LE(0, 32);
  header.writeUInt16LE(0, 34);
  header.writeUInt16LE(0, 36);
  header.writeUInt32LE(0, 38);
  header.writeUInt32LE(entry.centralZip64 ? ZIP32_MAX : entry.localHeaderOffset, 42);

  return Buffer.concat([header, entry.fileName, extra]);
}

function zip64EndOfCentralDirectory(plan) {
  const record = Buffer.alloc(56);
  record.writeUInt32LE(0x06064b50, 0);
  writeUInt64LE(record, 44, 4);
  record.writeUInt16LE(ZIP64_VERSION, 12);
  record.writeUInt16LE(ZIP64_VERSION, 14);
  record.writeUInt32LE(0, 16);
  record.writeUInt32LE(0, 20);
  writeUInt64LE(record, plan.entries.length, 24);
  writeUInt64LE(record, plan.entries.length, 32);
  writeUInt64LE(record, plan.centralDirectorySize, 40);
  writeUInt64LE(record, plan.centralDirectoryOffset, 48);
  return record;
}

function zip64EndOfCentralDirectoryLocator(zip64EndOffset) {
  const locator = Buffer.alloc(20);
  locator.writeUInt32LE(0x07064b50, 0);
  locator.writeUInt32LE(0, 4);
  writeUInt64LE(locator, zip64EndOffset, 8);
  locator.writeUInt32LE(1, 16);
  return locator;
}

function endOfCentralDirectory(plan) {
  const record = Buffer.alloc(22);
  const entryCount = plan.entries.length >= 0xffff ? 0xffff : plan.entries.length;
  const centralSize = plan.centralDirectorySize >= ZIP32_MAX ? ZIP32_MAX : plan.centralDirectorySize;
  const centralOffset = plan.centralDirectoryOffset >= ZIP32_MAX ? ZIP32_MAX : plan.centralDirectoryOffset;

  record.writeUInt32LE(0x06054b50, 0);
  record.writeUInt16LE(0, 4);
  record.writeUInt16LE(0, 6);
  record.writeUInt16LE(entryCount, 8);
  record.writeUInt16LE(entryCount, 10);
  record.writeUInt32LE(centralSize, 12);
  record.writeUInt32LE(centralOffset, 16);
  record.writeUInt16LE(0, 20);
  return record;
}

function zip64ExtraField(values) {
  const extra = Buffer.alloc(4 + values.length * 8);
  extra.writeUInt16LE(0x0001, 0);
  extra.writeUInt16LE(values.length * 8, 2);
  values.forEach((value, index) => writeUInt64LE(extra, value, 4 + index * 8));
  return extra;
}

function writeUInt64LE(buffer, value, offset) {
  const bigValue = BigInt(value);
  buffer.writeUInt32LE(Number(bigValue & 0xffffffffn), offset);
  buffer.writeUInt32LE(Number((bigValue >> 32n) & 0xffffffffn), offset + 4);
}

function dosDateTime(date) {
  const value = date instanceof Date ? date : new Date();
  const year = Math.max(1980, value.getFullYear());
  return {
    time:
      (value.getHours() << 11) |
      (value.getMinutes() << 5) |
      Math.floor(value.getSeconds() / 2),
    date:
      ((year - 1980) << 9) |
      ((value.getMonth() + 1) << 5) |
      value.getDate(),
  };
}

function updateCrc32(crc, chunk) {
  for (let index = 0; index < chunk.length; index += 1) {
    crc = CRC32_TABLE[(crc ^ chunk[index]) & 0xff] ^ (crc >>> 8);
  }
  return crc;
}

function buildCrc32Table() {
  const table = new Uint32Array(256);
  for (let index = 0; index < 256; index += 1) {
    let value = index;
    for (let bit = 0; bit < 8; bit += 1) {
      value = value & 1 ? 0xedb88320 ^ (value >>> 1) : value >>> 1;
    }
    table[index] = value >>> 0;
  }
  return table;
}

const CRC32_TABLE = buildCrc32Table();

module.exports = {
  collectDatasetFiles,
  createZipArchiveStream,
  planZipArchive,
};
