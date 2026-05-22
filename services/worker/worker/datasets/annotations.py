from __future__ import annotations

import csv
import json
from pathlib import Path
from xml.etree import ElementTree


def parse_split_file(split_path: Path, dataset_dir: Path | None = None) -> list[dict]:
    """Parse small explicit split files into bounded metadata records."""
    inferred_split = _split_from_filename(split_path)
    records: list[dict] = []
    with split_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            if len(row) == 1:
                tokens = row[0].strip().split()
            else:
                tokens = [token.strip() for token in row if token.strip()]
            record = _split_record(tokens, inferred_split, dataset_dir)
            if record is not None:
                records.append(record)
    return records[:10_000]


def parse_pascal_voc_bboxes(xml_path: Path) -> dict:
    """Parse Pascal VOC-style bbox XML without changing training behavior."""
    root = ElementTree.parse(xml_path).getroot()
    filename = _text(root.find("filename"))
    size = root.find("size")
    image_size = {
        "width": _int_text(size.find("width")) if size is not None else None,
        "height": _int_text(size.find("height")) if size is not None else None,
        "depth": _int_text(size.find("depth")) if size is not None else None,
    }
    objects = []
    for item in root.findall("object"):
        bbox = item.find("bndbox")
        if bbox is None:
            continue
        objects.append(
            {
                "label": _text(item.find("name")),
                "bbox": {
                    "xmin": _int_text(bbox.find("xmin")),
                    "ymin": _int_text(bbox.find("ymin")),
                    "xmax": _int_text(bbox.find("xmax")),
                    "ymax": _int_text(bbox.find("ymax")),
                },
            }
        )
    return {
        "schema_version": "image_annotation_v1",
        "format": "pascal_voc_xml",
        "path": str(xml_path),
        "filename": filename,
        "image_size": image_size,
        "objects": objects,
    }


def parse_annotation_json_bboxes(json_path: Path) -> dict:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    source = payload if isinstance(payload, dict) else {"annotations": payload}
    raw_items = source.get("objects") or source.get("annotations") or []
    objects = []
    if isinstance(raw_items, list):
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            bbox = _json_bbox(item.get("bbox") or item.get("box") or item)
            if bbox is None:
                continue
            objects.append({"label": str(item.get("label") or item.get("name") or ""), "bbox": bbox})
    return {
        "schema_version": "image_annotation_v1",
        "format": "annotation_json",
        "path": str(json_path),
        "filename": str(source.get("filename") or source.get("image") or ""),
        "objects": objects,
    }


def _split_record(tokens: list[str], inferred_split: str, dataset_dir: Path | None) -> dict | None:
    if not tokens:
        return None
    split = inferred_split
    if tokens[0].lower() in {"train", "val", "valid", "validation", "test"} and len(tokens) >= 2:
        split = _normalize_split(tokens[0])
        relative_path = tokens[1]
        label = tokens[2] if len(tokens) >= 3 else _label_from_path(relative_path)
    else:
        relative_path = tokens[0]
        label = tokens[1] if len(tokens) >= 2 else _label_from_path(relative_path)
        if len(tokens) >= 3 and tokens[2].lower() in {"train", "val", "valid", "validation", "test"}:
            split = _normalize_split(tokens[2])
    absolute_path = str((dataset_dir / relative_path).resolve()) if dataset_dir is not None else ""
    return {"split": split, "path": relative_path, "label": label, "absolute_path": absolute_path}


def _split_from_filename(path: Path) -> str:
    stem = path.stem.lower()
    if stem in {"val", "valid", "validation"}:
        return "val"
    if stem in {"train", "test"}:
        return stem
    return "unknown"


def _normalize_split(value: str) -> str:
    lowered = value.lower()
    if lowered in {"valid", "validation"}:
        return "val"
    return lowered


def _label_from_path(path_value: str) -> str:
    parts = Path(path_value).parts
    return parts[0] if len(parts) > 1 else ""


def _text(element) -> str:
    return "" if element is None or element.text is None else element.text.strip()


def _int_text(element) -> int | None:
    try:
        return int(_text(element))
    except ValueError:
        return None


def _json_bbox(value: object) -> dict | None:
    if isinstance(value, list) and len(value) >= 4:
        try:
            x, y, width, height = [int(float(item)) for item in value[:4]]
        except (TypeError, ValueError):
            return None
        return {"xmin": x, "ymin": y, "xmax": x + width, "ymax": y + height}
    if isinstance(value, dict):
        try:
            if all(key in value for key in ("xmin", "ymin", "xmax", "ymax")):
                return {key: int(float(value[key])) for key in ("xmin", "ymin", "xmax", "ymax")}
            if all(key in value for key in ("x", "y", "width", "height")):
                x = int(float(value["x"]))
                y = int(float(value["y"]))
                width = int(float(value["width"]))
                height = int(float(value["height"]))
                return {"xmin": x, "ymin": y, "xmax": x + width, "ymax": y + height}
        except (TypeError, ValueError):
            return None
    return None
