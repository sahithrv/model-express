from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path

from worker.training.modal_common import (
    _bounded_float,
    _bounded_int,
    _positive_int,
    _safe_path_part,
)
from worker.training.modal_exports import (
    _artifact_errors,
    _artifact_remote_base_uri,
    _export_error,
    _upload_manifest_artifacts,
)
from worker.training.modal_materialization import _dataset_checksum
from worker.training.modal_runtime import IMAGE_SUFFIXES, _bool


def _preview_dataset_tiers_enabled() -> bool:
    return _bool(os.getenv("MODEL_EXPRESS_PREVIEW_DATASET_TIERS"), default=False)


def _yolo_batch_preview_enabled() -> bool:
    return _bool(os.getenv("MODEL_EXPRESS_YOLO_BATCH_PREVIEW"), default=False)


def _dataset_tier_config(
    config: dict,
    *,
    dataset: dict,
    task_type: str,
    dataset_dir: Path | None = None,
    batch: dict | None = None,
    force_preview: bool = False,
) -> dict:
    tier = str(config.get("training_tier") or config.get("dataset_tier") or "").strip().lower()
    if not tier and isinstance(config.get("modal_batch"), dict):
        tier = str(config["modal_batch"].get("training_tier") or "").strip().lower()
    if force_preview:
        tier = "preview"
    if tier not in {"preview", "full", "champion_validation"}:
        tier = "full"
    fraction = _bounded_float(
        config.get("preview_fraction", config.get("dataset_fraction", config.get("preview_dataset_fraction", 0.25))),
        default=0.25,
        minimum=0.01,
        maximum=1.0,
    )
    seed = _bounded_int(config.get("preview_seed", config.get("seed", 42)), default=42, minimum=0, maximum=2**31 - 1)
    split_policy = str(config.get("split_policy") or "official_or_deterministic").strip().lower()
    image_size = _positive_int(config.get("image_size"), default=0)
    image_size_family = str(config.get("image_size_family") or "").strip().lower()
    if not image_size_family:
        image_size_family = _image_size_family(task_type, image_size)
    if isinstance(batch, dict):
        fraction = _bounded_float(batch.get("subset_fraction", fraction), default=fraction, minimum=0.01, maximum=1.0)
        seed = _bounded_int(batch.get("subset_seed", seed), default=seed, minimum=0, maximum=2**31 - 1)
        split_policy = str(batch.get("split_policy") or split_policy).strip().lower()
        image_size_family = str(batch.get("image_size_family") or image_size_family).strip().lower()
    return {
        "enabled": _preview_dataset_tiers_enabled(),
        "tier": tier,
        "task_type": task_type,
        "dataset_checksum": _dataset_checksum(dataset, config) or str(config.get("dataset_checksum_sha256") or ""),
        "fraction": fraction,
        "seed": seed,
        "split_policy": split_policy,
        "image_size_family": image_size_family,
        "dataset_dir": str(dataset_dir) if dataset_dir is not None else "",
    }


def _image_size_family(task_type: str, image_size: int) -> str:
    if image_size <= 0:
        return f"{task_type}:default"
    bucket = ((image_size + 31) // 32) * 32
    return f"{task_type}:{bucket}"


def _prepare_yolo_dataset_tier(
    dataset_dir: Path,
    data_config_path: Path,
    config: dict,
    *,
    dataset: dict,
    output_root: Path,
    batch: dict | None = None,
    force_preview: bool = False,
) -> tuple[Path, dict]:
    tier_config = _dataset_tier_config(
        config,
        dataset=dataset,
        task_type="object_detection",
        dataset_dir=dataset_dir,
        batch=batch,
        force_preview=force_preview,
    )
    if not tier_config["enabled"] or tier_config["tier"] != "preview":
        return data_config_path, {}
    from worker.datasets.tiers import materialize_yolo_preview_subset

    return materialize_yolo_preview_subset(
        dataset_dir=dataset_dir,
        data_config_path=data_config_path,
        output_root=output_root,
        dataset_checksum=str(tier_config["dataset_checksum"]),
        fraction=float(tier_config["fraction"]),
        seed=int(tier_config["seed"]),
        split_policy=str(tier_config["split_policy"]),
        image_size_family=str(tier_config["image_size_family"]),
    )


YOLO_CONFIG_FILE_NAMES = ("data.yaml", "data.yml", "dataset.yaml", "dataset.yml")


def _find_yolo_data_config(dataset_dir: Path, config: dict) -> Path | None:
    configured = str(config.get("yolo_data_config") or config.get("data_yaml") or "").strip()
    candidates: list[Path] = []
    if configured:
        configured_path = Path(configured)
        candidates.append(configured_path if configured_path.is_absolute() else dataset_dir / configured)
    for name in YOLO_CONFIG_FILE_NAMES:
        candidates.append(dataset_dir / name)
    for path in sorted(dataset_dir.rglob("*")):
        if path.is_file() and path.name.lower() in YOLO_CONFIG_FILE_NAMES:
            candidates.append(path)
    seen: set[Path] = set()
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
        if _looks_like_yolo_data_config(resolved):
            return resolved
    return None


def _looks_like_yolo_data_config(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
    except Exception:
        return False
    return ("train:" in text and ("val:" in text or "valid:" in text)) and ("names:" in text or "nc:" in text)


def _normalize_yolo_data_config_for_training(
    dataset_dir: Path,
    data_config_path: Path,
    *,
    output_root: Path,
) -> Path:
    loaded = _load_yolo_training_config(data_config_path)

    normalized = dict(loaded)
    normalized["path"] = str(dataset_dir.resolve())
    for split in ("train", "val", "test"):
        source_key = "valid" if split == "val" and "val" not in normalized and "valid" in normalized else split
        if source_key not in normalized:
            continue
        normalized[split] = _normalize_yolo_split_value(
            dataset_dir,
            data_config_path,
            loaded,
            normalized[source_key],
        )
    normalized.pop("valid", None)

    missing_required = [
        split
        for split in ("train", "val")
        if split not in normalized or not _yolo_split_value_has_images(normalized[split])
    ]
    if missing_required:
        raise ValueError(
            "YOLO detector training could not resolve local image paths for "
            f"{', '.join(missing_required)} from {data_config_path}."
        )

    output_root.mkdir(parents=True, exist_ok=True)
    normalized_path = output_root / "data.yaml"
    normalized_path.write_text(_dump_yolo_training_config(normalized), encoding="utf-8")
    return normalized_path


def _load_yolo_training_config(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml
    except Exception:
        loaded = _load_simple_yolo_training_config(text)
    else:
        loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"YOLO data config {path} must be a mapping.")
    return loaded


def _dump_yolo_training_config(config: dict) -> str:
    try:
        import yaml
    except Exception:
        return _dump_simple_yolo_training_config(config)
    return yaml.safe_dump(config, sort_keys=False)


def _load_simple_yolo_training_config(text: str) -> dict:
    config: dict = {}
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        raw = _strip_yaml_comment(lines[index]).rstrip()
        index += 1
        if not raw.strip() or raw.startswith((" ", "\t")) or ":" not in raw:
            continue
        key, value = raw.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value:
            config[key] = _parse_simple_yaml_value(value)
            continue
        block: list[str] = []
        while index < len(lines):
            child = _strip_yaml_comment(lines[index]).rstrip()
            if child and not child.startswith((" ", "\t")):
                break
            if child.strip():
                block.append(child.strip())
            index += 1
        config[key] = _parse_simple_yaml_block(block)
    return config


def _strip_yaml_comment(line: str) -> str:
    in_quote = ""
    for index, char in enumerate(line):
        if char in {"'", '"'}:
            in_quote = "" if in_quote == char else char if not in_quote else in_quote
            continue
        if char == "#" and not in_quote:
            return line[:index]
    return line


def _parse_simple_yaml_value(value: str) -> object:
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_unquote_yaml_scalar(part.strip()) for part in inner.split(",")]
    unquoted = _unquote_yaml_scalar(value)
    try:
        return int(unquoted)
    except ValueError:
        return unquoted


def _parse_simple_yaml_block(block: list[str]) -> object:
    if not block:
        return ""
    if all(line.startswith("- ") for line in block):
        return [_unquote_yaml_scalar(line[2:].strip()) for line in block]
    mapped: dict[int | str, str] = {}
    for line in block:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        parsed_key: int | str
        try:
            parsed_key = int(key)
        except ValueError:
            parsed_key = key
        mapped[parsed_key] = _unquote_yaml_scalar(value.strip())
    if mapped and all(isinstance(key, int) for key in mapped):
        return [mapped[key] for key in sorted(mapped)]
    return mapped


def _unquote_yaml_scalar(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        if value[0] == '"':
            try:
                return str(json.loads(value))
            except json.JSONDecodeError:
                pass
        return value[1:-1]
    return value


def _dump_simple_yolo_training_config(config: dict) -> str:
    lines: list[str] = []
    for key in ("path", "train", "val", "test", "nc"):
        if key in config:
            lines.append(f"{key}: {_format_simple_yaml_scalar(config[key])}")
    names = config.get("names")
    if isinstance(names, dict):
        lines.append("names:")
        for key in sorted(names, key=lambda item: int(item) if str(item).isdigit() else str(item)):
            lines.append(f"  {key}: {_format_simple_yaml_scalar(names[key])}")
    elif isinstance(names, list):
        lines.append("names:")
        for name in names:
            lines.append(f"  - {_format_simple_yaml_scalar(name)}")
    elif names is not None:
        lines.append(f"names: {_format_simple_yaml_scalar(names)}")
    for key, value in config.items():
        if key in {"path", "train", "val", "test", "nc", "names", "valid"}:
            continue
        if isinstance(value, (str, int, float, bool)):
            lines.append(f"{key}: {_format_simple_yaml_scalar(value)}")
    return "\n".join(lines) + "\n"


def _format_simple_yaml_scalar(value: object) -> str:
    if isinstance(value, list):
        return "[" + ", ".join(_format_simple_yaml_scalar(item) for item in value) + "]"
    text = str(value)
    if not text or any(char in text for char in ":#[]{}") or text.strip() != text:
        return json.dumps(text)
    return text


def _normalize_yolo_split_value(
    dataset_dir: Path,
    data_config_path: Path,
    config: dict,
    value: object,
) -> object:
    if isinstance(value, list):
        return [
            str(_resolve_existing_yolo_path(dataset_dir, data_config_path, config, str(item)))
            for item in value
            if str(item or "").strip()
        ]
    if value is None:
        return value
    text = str(value).strip()
    if not text:
        return value
    return str(_resolve_existing_yolo_path(dataset_dir, data_config_path, config, text))


def _resolve_existing_yolo_path(
    dataset_dir: Path,
    data_config_path: Path,
    config: dict,
    value: str,
) -> Path:
    candidates = _yolo_local_path_candidates(dataset_dir, data_config_path, config, value)
    for candidate in candidates:
        if candidate.exists() and _path_is_within(candidate, dataset_dir):
            return candidate.resolve()
    suffix = _normalized_yolo_suffix(value)
    if suffix:
        for candidate in sorted(dataset_dir.rglob(Path(suffix).name), key=lambda path: str(path)):
            if _path_suffix(candidate, suffix) and candidate.exists():
                return candidate.resolve()
    for candidate in candidates:
        if _path_is_within(candidate, dataset_dir):
            return candidate.resolve()
    fallback = (data_config_path.parent / value).resolve()
    if _path_is_within(fallback, dataset_dir):
        return fallback
    raise ValueError(f"YOLO data config path is outside the materialized dataset: {value!r}")


def _yolo_local_path_candidates(
    dataset_dir: Path,
    data_config_path: Path,
    config: dict,
    value: str,
) -> list[Path]:
    _validate_yolo_path_value(value)
    path = Path(value)
    candidates: list[Path] = []
    config_parent = data_config_path.parent
    if path.is_absolute():
        for suffix in _absolute_path_suffixes(path):
            candidates.append(dataset_dir / suffix)
            candidates.append(config_parent / suffix)
        return _unique_paths(candidates)

    root_value = str(config.get("path") or "").strip()
    if root_value:
        _validate_yolo_path_value(root_value)
        root = Path(root_value)
        if root.is_absolute():
            if root.name:
                candidates.append(dataset_dir / root.name / path)
                candidates.append(config_parent / root.name / path)
            for suffix in _absolute_path_suffixes(root):
                candidates.append(dataset_dir / suffix / path)
                candidates.append(config_parent / suffix / path)
        else:
            candidates.append((config_parent / root / path).resolve())
    candidates.append(config_parent / path)
    candidates.append(dataset_dir / path)
    return _unique_paths(candidates)


def _absolute_path_suffixes(path: Path) -> list[Path]:
    parts = [part for part in path.parts if part and part != path.anchor]
    suffixes: list[Path] = []
    for index in range(len(parts)):
        suffixes.append(Path(*parts[index:]))
    return suffixes


def _normalized_yolo_suffix(value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        suffixes = _absolute_path_suffixes(path)
        return suffixes[-1].as_posix() if suffixes else ""
    return value.replace("\\", "/").lstrip("./").strip()


def _path_suffix(path: Path, suffix: str) -> bool:
    normalized_path = path.resolve().as_posix().rstrip("/").lower()
    normalized_suffix = suffix.replace("\\", "/").strip("/").lower()
    return bool(normalized_suffix) and normalized_path.endswith(f"/{normalized_suffix}")


def _validate_yolo_path_value(value: str) -> None:
    text = str(value or "").strip()
    if not text:
        return
    lowered = text.lower()
    if "://" in lowered or lowered.startswith(("s3:", "gs:", "http:", "https:", "file:")):
        raise ValueError(f"YOLO data config path uses an unsupported URI: {value!r}")
    if ".." in Path(text.replace("\\", "/")).parts:
        raise ValueError(f"YOLO data config path contains parent traversal: {value!r}")


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except (OSError, ValueError):
        return False


def _unique_paths(paths: list[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _yolo_split_value_has_images(value: object) -> bool:
    values = value if isinstance(value, list) else [value]
    for item in values:
        if item is None:
            continue
        path = Path(str(item))
        if path.is_dir() and any(child.suffix.lower() in IMAGE_SUFFIXES for child in path.rglob("*") if child.is_file()):
            return True
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            return True
        if path.is_file():
            try:
                lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
            except OSError:
                lines = []
            for line in lines:
                image_path = Path(line.strip())
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
                    return True
    return False


def _yolo_config_declares_split(path: Path, split: str) -> bool:
    split = "val" if split == "valid" else split
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    keys = [split]
    if split == "val":
        keys.append("valid")
    return any(re.search(rf"(?m)^\s*{re.escape(key)}\s*:", text) is not None for key in keys)


def _yolo_metrics_from_object(metrics: object, *, class_names: list[str] | None = None) -> dict:
    results_dict = getattr(metrics, "results_dict", None)
    if not isinstance(results_dict, dict):
        results_dict = {}
    box = getattr(metrics, "box", None)
    speed = getattr(metrics, "speed", None)
    out = {
        "mAP50_95": _first_metric_value(results_dict, "metrics/mAP50-95(B)", "metrics/mAP50-95", "mAP50_95", "map50_95"),
        "mAP50": _first_metric_value(results_dict, "metrics/mAP50(B)", "metrics/mAP50", "mAP50", "map50"),
        "precision": _first_metric_value(results_dict, "metrics/precision(B)", "metrics/precision", "precision"),
        "recall": _first_metric_value(results_dict, "metrics/recall(B)", "metrics/recall", "recall"),
        "box_loss": _first_metric_value(results_dict, "val/box_loss", "box_loss"),
        "cls_loss": _first_metric_value(results_dict, "val/cls_loss", "cls_loss"),
        "dfl_loss": _first_metric_value(results_dict, "val/dfl_loss", "dfl_loss"),
    }
    if box is not None:
        out["mAP50_95"] = out["mAP50_95"] or _object_float(box, "map")
        out["mAP50"] = out["mAP50"] or _object_float(box, "map50")
        out["precision"] = out["precision"] or _object_float(box, "mp")
        out["recall"] = out["recall"] or _object_float(box, "mr")
        per_class_metrics = _yolo_per_class_metrics_from_box(box, class_names or _yolo_class_names_from_metrics(metrics))
        if per_class_metrics:
            out["per_class_metrics"] = per_class_metrics
    if isinstance(speed, dict):
        inference_ms = _number(speed.get("inference"))
        preprocess_ms = _number(speed.get("preprocess"))
        postprocess_ms = _number(speed.get("postprocess"))
        if inference_ms > 0:
            out["latency_model_ms"] = inference_ms
        if preprocess_ms > 0:
            out["latency_preprocess_ms"] = preprocess_ms
        if postprocess_ms > 0:
            out["latency_postprocess_ms"] = postprocess_ms
    cleaned: dict[str, object] = {}
    for key, value in out.items():
        if key == "per_class_metrics":
            if isinstance(value, dict) and value:
                cleaned[key] = value
            continue
        if _number(value) > 0:
            cleaned[key] = round(float(value), 6)
    return cleaned


def _find_yolo_results_csv(run_root: Path) -> Path | None:
    candidates = [
        run_root / "train" / "results.csv",
        run_root / "results.csv",
    ]
    candidates.extend(sorted(run_root.glob("**/results.csv"), key=lambda path: len(path.parts)))
    for path in candidates:
        if path.is_file():
            return path
    return None


def _yolo_epoch_metrics_from_row(row: dict, *, learning_rate: float, image_size: int) -> dict:
    metrics = {
        "mAP50_95": _first_metric_value(row, "metrics/mAP50-95(B)", "metrics/mAP50-95", "mAP50_95", "map50_95"),
        "mAP50": _first_metric_value(row, "metrics/mAP50(B)", "metrics/mAP50", "mAP50", "map50"),
        "precision": _first_metric_value(row, "metrics/precision(B)", "metrics/precision", "precision"),
        "recall": _first_metric_value(row, "metrics/recall(B)", "metrics/recall", "recall"),
        "box_loss": _first_metric_value(row, "val/box_loss", "box_loss"),
        "cls_loss": _first_metric_value(row, "val/cls_loss", "cls_loss"),
        "dfl_loss": _first_metric_value(row, "val/dfl_loss", "dfl_loss"),
    }
    train_box = _first_metric_value(row, "train/box_loss")
    train_cls = _first_metric_value(row, "train/cls_loss")
    train_dfl = _first_metric_value(row, "train/dfl_loss")
    val_box = _first_metric_value(row, "val/box_loss")
    val_cls = _first_metric_value(row, "val/cls_loss")
    val_dfl = _first_metric_value(row, "val/dfl_loss")
    if train_box > 0 or train_cls > 0 or train_dfl > 0:
        metrics["train_loss"] = train_box + train_cls + train_dfl
    if val_box > 0 or val_cls > 0 or val_dfl > 0:
        metrics["val_loss"] = val_box + val_cls + val_dfl
    metrics["learning_rate"] = _first_metric_value(row, "lr/pg0", "lr/pg1", "lr/pg2", "learning_rate") or learning_rate
    metrics["image_size"] = image_size
    if metrics["mAP50_95"] > 0:
        metrics["map50_95"] = metrics["mAP50_95"]
    if metrics["mAP50"] > 0:
        metrics["map50"] = metrics["mAP50"]
    return {key: round(float(value), 6) for key, value in metrics.items() if _number(value) > 0}


def _first_metric_value(values: dict, *keys: str) -> float:
    normalized_values = {}
    for raw_key, raw_value in values.items():
        normalized_values[str(raw_key).strip()] = raw_value
    for key in keys:
        number = _number(normalized_values.get(key))
        if number > 0:
            return number
    return 0.0


def _object_float(obj: object, attr: str) -> float:
    return _number(getattr(obj, attr, None))


def _numeric_sequence(value: object) -> list[float]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        try:
            value = value.detach()
        except Exception:
            pass
    if hasattr(value, "cpu"):
        try:
            value = value.cpu()
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            value = value.tolist()
        except Exception:
            pass
    if isinstance(value, (list, tuple)):
        out: list[float] = []
        for item in value:
            if isinstance(item, (list, tuple)):
                continue
            number = _number(item)
            if number >= 0:
                out.append(number)
        return out
    number = _number(value)
    return [number] if number > 0 else []


def _numeric_matrix(value: object) -> list[list[float]]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        try:
            value = value.detach()
        except Exception:
            pass
    if hasattr(value, "cpu"):
        try:
            value = value.cpu()
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            value = value.tolist()
        except Exception:
            pass
    if not isinstance(value, (list, tuple)):
        return []
    out: list[list[float]] = []
    for row in value:
        if not isinstance(row, (list, tuple)):
            continue
        numbers = [_number(item) for item in row]
        if any(number > 0 for number in numbers):
            out.append(numbers)
    return out


def _class_index_sequence(value: object, fallback_count: int) -> list[int]:
    values = _numeric_sequence(value)
    if values:
        return [int(item) for item in values]
    return list(range(max(0, fallback_count)))


def _yolo_class_names_from_metrics(metrics: object) -> list[str]:
    names = getattr(metrics, "names", None)
    if isinstance(names, dict):
        return [str(names[key]).strip() for key in sorted(names) if str(names[key]).strip()][:200]
    if isinstance(names, list):
        return [str(item).strip() for item in names if str(item).strip()][:200]
    return []


def _yolo_class_name(class_names: list[str], class_index: int, row_index: int) -> str:
    if 0 <= class_index < len(class_names) and class_names[class_index].strip():
        return class_names[class_index].strip()
    if 0 <= row_index < len(class_names) and class_names[row_index].strip():
        return class_names[row_index].strip()
    return f"class_{class_index if class_index >= 0 else row_index}"


def _yolo_metric_at(values: list[float], row_index: int, class_index: int) -> float:
    if 0 <= class_index < len(values):
        return values[class_index]
    if 0 <= row_index < len(values):
        return values[row_index]
    return 0.0


def _yolo_per_class_metrics_from_box(box: object, class_names: list[str]) -> dict:
    precision_values = _numeric_sequence(getattr(box, "p", None))
    recall_values = _numeric_sequence(getattr(box, "r", None))
    ap50_values = _numeric_sequence(getattr(box, "ap50", None))
    map_values = _numeric_sequence(getattr(box, "maps", None))
    all_ap = _numeric_matrix(getattr(box, "all_ap", None))
    row_count = max(len(precision_values), len(recall_values), len(ap50_values), len(map_values), len(all_ap))
    if row_count == 0:
        return {}
    class_indices = _class_index_sequence(getattr(box, "ap_class_index", None), row_count)
    out: dict[str, dict[str, float]] = {}
    for row_index in range(row_count):
        class_index = class_indices[row_index] if row_index < len(class_indices) else row_index
        ap_row = all_ap[row_index] if row_index < len(all_ap) else []
        ap50 = _yolo_metric_at(ap50_values, row_index, class_index)
        if ap50 <= 0 and ap_row:
            ap50 = ap_row[0]
        map50_95 = _yolo_metric_at(map_values, row_index, class_index)
        if map50_95 <= 0 and ap_row:
            positive_ap = [value for value in ap_row if value > 0]
            if positive_ap:
                map50_95 = sum(positive_ap) / len(positive_ap)
        precision = _yolo_metric_at(precision_values, row_index, class_index)
        recall = _yolo_metric_at(recall_values, row_index, class_index)
        if precision <= 0 and recall <= 0 and ap50 <= 0 and map50_95 <= 0:
            continue
        out[_yolo_class_name(class_names, class_index, row_index)] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "AP50": round(ap50, 6),
            "AP50_95": round(map50_95, 6),
        }
    return _yolo_per_class_metrics_with_macro_average(out)


def _metric_float(values: dict, *keys: str, default: float) -> float:
    for key in keys:
        number = _number(values.get(key))
        if number > 0:
            return float(number)
    return float(default)


def _number(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _yolo_best_model_path(run_root: Path) -> Path | None:
    candidates = []
    for name in ("best.pt", "last.pt"):
        candidates.extend(run_root.rglob(name))
    existing = [path for path in candidates if path.is_file()]
    if not existing:
        return None
    return max(existing, key=lambda path: path.stat().st_mtime)


def _yolo_class_names(detector, config: dict) -> list[str]:
    for key in ("class_names", "class_labels", "classes"):
        value = config.get(key)
        if isinstance(value, list):
            names = [str(item).strip() for item in value if str(item).strip()]
            if names:
                return names[:200]
    names = getattr(detector, "names", None)
    if isinstance(names, dict):
        return [str(names[key]).strip() for key in sorted(names) if str(names[key]).strip()][:200]
    if isinstance(names, list):
        return [str(item).strip() for item in names if str(item).strip()][:200]
    metadata = config.get("metadata_summary") if isinstance(config.get("metadata_summary"), dict) else {}
    yolo_summary = metadata.get("yolo_summary") if isinstance(metadata.get("yolo_summary"), dict) else {}
    value = yolo_summary.get("class_names")
    if isinstance(value, list):
        names = [str(item).strip() for item in value if str(item).strip()]
        if names:
            return names[:200]
    return ["class_0"]


def _yolo_model_profile(
    *,
    model_name: str,
    model_path: Path | None,
    image_size: int,
    class_names: list[str],
    confidence_threshold: float,
    iou_threshold: float,
    metrics: dict,
) -> dict:
    size_mb = 0.0
    if model_path is not None and model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
    model_latency_ms = _metric_float(
        metrics,
        "latency_model_ms",
        default=_fallback_yolo_latency_ms(model_name, image_size),
    )
    preprocess_ms = _metric_float(metrics, "latency_preprocess_ms", default=max(1.0, image_size / 512))
    postprocess_ms = _metric_float(metrics, "latency_postprocess_ms", default=max(1.0, len(class_names) * 0.08))
    pipeline_ms = model_latency_ms + preprocess_ms + postprocess_ms
    return {
        "task_type": "object_detection",
        "model_kind": "ultralytics_yolo_detector",
        "runtime": "onnx",
        "model_size_mb": round(size_mb, 3),
        "estimated_model_latency_ms": round(model_latency_ms, 3),
        "estimated_preprocessing_latency_ms": round(preprocess_ms, 3),
        "estimated_postprocessing_latency_ms": round(postprocess_ms, 3),
        "estimated_pipeline_latency_ms": round(pipeline_ms, 3),
        "estimated_latency_ms": round(pipeline_ms, 3),
        "latency_p50_ms": round(pipeline_ms, 3),
        "latency_p95_ms": round(pipeline_ms * 1.35, 3),
        "estimated_throughput_images_per_second": round(1000.0 / max(pipeline_ms, 1.0), 3),
        "image_size": image_size,
        "input_shape": [1, 3, image_size, image_size],
        "class_labels": class_names,
        "confidence_threshold": round(confidence_threshold, 4),
        "iou_threshold": round(iou_threshold, 4),
        "pretrained": True,
    }


def _fallback_yolo_latency_ms(model_name: str, image_size: int) -> float:
    normalized = model_name.lower()
    base = 14.0
    if "yolo11n" in normalized:
        base = 8.0
    elif "yolo11s" in normalized:
        base = 14.0
    elif "yolo11m" in normalized:
        base = 24.0
    elif "yolo11l" in normalized:
        base = 38.0
    elif "yolo11x" in normalized:
        base = 62.0
    return base * max(0.35, (image_size / 640) ** 2)


def _export_yolo_detector_bundle(
    *,
    model_path: Path | None,
    model_name: str,
    class_names: list[str],
    image_size: int,
    model_profile: dict,
    training_config: dict,
    dataset: dict,
    job_id: str,
) -> dict:
    if model_path is None or not model_path.exists():
        return _export_error("YOLO_CHECKPOINT_UNAVAILABLE", "Ultralytics did not produce best.pt or last.pt.")
    try:
        from ultralytics import YOLO
        from worker.datasets.storage import upload_file_to_s3_uri
        from worker.exporting.artifacts import produce_existing_champion_export_manifest
    except Exception as exc:
        return _export_error("EXPORT_DEPENDENCY_UNAVAILABLE", str(exc))

    export_dir = Path(os.getenv("WORKER_CHAMPION_EXPORT_ROOT", ".cache/champion_exports")) / _safe_path_part(job_id) / "yolo"
    validation_errors: list[str] = []
    onnx_path: Path | None = None
    try:
        exported = YOLO(str(model_path)).export(format="onnx", imgsz=image_size, opset=12)
        candidate = Path(str(exported))
        if candidate.exists():
            onnx_path = candidate
    except Exception as exc:
        validation_errors.append(f"YOLO_ONNX_EXPORT_FAILED: {exc}")

    source_path = onnx_path or model_path
    artifact_format = "onnx" if onnx_path is not None else "pytorch"
    try:
        manifest = produce_existing_champion_export_manifest(
            export_dir=export_dir,
            source_artifact_path=source_path,
            artifact_format=artifact_format,
            model_name=model_name,
            class_names=class_names,
            image_size=image_size,
            preprocessing={"resize_strategy": "letterbox", "normalization": "none"},
            model_profile=model_profile,
            training_config={**training_config, "task_type": "object_detection", "model_kind": "ultralytics_yolo_detector"},
            sample_input_shape=[1, 3, image_size, image_size],
        )
        remote_base = _artifact_remote_base_uri(dataset, job_id)
        public_manifest, artifact_uris = _upload_manifest_artifacts(manifest, remote_base, upload_file_to_s3_uri)
        validation_errors.extend(_artifact_errors(public_manifest))
        manifest_uri = f"{remote_base}/manifest.json"
        manifest_path = export_dir / "manifest.remote.json"
        manifest_path.write_text(json.dumps(public_manifest, indent=2, sort_keys=True), encoding="utf-8")
        upload_file_to_s3_uri(manifest_path, manifest_uri)
        onnx_artifact = next((item for item in artifact_uris if item["format"] == "onnx"), None)
        primary_artifact = onnx_artifact or (artifact_uris[0] if artifact_uris else None)
        status = "READY" if primary_artifact else "PENDING_ARTIFACT"
        return {
            "status": status,
            "format": primary_artifact["format"] if primary_artifact else artifact_format,
            "artifact_uri": primary_artifact["uri"] if primary_artifact else "",
            "onnx_artifact_uri": onnx_artifact["uri"] if onnx_artifact else "",
            "manifest_uri": manifest_uri,
            "manifest": public_manifest,
            "validation_errors": validation_errors,
        }
    except Exception as exc:
        validation_errors.append(f"EXPORT_FAILED: {exc}")
        return {
            "status": "FAILED",
            "format": artifact_format,
            "artifact_uri": "",
            "onnx_artifact_uri": "",
            "manifest_uri": "",
            "manifest": {},
            "validation_errors": validation_errors,
        }


def _yolo_per_class_metrics(class_names: list[str], metrics: dict) -> dict:
    per_class = metrics.get("per_class_metrics")
    if isinstance(per_class, dict) and per_class:
        return _yolo_per_class_metrics_with_macro_average(
            {
                str(class_name): {
                    "precision": _metric_float(metric_values if isinstance(metric_values, dict) else {}, "precision", default=0.0),
                    "recall": _metric_float(metric_values if isinstance(metric_values, dict) else {}, "recall", default=0.0),
                    "AP50": _metric_float(metric_values if isinstance(metric_values, dict) else {}, "AP50", "mAP50", "map50", default=0.0),
                    "AP50_95": _metric_float(metric_values if isinstance(metric_values, dict) else {}, "AP50_95", "mAP50_95", "map50_95", default=0.0),
                }
                for class_name, metric_values in per_class.items()
                if str(class_name).strip() and str(class_name).lower() != "macro avg"
            }
        )
    return {}


def _yolo_per_class_metrics_with_macro_average(per_class: dict) -> dict:
    cleaned: dict[str, dict[str, float]] = {}
    for class_name, metric_values in per_class.items():
        if not isinstance(metric_values, dict):
            continue
        normalized = {
            "precision": round(_number(metric_values.get("precision")), 6),
            "recall": round(_number(metric_values.get("recall")), 6),
            "AP50": round(_number(metric_values.get("AP50")), 6),
            "AP50_95": round(_number(metric_values.get("AP50_95")), 6),
        }
        if any(value > 0 for value in normalized.values()):
            cleaned[str(class_name).strip()] = normalized
    values = list(cleaned.values())
    if values:
        cleaned["macro avg"] = {
            "precision": round(sum(row["precision"] for row in values) / len(values), 6),
            "recall": round(sum(row["recall"] for row in values) / len(values), 6),
            "AP50": round(sum(row["AP50"] for row in values) / len(values), 6),
            "AP50_95": round(sum(row["AP50_95"] for row in values) / len(values), 6),
        }
    return cleaned


def _detection_holistic_scores(
    *,
    map50_95: float,
    map50: float,
    precision: float,
    recall: float,
    box_loss: float,
    cls_loss: float,
    dfl_loss: float,
    estimated_cost_usd: float,
    runtime_seconds: float,
    model_profile: dict,
) -> dict:
    latency_ms = float(model_profile.get("estimated_pipeline_latency_ms") or model_profile.get("estimated_latency_ms") or 0)
    quality_score = (map50_95 * 0.62) + (map50 * 0.18) + (recall * 0.14) + (precision * 0.06)
    loss_total = box_loss + cls_loss + dfl_loss
    loss_score = max(0.0, min(1.0, 1.0 - loss_total / 3.0))
    latency_score = max(0.0, min(1.0, 1.0 - latency_ms / 160.0))
    cost_score = max(0.0, min(1.0, 1.0 - estimated_cost_usd / 10.0))
    runtime_score = max(0.0, min(1.0, 1.0 - runtime_seconds / 1800.0))
    overall_score = quality_score * 0.70 + loss_score * 0.12 + latency_score * 0.10 + cost_score * 0.05 + runtime_score * 0.03
    return {
        "quality_score": round(quality_score, 6),
        "loss_health_score": round(loss_score, 6),
        "latency_score": round(latency_score, 6),
        "cost_score": round(cost_score, 6),
        "runtime_score": round(runtime_score, 6),
        "overall_score": round(overall_score, 6),
        "detection_metrics": {
            "mAP50_95": round(map50_95, 6),
            "mAP50": round(map50, 6),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "box_loss": round(box_loss, 6),
            "cls_loss": round(cls_loss, 6),
            "dfl_loss": round(dfl_loss, 6),
        },
    }
