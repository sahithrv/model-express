from __future__ import annotations

import contextvars
import os
from pathlib import PurePosixPath


for _thread_key, _thread_value in {
    "OMP_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
    "OPENBLAS_NUM_THREADS": "4",
    "NUMEXPR_NUM_THREADS": "4",
    "TOKENIZERS_PARALLELISM": "false",
}.items():
    os.environ.setdefault(_thread_key, _thread_value)

try:
    import modal
except Exception:  # pragma: no cover - local helper tests can run without Modal installed.
    modal = None


def _modal_remote_path_env(name: str, default: str) -> PurePosixPath:
    value = os.getenv(name, default).strip() or default
    path = PurePosixPath(value.replace("\\", "/"))
    if not path.is_absolute():
        raise ValueError(f"{name} must be an absolute POSIX path for Modal, got {value!r}.")
    if str(path) in {"/", "/root", "/tmp"}:
        raise ValueError(f"{name} cannot be mounted at {path}.")
    return path


APP_NAME = "model-express-training"
DEFAULT_GPU = os.getenv("MODAL_GPU_TYPE", "T4")
DATASET_MATERIALIZATION_ROOT = _modal_remote_path_env(
    "MODEL_EXPRESS_MODAL_DATASET_CACHE_ROOT",
    "/cache/model-express/datasets",
)
DATASET_VOLUME_NAME = os.getenv(
    "MODEL_EXPRESS_MODAL_DATASET_CACHE_VOLUME",
    "model-express-dataset-cache",
)
TORCH_CACHE_ROOT = _modal_remote_path_env(
    "MODEL_EXPRESS_MODAL_TORCH_CACHE_ROOT",
    "/cache/model-express/torch",
)
TORCH_CACHE_VOLUME_NAME = os.getenv(
    "MODEL_EXPRESS_MODAL_TORCH_CACHE_VOLUME",
    "model-express-torch-cache",
)
DEFAULT_ORCHESTRATOR_REPORT_TIMEOUT_SECONDS = 300
DEFAULT_MODAL_DATASET_MATERIALIZATION_TIMEOUT_SECONDS = 60 * 60
DEFAULT_MODAL_DATASET_PROFILE_TIMEOUT_SECONDS = 60 * 60
DEFAULT_MODAL_TRAINING_SCALEDOWN_WINDOW_SECONDS = 10 * 60
DEFAULT_COST_SENSITIVE_MODAL_TRAINING_SCALEDOWN_WINDOW_SECONDS = 120
DEFAULT_MODAL_METADATA_DATALOADER_WORKERS = 4
DEFAULT_MODAL_IMAGEFOLDER_DATALOADER_WORKERS = 2
DEFAULT_MODAL_TRAINING_DATASET_CACHE_ROOT = "/tmp/model-express/training-datasets"
DEFAULT_METADATA_BUNDLE_PAGE_SIZE = 5000
DEFAULT_METADATA_BUNDLE_MAX_RECORDS = 50_000
_MODAL_STAGE_EVENTS: contextvars.ContextVar[list[dict] | None] = contextvars.ContextVar(
    "model_express_modal_stage_events",
    default=None,
)
METADATA_ENDPOINT_UNAVAILABLE_STATUS_CODES = {404, 405, 501}
COMMON_IMAGE_ROOT_NAMES = ("images", "image", "imgs", "img", "JPEGImages", "jpegimages", "data")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ROOT_METADATA_DIR_NAMES = {
    "annotation",
    "annotations",
    "attribute",
    "attributes",
    "bbox",
    "bboxes",
    "boxes",
    "keypoint",
    "keypoints",
    "label",
    "labels",
    "landmark",
    "landmarks",
    "manifest",
    "manifests",
    "meta",
    "metadata",
    "part",
    "parts",
    "split",
    "splits",
}

EFFECTIVE_NUMBER_CLASS_BALANCING = {
    "effective_number",
    "effective_number_loss",
    "effective_number_class_balanced_loss",
    "class_balanced_loss",
    "class_balanced_effective_number",
}
if modal is not None:
    try:
        dataset_volume = modal.Volume.from_name(DATASET_VOLUME_NAME, create_if_missing=True)
        dataset_volume_mounts = {str(DATASET_MATERIALIZATION_ROOT): dataset_volume}
    except Exception:  # pragma: no cover - depends on Modal runtime/account setup.
        dataset_volume = None
        dataset_volume_mounts = {}
    try:
        torch_cache_volume = modal.Volume.from_name(TORCH_CACHE_VOLUME_NAME, create_if_missing=True)
        torch_cache_volume_mounts = {str(TORCH_CACHE_ROOT): torch_cache_volume}
    except Exception:  # pragma: no cover - depends on Modal runtime/account setup.
        torch_cache_volume = None
        torch_cache_volume_mounts = {}
    training_volume_mounts = dict(torch_cache_volume_mounts)
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("libglib2.0-0", "libgl1")
        .pip_install(
            "boto3",
            "numpy",
            "pillow",
            "requests",
            "scikit-learn",
            "onnx",
            "onnxscript",
            "pyyaml",
            "torch",
            "torchvision",
            "ultralytics",
        )
        .env({"TORCH_HOME": str(TORCH_CACHE_ROOT)})
        .add_local_python_source("worker")
    )
    app = modal.App(APP_NAME)
else:
    image = None
    dataset_volume = None
    dataset_volume_mounts = {}
    torch_cache_volume = None
    torch_cache_volume_mounts = {}
    training_volume_mounts = {}

    class _UnavailableModalApp:
        def function(self, *args, **kwargs):
            def decorator(func):
                return func

            return decorator

    app = _UnavailableModalApp()


def _positive_int_env(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _modal_dataset_materialization_timeout_seconds() -> int:
    return _positive_int_env(
        "MODEL_EXPRESS_MODAL_MATERIALIZATION_TIMEOUT_SECONDS",
        DEFAULT_MODAL_DATASET_MATERIALIZATION_TIMEOUT_SECONDS,
    )


def _modal_dataset_profile_timeout_seconds() -> int:
    return _positive_int_env(
        "MODEL_EXPRESS_MODAL_PROFILE_TIMEOUT_SECONDS",
        DEFAULT_MODAL_DATASET_PROFILE_TIMEOUT_SECONDS,
    )


def _optional_positive_int_env(name: str, default: int = 0) -> int | None:
    value = os.getenv(name, "").strip()
    if not value:
        return default if default > 0 else None
    try:
        parsed = int(value)
    except ValueError:
        return default if default > 0 else None
    return parsed if parsed > 0 else None


def _modal_training_min_containers() -> int | None:
    return _optional_positive_int_env("MODEL_EXPRESS_MODAL_TRAIN_MIN_CONTAINERS")


def _modal_training_buffer_containers() -> int | None:
    return _optional_positive_int_env("MODEL_EXPRESS_MODAL_TRAIN_BUFFER_CONTAINERS")


def _modal_training_scaledown_window_seconds() -> int | None:
    return _optional_positive_int_env(
        "MODEL_EXPRESS_MODAL_TRAIN_SCALEDOWN_WINDOW_SECONDS",
        DEFAULT_COST_SENSITIVE_MODAL_TRAINING_SCALEDOWN_WINDOW_SECONDS
        if _modal_cost_sensitive_defaults_enabled()
        else DEFAULT_MODAL_TRAINING_SCALEDOWN_WINDOW_SECONDS,
    )


def _modal_cost_sensitive_defaults_enabled() -> bool:
    return _bool(os.getenv("MODEL_EXPRESS_MODAL_COST_SENSITIVE_DEFAULTS"), default=False)
