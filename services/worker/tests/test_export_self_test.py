from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from worker.exporting.artifacts import produce_champion_export_artifacts
from worker.exporting.metadata import build_champion_export_metadata
from worker.exporting.preprocessing import image_to_chw_float32_array, prepare_image_for_inference
from worker.exporting.self_test import class_label_order_hash, run_onnx_pytorch_self_test
from worker.training.preprocessing_registry import build_image_transform


class ExportSelfTestTests(unittest.TestCase):
    def test_train_eval_transform_matches_export_preprocessing_contract(self) -> None:
        try:
            import numpy as np
            import torch
            import torchvision  # noqa: F401
        except Exception as exc:  # pragma: no cover - depends on optional local deps
            raise unittest.SkipTest(f"torchvision preprocessing deps are unavailable: {exc}") from exc

        image = Image.new("RGB", (8, 4), (255, 0, 0))
        preprocessing = {
            "resize_strategy": "preserve_aspect_pad",
            "crop_strategy": "none",
            "normalization": "none",
        }
        metadata = build_champion_export_metadata(
            model_name="tiny_linear",
            class_names=["cat", "dog"],
            image_size=8,
            preprocessing=preprocessing,
        )
        metadata["input_shape"] = [1, 3, 8, 8]

        eval_tensor = build_image_transform(8, {}, preprocessing, training=False)(image)
        prepared = prepare_image_for_inference(Image, image, metadata)
        export_tensor = image_to_chw_float32_array(prepared, metadata)

        self.assertEqual(list(eval_tensor.shape), [3, 8, 8])
        np.testing.assert_allclose(eval_tensor.detach().cpu().numpy(), export_tensor, atol=0.0)
        self.assertIsInstance(eval_tensor, torch.Tensor)

    def test_onnx_export_self_test_records_passed_manifest_for_heldout_sample(self) -> None:
        try:
            import onnxruntime  # noqa: F401
            import onnxscript  # noqa: F401
            import torch
        except Exception as exc:  # pragma: no cover - depends on optional local deps
            raise unittest.SkipTest(f"ONNX export deps are unavailable: {exc}") from exc

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "dog.png"
            Image.new("RGB", (4, 4), (40, 40, 40)).save(image_path)
            preprocessing = {"resize_strategy": "squash", "normalization": "none"}
            metadata = build_champion_export_metadata(
                model_name="tiny_linear",
                class_names=["cat", "dog"],
                image_size=4,
                preprocessing=preprocessing,
            )
            metadata["input_shape"] = [1, 3, 4, 4]
            prepared = prepare_image_for_inference(Image, Image.open(image_path), metadata)
            input_tensor = torch.tensor(image_to_chw_float32_array(prepared, metadata)).unsqueeze(0)
            model = _tiny_linear_model(torch)
            with torch.no_grad():
                logits = model(input_tensor)[0].tolist()

            manifest = produce_champion_export_artifacts(
                export_dir=root / "export",
                model_name="tiny_linear",
                class_names=["cat", "dog"],
                image_size=4,
                model=model,
                preprocessing=preprocessing,
                formats=("onnx",),
                export_self_test_samples=[
                    {
                        "sample_id": "heldout:dog",
                        "image_id": "dog.png",
                        "image_path": str(image_path),
                        "split": "test",
                        "true_index": 1,
                        "true_label": "dog",
                        "predicted_index": 1,
                        "predicted_label": "dog",
                        "class_label_order_hash": class_label_order_hash(["cat", "dog"]),
                        "input_tensor": input_tensor,
                        "pytorch_logits": logits,
                        "image_metadata": {
                            "demo_source_type": "heldout_test_original_bytes",
                            "parity_safe": True,
                        },
                    }
                ],
            )

        self_test = manifest["metadata"]["export_self_test"]
        self.assertEqual(manifest["status"], "created")
        self.assertEqual(self_test["status"], "passed")
        self.assertTrue(self_test["export_verified"])
        self.assertEqual(self_test["sample_count"], 1)
        sample = self_test["samples"][0]
        self.assertEqual(sample["true_label"], "dog")
        self.assertEqual(sample["pytorch"]["predicted_label"], "dog")
        self.assertEqual(sample["onnx"]["predicted_label"], "dog")
        self.assertTrue(sample["comparison"]["top_k_labels_match"])
        self.assertTrue(sample["input_tensor"]["sha256"])

    def test_onnx_self_test_fails_loudly_on_label_order_mismatch(self) -> None:
        torch, model, tensor, logits = _self_test_fixture()
        metadata = _self_test_metadata()

        with tempfile.TemporaryDirectory() as temp_dir:
            onnx_path = Path(temp_dir) / "model.onnx"
            onnx_path.write_bytes(b"fake")
            with patch.dict("sys.modules", {"torch": torch}):
                with patch("worker.exporting.self_test._load_onnx_session", return_value=object()):
                    with patch("worker.exporting.self_test._run_onnx_logits", return_value=logits):
                        result = run_onnx_pytorch_self_test(
                            model=model,
                            onnx_path=onnx_path,
                            metadata=metadata,
                            samples=[
                                {
                                    "input_tensor": tensor,
                                    "pytorch_logits": logits,
                                    "true_index": 1,
                                    "true_label": "dog",
                                    "predicted_index": 1,
                                    "predicted_label": "dog",
                                    "class_label_order_hash": class_label_order_hash(["dog", "cat"]),
                                }
                            ],
                        )

        self.assertEqual(result["status"], "failed")
        self.assertTrue(_diagnostic_code_present(result, "LABEL_MAP_MISMATCH"))
        self.assertIsNotNone(torch)

    def test_onnx_self_test_fails_on_output_mismatch(self) -> None:
        _, model, tensor, logits = _self_test_fixture()
        metadata = _self_test_metadata()

        with tempfile.TemporaryDirectory() as temp_dir:
            onnx_path = Path(temp_dir) / "model.onnx"
            onnx_path.write_bytes(b"fake")
            with patch.dict("sys.modules", {"torch": _FakeTorch()}):
                with patch("worker.exporting.self_test._load_onnx_session", return_value=object()):
                    with patch("worker.exporting.self_test._run_onnx_logits", return_value=[logits[1], logits[0]]):
                        result = run_onnx_pytorch_self_test(
                            model=model,
                            onnx_path=onnx_path,
                            metadata=metadata,
                            samples=[
                                {
                                    "input_tensor": tensor,
                                    "pytorch_logits": logits,
                                    "true_index": 1,
                                    "true_label": "dog",
                                    "predicted_index": 1,
                                    "predicted_label": "dog",
                                    "class_label_order_hash": class_label_order_hash(["cat", "dog"]),
                                }
                            ],
                        )

        self.assertEqual(result["status"], "failed")
        self.assertTrue(_diagnostic_code_present(result, "ONNX_OUTPUT_MISMATCH"))
        self.assertTrue(_diagnostic_code_present(result, "PREDICTED_LABEL_MISMATCH"))


def _self_test_metadata() -> dict:
    metadata = build_champion_export_metadata(
        model_name="tiny_linear",
        class_names=["cat", "dog"],
        image_size=4,
        preprocessing={"normalization": "none"},
    )
    metadata["input_shape"] = [1, 3, 4, 4]
    return metadata


def _self_test_fixture():
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover - depends on optional local deps
        raise unittest.SkipTest(f"numpy is unavailable: {exc}") from exc

    torch = _FakeTorch()
    logits = [0.25, 1.25]
    model = _FakeModel(logits)
    tensor = _FakeTensor(np.zeros((1, 3, 4, 4), dtype=np.float32))
    return torch, model, tensor, logits


def _tiny_linear_model(torch):
    from torch import nn

    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 2))
    with torch.no_grad():
        model[1].weight.zero_()
        model[1].bias[:] = torch.tensor([0.25, 1.25])
    return model


class _FakeTorch:
    float32 = "float32"

    def tensor(self, value, dtype=None):
        return _FakeTensor(value)

    def no_grad(self):
        return _FakeNoGrad()


class _FakeNoGrad:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeModel:
    def __init__(self, logits):
        self._logits = logits

    def eval(self):
        return self

    def cpu(self):
        return self

    def __call__(self, tensor):
        return _FakeTensor([self._logits])


class _FakeTensor:
    def __init__(self, value):
        import numpy as np

        self._array = np.asarray(value, dtype=np.float32)

    @property
    def ndim(self):
        return self._array.ndim

    @property
    def shape(self):
        return self._array.shape

    def detach(self):
        return self

    def cpu(self):
        return self

    def float(self):
        return self

    def unsqueeze(self, axis):
        import numpy as np

        return _FakeTensor(np.expand_dims(self._array, axis))

    def numpy(self):
        return self._array

    def reshape(self, *shape):
        return _FakeTensor(self._array.reshape(*shape))

    def tolist(self):
        return self._array.tolist()

    def __getitem__(self, item):
        return _FakeTensor(self._array[item])


def _diagnostic_code_present(result: dict, code: str) -> bool:
    return any(item.get("code") == code for item in result.get("diagnostics", []) if isinstance(item, dict))


if __name__ == "__main__":
    unittest.main()
