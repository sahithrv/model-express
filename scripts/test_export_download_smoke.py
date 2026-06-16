from __future__ import annotations

import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import export_download_smoke as smoke  # noqa: E402


class ExportDownloadSmokeTests(unittest.TestCase):
    def test_local_temp_smoke_creates_saved_zip_with_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "fixture"
            saved_dir = Path(temp_dir) / "saved"

            result = smoke.run_smoke(export_root=root, save_dir=saved_dir, verbose=False)

            self.assertTrue(all(check["ok"] for check in result["checks"]))
            self.assertEqual(result["project"]["id"], result["champion"]["project_id"])
            self.assertEqual(result["champion"]["id"], result["export_record"]["champion_id"])
            bundle = result["export_record"]["metadata"]["portable_inference_bundle"]
            self.assertEqual(bundle["schema_version"], "portable_inference_bundle_v1")
            self.assertEqual(bundle["status"], "created")

            saved_zip = Path(result["saved_zip"])
            self.assertTrue(saved_zip.exists())
            with zipfile.ZipFile(saved_zip) as archive:
                manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
            self.assertEqual(manifest["schema_version"], "portable_inference_manifest_v1")

    def test_resolve_artifact_location_rejects_remote_uri(self) -> None:
        with self.assertRaises(smoke.SmokeFailure):
            smoke.resolve_artifact_location({"artifact_uri": "https://example.invalid/portable_inference_bundle.zip"})

    def test_saved_zip_without_manifest_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = Path(temp_dir) / "missing_manifest.zip"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr("README.md", "not enough")

            with self.assertRaises(smoke.SmokeFailure):
                smoke.assert_saved_zip_has_manifest(zip_path)


if __name__ == "__main__":
    unittest.main()
