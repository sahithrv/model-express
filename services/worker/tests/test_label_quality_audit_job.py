from __future__ import annotations

import unittest

from worker.datasets.label_quality import build_label_quality_profile_patch
from worker.jobs import run_job


class FakeAuditClient:
    def __init__(self) -> None:
        self.updated_profiles: list[tuple[str, dict]] = []
        self.completed_jobs: list[str] = []

    def get_dataset(self, dataset_id: str) -> dict:
        return {
            "id": dataset_id,
            "profile": {
                "class_count": 2,
                "total_images": 11,
                "class_distribution": {"cat": 10, "dog": 1},
                "imbalance_ratio": 10.0,
                "leakage_warnings": ["duplicate image filename(s) detected"],
                "visual_exemplars": [{"path": "dog/one.jpg", "class_name": "dog"}],
                "artifacts": [],
            },
        }

    def update_dataset_profile(self, dataset_id: str, profile: dict) -> dict:
        self.updated_profiles.append((dataset_id, profile))
        return profile

    def complete_job(self, job_id: str, mlflow_run_id: str = "") -> dict:
        self.completed_jobs.append(job_id)
        return {"id": job_id}


class LabelQualityAuditJobTests(unittest.TestCase):
    def test_build_label_quality_profile_patch_is_report_only(self) -> None:
        profile = {"class_distribution": {"cat": 10, "dog": 1}, "artifacts": []}

        patched = build_label_quality_profile_patch(
            {
                "audit_type": "label_noise_audit",
                "report_only": True,
                "dataset_id": "dataset-1",
                "mechanism": "label_noise_audit",
                "evidence_used": ["minority class failure"],
            },
            profile,
        )

        self.assertTrue(patched["label_quality_audit"]["report_only"])
        self.assertEqual(patched["label_quality_audit"]["audit_type"], "label_noise_audit")
        self.assertEqual(patched["artifacts"][-1]["artifact_type"], "label_quality_audit")
        self.assertIn("label_quality_audits", patched)

    def test_label_quality_audit_job_updates_profile_and_completes(self) -> None:
        client = FakeAuditClient()
        job = {
            "id": "job-1",
            "template": "label_quality_audit",
            "config": {
                "audit_type": "hard_example_audit",
                "report_only": True,
                "dataset_id": "dataset-1",
                "plan_id": "plan-1",
                "experiment_index": 2,
                "mechanism": "hard_example_audit",
                "intervention": "Review hard examples",
                "expected_effect": "Identify suspicious samples without mutation",
            },
        }

        run_job(client, job)

        self.assertEqual(client.completed_jobs, ["job-1"])
        self.assertEqual(client.updated_profiles[0][0], "dataset-1")
        audit = client.updated_profiles[0][1]["label_quality_audit"]
        self.assertEqual(audit["audit_type"], "hard_example_audit")
        self.assertTrue(audit["report_only"])
        self.assertGreaterEqual(len(audit["findings"]), 1)


if __name__ == "__main__":
    unittest.main()
