"""Test runner capable of executing test functions with pytest or standard unittest."""
import sys
import unittest
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
os.environ["PSEUDO_SECRET"] = os.getenv("PSEUDO_SECRET", "test_secret_key_32_characters_long_for_hmac")

from src.transform import assign_split, parse_age, assemble
from src.anonymize import pseudo_id, reencode_image
from src.validate import validate_nih, validate_pneumonia, validate_ehr, nih_schema, pneumo_schema, ehr_patients_schema
import pandera.pandas as pa
from PIL import Image


class TestDataQuality(unittest.TestCase):
    def test_age_outlier_detection(self):
        ages = pd.Series(["025Y", "110Y", "414Y", "-5Y", "invalid", "085Y"])
        parsed = parse_age(ages)
        self.assertEqual(parsed.iloc[0], 25)
        self.assertEqual(parsed.iloc[1], 110)
        self.assertTrue(pd.isna(parsed.iloc[2]))
        self.assertTrue(pd.isna(parsed.iloc[3]))
        self.assertTrue(pd.isna(parsed.iloc[4]))
        self.assertEqual(parsed.iloc[5], 85)

    def test_patient_level_split_no_leakage(self):
        patient_ids = [f"P_{i:04d}" for i in range(200)]
        splits = {p: assign_split(p) for p in patient_ids}
        records = []
        for p in patient_ids:
            for img_idx in range(3):
                records.append({"image_id": f"IMG_{p}_{img_idx}", "patient_id": p, "split": splits[p]})
        df = pd.DataFrame(records)
        splits_per_patient = df.groupby("patient_id")["split"].nunique()
        self.assertTrue((splits_per_patient == 1).all())

    def test_split_determinism(self):
        pid = "P_a1b2c3d4"
        self.assertEqual(assign_split(pid), assign_split(pid))
        self.assertIn(assign_split(pid), ("train", "val", "test"))

    def test_split_proportions(self):
        patients = [f"P_patient_{i}" for i in range(1000)]
        splits = [assign_split(p) for p in patients]
        counts = pd.Series(splits).value_counts(normalize=True)
        self.assertTrue(0.65 <= counts.get("train", 0) <= 0.75)
        self.assertTrue(0.10 <= counts.get("val", 0) <= 0.20)
        self.assertTrue(0.10 <= counts.get("test", 0) <= 0.20)

    def test_pseudonymization_consistency(self):
        secret = "test_secret_for_dataops"
        p1 = pseudo_id("patient_1001", secret, "P")
        p2 = pseudo_id("patient_1001", secret, "P")
        p3 = pseudo_id("patient_1002", secret, "P")
        self.assertEqual(p1, p2)
        self.assertNotEqual(p1, p3)
        self.assertTrue(p1.startswith("P_"))

    def test_pseudo_id_requires_secret(self):
        with self.assertRaises(ValueError):
            pseudo_id("patient_1", secret="")

    def test_nih_schema(self):
        df = pd.DataFrame({
            "image_name": ["00000001_000.png", "00000001_001.png"],
            "patient_raw_id": [1, 1],
            "follow_up_number": [0, 1],
            "sex": ["M", "M"],
            "view_position": ["PA", "AP"],
        })
        validated = nih_schema.validate(df)
        self.assertEqual(len(validated), 2)

    def test_pneumonia_leakage_warning(self):
        meta = pd.DataFrame({
            "image_name": ["img1.jpeg", "img2.jpeg"],
            "label": ["NORMAL", "PNEUMONIA"],
            "patient_key": ["patient_X", "patient_X"],
            "orig_split": ["train", "test"],
            "key_inferred": [True, True],
        })
        report = validate_pneumonia(meta, strict=False)
        self.assertEqual(report["patients_in_multiple_original_splits"], 1)

    def test_reencode_image(self):
        tmp_dir = ROOT / "data" / "staging"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        src = tmp_dir / "test_raw.png"
        dst = tmp_dir / "test_clean.png"
        Image.new("RGB", (64, 64), color="blue").save(src)
        info = reencode_image(src, dst)
        self.assertEqual(info["width"], 64)
        self.assertEqual(info["height"], 64)
        with Image.open(dst) as im:
            self.assertEqual(im.mode, "L")
        src.unlink(missing_ok=True)
        dst.unlink(missing_ok=True)


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDataQuality)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Écrit le résultat pour le tracking DVC
    import json
    from datetime import datetime, timezone
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "ok": result.wasSuccessful(),
    }
    (reports_dir / "quality_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    
    sys.exit(0 if result.wasSuccessful() else 1)
