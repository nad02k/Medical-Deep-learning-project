"""Tests de qualité des données (Data Quality) pour le pipeline DataOps.

Couvre les exigences minimales et avancées de la Tâche 3 :
1. Schémas et types pour chaque modalité (image, métadonnées, EHR)
2. Taux de valeurs manquantes, détection de doublons et valeurs aberrantes
3. Cohérence inter-modalités (intégrité référentielle, appariement)
4. Absence de fuite de données entre ensembles train / val / test (découpage par entité/patient)
5. Déterminisme et stabilité des splits
"""
import pytest
import pandas as pd
import numpy as np

from src import config
from src.anonymize import pseudo_id
from src.transform import assign_split, parse_age
from src.validate import validate_nih, validate_pneumonia, validate_ehr


class TestDataQualityNIH:
    def test_age_outlier_detection(self):
        """Vérifie que les âges aberrants (> MAX_AGE ou < 0) sont neutralisés (deviennent NaN)."""
        ages = pd.Series(["025Y", "110Y", "414Y", "-5Y", "invalid", "085Y"])
        parsed = parse_age(ages)
        assert parsed.iloc[0] == 25
        assert parsed.iloc[1] == 110
        assert pd.isna(parsed.iloc[2])  # 414Y > 110
        assert pd.isna(parsed.iloc[3])  # négatif
        assert pd.isna(parsed.iloc[4])  # invalide
        assert parsed.iloc[5] == 85

    def test_missing_image_files_warning(self, sample_nih_meta):
        """Vérifie que le contrôle signale les images référencées dans le CSV mais absentes sur disque."""
        actual_files = {"00000001_000.png"}  # 2 images manquent
        report = validate_nih(sample_nih_meta, actual_files, strict=False)
        assert report["rows_without_image"] == 2
        assert len(report["warnings"]) > 0

    def test_patient_sex_conflict_detection(self, sample_nih_meta):
        """Vérifie la détection d'incohérence : un même patient avec deux sexes déclarés."""
        conflict_df = sample_nih_meta.copy()
        # patient 1 a 2 lignes, mettons M sur l'une et F sur l'autre
        conflict_df.loc[1, "sex"] = "F"
        report = validate_nih(
            conflict_df, set(
                conflict_df["image_name"]), strict=False)
        assert report["patients_with_sex_conflict"] == 1
        assert any("sexe contradictoire" in c for c in report["critical"])


class TestDataLeakageAndSplits:
    def test_patient_level_split_no_leakage(self):
        """Vérifie qu'aucun patient ne se retrouve à la fois dans train et test/val."""
        patient_ids = [f"P_{i:04d}" for i in range(200)]
        splits = {p: assign_split(p) for p in patient_ids}

        # Simuler plusieurs images par patient
        records = []
        for p in patient_ids:
            for img_idx in range(3):
                records.append({"image_id": f"IMG_{p}_{img_idx}",
                               "patient_id": p, "split": splits[p]})
        df = pd.DataFrame(records)

        # Vérifier pour chaque patient que le split est unique (0 fuite)
        splits_per_patient = df.groupby("patient_id")["split"].nunique()
        assert (splits_per_patient == 1).all(
        ), "Fuite détectée : un patient a plusieurs splits !"

    def test_split_determinism(self):
        """Vérifie la reproductibilité stricte : un même patient a toujours le même split."""
        pid = "P_a1b2c3d4"
        split1 = assign_split(pid)
        split2 = assign_split(pid)
        assert split1 == split2
        assert split1 in ("train", "val", "test")

    def test_split_distribution_proportions(self):
        """Vérifie que la distribution statistique des splits respecte approximativement 70/15/15."""
        patients = [f"P_patient_{i}" for i in range(1000)]
        splits = [assign_split(p) for p in patients]
        counts = pd.Series(splits).value_counts(normalize=True)

        # Tolérance de +/- 5% pour 1000 échantillons aléatoires hashés
        assert 0.65 <= counts.get("train", 0) <= 0.75
        assert 0.10 <= counts.get("val", 0) <= 0.20
        assert 0.10 <= counts.get("test", 0) <= 0.20


class TestInterModalityCoherence:
    def test_ehr_orphan_labs_detected(
            self, sample_ehr_patients, sample_ehr_labs):
        """Vérifie que les examens de laboratoire sans patient correspondant sont détectés."""
        # Ajout d'un lab avec un subject_id inexistant
        corrupt_labs = sample_ehr_labs.copy()
        corrupt_labs.loc[len(corrupt_labs)] = {
            "labevent_id": 999,
            "subject_id": 99999,  # Inconnu
            "hadm_id": None,
            "itemid": 50912,
            "charttime": pd.to_datetime("2020-01-01"),
            "valuenum": 1.0,
            "valueuom": "mg/dL",
            "flag": None,
        }
        report = validate_ehr(sample_ehr_patients, corrupt_labs, strict=False)
        assert any("labos sans patient" in w for w in report["warnings"])

    def test_pseudonymization_irreversibility_and_consistency(self):
        """Vérifie que la pseudonymisation est déterministe avec le secret et sans collision triviale."""
        secret = "super_secure_secret_for_dataops"
        p1 = pseudo_id("patient_1001", secret, "P")
        p2 = pseudo_id("patient_1001", secret, "P")
        p3 = pseudo_id("patient_1002", secret, "P")

        assert p1 == p2, "La pseudonymisation doit être déterministe pour un même identifiant et secret"
        assert p1 != p3, "Deux identifiants différents doivent avoir des pseudonymes différents"
        assert p1.startswith("P_")
        assert len(p1) == 18  # P_ + 16 caractères hexadécimaux
