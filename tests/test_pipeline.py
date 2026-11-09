"""Tests d'intégration et unitaires pour les composants du pipeline DataOps."""
import pytest
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path

from src.anonymize import pseudo_id, reencode_image
from src.transform import assign_split, assemble
from src.validate import validate_pneumonia


def test_pseudo_id_requires_secret():
    """Vérifie que la pseudonymisation échoue expressément si aucun secret n'est fourni."""
    with pytest.raises(ValueError, match="PSEUDO_SECRET manquant"):
        pseudo_id("patient_1", secret="")


def test_reencode_image_strips_metadata(tmp_path: Path):
    """Vérifie qu'une image réencodée est convertie en niveaux de gris et stripped de métadonnées."""
    src_img_path = tmp_path / "raw.png"
    # Créer une image RGB avec métadonnées
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    img.save(src_img_path, "PNG")

    dst_img_path = tmp_path / "clean.png"
    info = reencode_image(src_img_path, dst_img_path)

    assert dst_img_path.exists()
    assert info["width"] == 100
    assert info["height"] == 100
    assert isinstance(info["sha256"], str) and len(info["sha256"]) == 64

    with Image.open(dst_img_path) as reloaded:
        assert reloaded.mode == "L", "L'image doit être convertie en niveaux de gris ('L')"


def test_validate_pneumonia_leakage_detection():
    """Vérifie la détection d'un patient présent dans plusieurs splits originaux Kaggle."""
    meta = pd.DataFrame({
        "image_name": ["img1.jpeg", "img2.jpeg"],
        "label": ["NORMAL", "PNEUMONIA"],
        "patient_key": ["patient_X", "patient_X"],  # Même patient
        "orig_split": ["train", "test"],             # Mais deux splits différents Kaggle !
        "key_inferred": [True, True],
    })
    report = validate_pneumonia(meta, strict=False)
    assert report["patients_in_multiple_original_splits"] == 1
    assert any("fuite" in w for w in report["warnings"])


def test_assemble_consistency():
    """Vérifie que l'assemblage génère les 5 tables attendues avec les colonnes correctes."""
    meta = pd.DataFrame({
        "image_id": ["I_1", "I_2"],
        "patient_pseudo_id": ["P_1", "P_2"],
        "view_position": ["PA", "AP"],
        "follow_up_number": [0, 1],
        "label_detail": [None, None],
        "sex": ["M", "F"],
        "age_years": [45.0, 60.0],
        "age_missing": [0, 0],
    })
    img = pd.DataFrame({
        "image_id": ["I_1", "I_2"],
        "object_key": ["images/I_1.png", "images/I_2.png"],
        "sha256": ["hash1", "hash2"],
        "width": [1024, 1024],
        "height": [1024, 1024],
    })
    labels = pd.DataFrame({
        "image_id": ["I_1", "I_2"],
        "pathology": ["Pneumonia", "Pneumonia"],
        "present": [1, 0],
    })

    tables = assemble(meta, img, labels, source="test_source")
    assert "patients" in tables
    assert "images" in tables
    assert "clinical_features" in tables
    assert "labels" in tables
    assert "splits" in tables

    assert len(tables["patients"]) == 2
    assert len(tables["images"]) == 2
    assert len(tables["splits"]) == 2
