"""Tests de validation des schémas de données avec Pandera."""
import pytest
import pandas as pd
import pandera.pandas as pa

from src.validate import nih_schema, pneumo_schema, ehr_patients_schema, ehr_labs_schema


def test_nih_schema_valid(sample_nih_meta):
    """Vérifie que des métadonnées NIH valides passent la validation de schéma."""
    validated = nih_schema.validate(sample_nih_meta)
    assert len(validated) == 3


def test_nih_schema_invalid_sex(sample_nih_meta):
    """Vérifie qu'une valeur invalide de sexe lève une erreur de validation."""
    invalid_df = sample_nih_meta.copy()
    invalid_df.loc[0, "sex"] = "UNKNOWN"
    with pytest.raises(pa.errors.SchemaError):
        nih_schema.validate(invalid_df)


def test_nih_schema_duplicate_image(sample_nih_meta):
    """Vérifie qu'un doublon de nom d'image est détecté."""
    invalid_df = sample_nih_meta.copy()
    invalid_df.loc[1, "image_name"] = invalid_df.loc[0, "image_name"]
    with pytest.raises(pa.errors.SchemaError):
        nih_schema.validate(invalid_df)


def test_pneumo_schema_valid(sample_pneumonia_meta):
    """Vérifie que des métadonnées Pneumonia valides passent la validation."""
    validated = pneumo_schema.validate(sample_pneumonia_meta)
    assert len(validated) == 2


def test_pneumo_schema_invalid_label(sample_pneumonia_meta):
    """Vérifie qu'un label hors (NORMAL, PNEUMONIA) est rejeté."""
    invalid_df = sample_pneumonia_meta.copy()
    invalid_df.loc[0, "label"] = "COVID"
    with pytest.raises(pa.errors.SchemaError):
        pneumo_schema.validate(invalid_df)


def test_ehr_patients_schema_valid(sample_ehr_patients):
    """Vérifie que les patients EHR valides passent le schéma."""
    validated = ehr_patients_schema.validate(sample_ehr_patients)
    assert len(validated) == 3


def test_ehr_patients_schema_invalid_age(sample_ehr_patients):
    """Vérifie qu'un âge hors borne [0, 120] est rejeté."""
    invalid_df = sample_ehr_patients.copy()
    invalid_df.loc[0, "anchor_age"] = 150
    with pytest.raises(pa.errors.SchemaError):
        ehr_patients_schema.validate(invalid_df)
