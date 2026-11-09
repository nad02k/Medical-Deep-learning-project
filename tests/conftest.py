"""Pytest fixtures for medimodal pipeline tests."""
import os
import shutil
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

# S'assurer que le secret de pseudonymisation est disponible pour les tests
os.environ["PSEUDO_SECRET"] = os.getenv("PSEUDO_SECRET", "test_secret_key_32_characters_long_for_hmac")

@pytest.fixture
def tmp_staging(tmp_path: Path) -> Path:
    staging = tmp_path / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    yield staging
    shutil.rmtree(staging, ignore_errors=True)

@pytest.fixture
def sample_nih_meta() -> pd.DataFrame:
    """Échantillon synthétique conforme au schéma NIH ChestX-ray14."""
    return pd.DataFrame({
        "image_name": ["00000001_000.png", "00000001_001.png", "00000002_000.png"],
        "patient_raw_id": [1, 1, 2],
        "follow_up_number": [0, 1, 0],
        "sex": ["M", "M", "F"],
        "view_position": ["PA", "AP", "PA"],
        "age_raw": ["058Y", "059Y", "042Y"],
        "finding_labels": ["Infiltration|Pneumonia", "Atelectasis", "No Finding"],
    })

@pytest.fixture
def sample_pneumonia_meta() -> pd.DataFrame:
    """Échantillon synthétique conforme au schéma Kaggle Pneumonia."""
    return pd.DataFrame({
        "image_name": ["train/PNEUMONIA/person1_bacteria_1.jpeg", "test/NORMAL/IM-0001-0001.jpeg"],
        "label": ["PNEUMONIA", "NORMAL"],
        "patient_key": ["person1", "IM-0001"],
        "label_detail": ["bacteria", "normal"],
        "orig_split": ["train", "test"],
        "key_inferred": [True, True],
    })

@pytest.fixture
def sample_ehr_patients() -> pd.DataFrame:
    """Échantillon synthétique conforme au schéma EHR patients."""
    return pd.DataFrame({
        "subject_id": [10001, 10002, 10003],
        "gender": ["M", "F", "M"],
        "anchor_age": [65, 45, 82],
        "anchor_year_group": ["2014 - 2016", "2017 - 2019", "2008 - 2010"],
        "dod": [None, None, "2115-05-12"],
    })

@pytest.fixture
def sample_ehr_labs() -> pd.DataFrame:
    """Échantillon synthétique conforme au schéma EHR labevents."""
    return pd.DataFrame({
        "labevent_id": [101, 102, 103, 104],
        "subject_id": [10001, 10001, 10002, 10003],
        "hadm_id": [2001, 2001, 2002, None],
        "itemid": [50912, 50971, 51221, 51301],
        "charttime": pd.to_datetime(["2020-01-01 10:00", "2020-01-01 10:30", "2020-01-02 08:00", "2020-01-03 12:00"]),
        "valuenum": [1.2, 4.5, 38.0, 7.8],
        "valueuom": ["mg/dL", "mEq/L", "%", "K/uL"],
        "flag": [None, None, None, "abnormal"],
    })
