"""Multimodal alignment module — chest X-ray images ↔ EHR clinical data.

This module provides functions to align imaging data with structured
Electronic Health Record (EHR) data using patient pseudo-identifiers,
aggregate laboratory results into feature vectors, assess alignment quality,
and produce train/val/test splits for downstream multimodal learning.

Typical usage:
    >>> aligned = align_image_ehr(images_df, ehr_patients_df, ehr_labs_df)
    >>> quality = check_alignment_quality(aligned)
    >>> datasets = create_multimodal_dataset(aligned, splits_df)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────
# Key lab items we want to aggregate into the feature vector.
# MIMIC-IV itemids for common labs; extend as needed.
KEY_LAB_ITEMS: Dict[int, str] = {
    50912: "creatinine",
    50971: "potassium",
    50983: "sodium",
    50893: "calcium",
    51221: "hematocrit",
    51222: "hemoglobin",
    51301: "wbc",
    51265: "platelet",
    50882: "bicarbonate",
    50802: "base_excess",
    50818: "pco2",
    50821: "po2",
    51006: "bun",
    50862: "albumin",
    50885: "bilirubin_total",
    51003: "troponin_t",
}

# Aggregation statistics computed per lab item.
AGG_STATS = ("mean", "min", "max", "last")


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────


def align_image_ehr(
    images_df: pd.DataFrame,
    ehr_patients_df: pd.DataFrame,
    ehr_labs_df: pd.DataFrame,
    *,
    patient_id_col: str = "subject_id",
    study_id_col: str = "study_id",
    image_path_col: str = "image_path",
    time_window_hours: Optional[float] = 48.0,
) -> Dict[str, Any]:
    """Align chest X-ray images with EHR clinical data via patient pseudo IDs.

    Parameters
    ----------
    images_df : pd.DataFrame
        Image metadata.  Must contain at least *patient_id_col*,
        *study_id_col*, *image_path_col*, and optionally ``study_datetime``.
    ehr_patients_df : pd.DataFrame
        Patient demographics (one row per patient).  Must contain
        *patient_id_col*.
    ehr_labs_df : pd.DataFrame
        Laboratory results.  Must contain *patient_id_col*, ``itemid``,
        ``valuenum``, and optionally ``charttime``.
    patient_id_col : str
        Column name for the patient pseudo-identifier.
    study_id_col : str
        Column name for the imaging study identifier.
    image_path_col : str
        Column name for the image file path / DICOM reference.
    time_window_hours : float | None
        If set, only lab results within ±*time_window_hours* of the study
        datetime are considered.  ``None`` disables temporal filtering.

    Returns
    -------
    dict
        ``aligned_pairs`` – list of dicts, each with image ref + EHR features.
        ``unmatched_images`` – DataFrame of images with no EHR counterpart.
        ``unmatched_ehr`` – DataFrame of EHR patients with no images.
    """
    logger.info(
        "Aligning %d images with %d patients and %d lab records",
        len(images_df),
        len(ehr_patients_df),
        len(ehr_labs_df),
    )

    # --- 1. Identify patient overlap --------------------------------
    image_patient_ids = set(images_df[patient_id_col].unique())
    ehr_patient_ids = set(ehr_patients_df[patient_id_col].unique())
    common_ids = image_patient_ids & ehr_patient_ids

    logger.info(
        "Patient overlap: %d (images-only=%d, EHR-only=%d)",
        len(common_ids),
        len(image_patient_ids - common_ids),
        len(ehr_patient_ids - common_ids),
    )

    # --- 2. Build aligned pairs -------------------------------------
    aligned_pairs: List[Dict[str, Any]] = []

    # Pre-filter labs to only matched patients for speed.
    matched_labs = ehr_labs_df[ehr_labs_df[patient_id_col].isin(common_ids)]

    for pid in sorted(common_ids):
        patient_images = images_df[images_df[patient_id_col] == pid]
        patient_labs = matched_labs[matched_labs[patient_id_col] == pid]
        patient_demo = ehr_patients_df[ehr_patients_df[patient_id_col] == pid]

        # Aggregate labs into feature vector.
        lab_features = aggregate_labs_for_patient(patient_labs, pid)

        # Demographics row (take first if duplicates).
        demo_dict = (
            patient_demo.iloc[0].to_dict() if len(patient_demo) > 0 else {}
        )

        for _, img_row in patient_images.iterrows():
            # Optional temporal filtering.
            study_labs = patient_labs
            if (
                time_window_hours is not None
                and "study_datetime" in img_row.index
                and "charttime" in patient_labs.columns
            ):
                study_dt = pd.to_datetime(img_row["study_datetime"])
                delta = pd.Timedelta(hours=time_window_hours)
                charttimes = pd.to_datetime(patient_labs["charttime"])
                mask = (charttimes >= study_dt - delta) & (
                    charttimes <= study_dt + delta
                )
                study_labs = patient_labs[mask]
                # Re-aggregate with temporally-scoped labs.
                lab_features = aggregate_labs_for_patient(study_labs, pid)

            aligned_pairs.append(
                {
                    patient_id_col: pid,
                    study_id_col: img_row.get(study_id_col),
                    image_path_col: img_row[image_path_col],
                    "demographics": demo_dict,
                    "lab_features": lab_features,
                    "labels": img_row.get("labels", None),
                }
            )

    # --- 3. Identify unmatched records ------------------------------
    unmatched_images = images_df[
        ~images_df[patient_id_col].isin(common_ids)
    ].copy()
    unmatched_ehr = ehr_patients_df[
        ~ehr_patients_df[patient_id_col].isin(common_ids)
    ].copy()

    result = {
        "aligned_pairs": aligned_pairs,
        "unmatched_images": unmatched_images,
        "unmatched_ehr": unmatched_ehr,
    }

    logger.info(
        "Alignment complete: %d aligned pairs, %d unmatched images, %d unmatched EHR",
        len(aligned_pairs),
        len(unmatched_images),
        len(unmatched_ehr),
    )

    return result


def aggregate_labs_for_patient(
    labs_df: pd.DataFrame,
    patient_id: Any,
    *,
    itemid_col: str = "itemid",
    value_col: str = "valuenum",
    time_col: str = "charttime",
) -> Dict[str, float]:
    """Aggregate lab results for a single patient into a flat feature vector.

    For each key lab item (see :data:`KEY_LAB_ITEMS`) that appears in
    *labs_df*, the function computes the **mean**, **min**, **max**, and
    **last** observed value.  "Last" is determined by *time_col* when
    present; otherwise it falls back to the last row in DataFrame order.

    Parameters
    ----------
    labs_df : pd.DataFrame
        Laboratory records for **one patient**.
    patient_id : Any
        Patient identifier (used only for logging).
    itemid_col : str
        Column containing the lab item identifier.
    value_col : str
        Column containing the numeric result.
    time_col : str
        Column containing the chart timestamp (used for "last" value).

    Returns
    -------
    dict
        Flat mapping ``{"{lab_name}_{stat}": float}``, e.g.
        ``{"creatinine_mean": 1.1, "creatinine_last": 1.3, …}``.
    """
    features: Dict[str, float] = {}

    if labs_df.empty:
        logger.debug("No lab data for patient %s", patient_id)
        return features

    for itemid, lab_name in KEY_LAB_ITEMS.items():
        item_rows = labs_df[labs_df[itemid_col] == itemid]
        if item_rows.empty:
            continue

        values = item_rows[value_col].dropna()
        if values.empty:
            continue

        features[f"{lab_name}_mean"] = float(values.mean())
        features[f"{lab_name}_min"] = float(values.min())
        features[f"{lab_name}_max"] = float(values.max())

        # "Last" value: by time if available, else last row.
        if time_col in item_rows.columns:
            try:
                sorted_rows = item_rows.sort_values(time_col)
                last_val = sorted_rows[value_col].dropna().iloc[-1]
            except (IndexError, KeyError):
                last_val = values.iloc[-1]
        else:
            last_val = values.iloc[-1]

        features[f"{lab_name}_last"] = float(last_val)

    return features


def check_alignment_quality(aligned: Dict[str, Any]) -> Dict[str, Any]:
    """Report alignment statistics and data quality metrics.

    Parameters
    ----------
    aligned : dict
        Output of :func:`align_image_ehr`.

    Returns
    -------
    dict
        ``total_aligned``        – number of aligned image–EHR pairs.
        ``total_unmatched_img``  – images without EHR match.
        ``total_unmatched_ehr``  – EHR patients without images.
        ``match_rate_images``    – fraction of images matched (0-1).
        ``match_rate_ehr``       – fraction of EHR patients matched (0-1).
        ``lab_feature_coverage`` – fraction of aligned pairs that have ≥ 1 lab
                                   feature populated.
        ``avg_lab_features``     – mean number of lab features per aligned pair.
        ``missing_labels_pct``   – fraction of aligned pairs with missing labels.
    """
    pairs = aligned["aligned_pairs"]
    n_matched = len(pairs)
    n_unmatched_img = len(aligned["unmatched_images"])
    n_unmatched_ehr = len(aligned["unmatched_ehr"])

    total_images = n_matched + n_unmatched_img
    total_ehr = (
        n_matched + n_unmatched_ehr
    )  # approximation; unique patients may differ

    # Lab feature coverage.
    lab_counts = [len(p.get("lab_features", {})) for p in pairs]
    has_labs = sum(1 for c in lab_counts if c > 0)

    # Missing labels.
    missing_labels = sum(1 for p in pairs if p.get("labels") is None)

    report = {
        "total_aligned": n_matched,
        "total_unmatched_img": n_unmatched_img,
        "total_unmatched_ehr": n_unmatched_ehr,
        "match_rate_images": n_matched / total_images if total_images else 0.0,
        "match_rate_ehr": n_matched / total_ehr if total_ehr else 0.0,
        "lab_feature_coverage": has_labs / n_matched if n_matched else 0.0,
        "avg_lab_features": float(np.mean(lab_counts)) if lab_counts else 0.0,
        "missing_labels_pct": missing_labels / n_matched if n_matched else 0.0,
    }

    logger.info("Alignment quality report: %s", report)
    return report


def create_multimodal_dataset(
    aligned: Dict[str, Any],
    splits_df: pd.DataFrame,
    *,
    patient_id_col: str = "subject_id",
    split_col: str = "split",
    image_path_col: str = "image_path",
) -> Dict[str, pd.DataFrame]:
    """Create train / val / test DataFrames with aligned image + EHR features.

    Parameters
    ----------
    aligned : dict
        Output of :func:`align_image_ehr`.
    splits_df : pd.DataFrame
        Mapping of patient IDs to split names.  Must contain
        *patient_id_col* and *split_col* (values: ``"train"``,
        ``"val"``, ``"test"``).
    patient_id_col : str
        Column name for patient identifier.
    split_col : str
        Column name indicating the dataset split.
    image_path_col : str
        Column name for the image path field in the aligned pairs.

    Returns
    -------
    dict
        ``{"train": DataFrame, "val": DataFrame, "test": DataFrame}``
        Each DataFrame contains columns for image path, flattened lab
        features, demographic features, and labels.
    """
    pairs = aligned["aligned_pairs"]
    if not pairs:
        logger.warning("No aligned pairs — returning empty splits.")
        return {"train": pd.DataFrame(), "val": pd.DataFrame(), "test": pd.DataFrame()}

    # Flatten aligned pairs into a single DataFrame.
    rows: List[Dict[str, Any]] = []
    for p in pairs:
        row: Dict[str, Any] = {
            patient_id_col: p[patient_id_col],
            image_path_col: p.get(image_path_col),
            "study_id": p.get("study_id"),
            "labels": p.get("labels"),
        }
        # Flatten demographics.
        for k, v in p.get("demographics", {}).items():
            row[f"demo_{k}"] = v
        # Flatten lab features.
        for k, v in p.get("lab_features", {}).items():
            row[f"lab_{k}"] = v
        rows.append(row)

    records_df = pd.DataFrame(rows)

    # Merge with split information.
    records_df = records_df.merge(
        splits_df[[patient_id_col, split_col]],
        on=patient_id_col,
        how="left",
    )

    # Patients without a split assignment go to "train" by default.
    records_df[split_col] = records_df[split_col].fillna("train")

    result: Dict[str, pd.DataFrame] = {}
    for split_name in ("train", "val", "test"):
        split_data = records_df[records_df[split_col] == split_name].copy()
        result[split_name] = split_data.reset_index(drop=True)
        logger.info("Split '%s': %d samples", split_name, len(split_data))

    return result
