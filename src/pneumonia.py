"""Kaggle Chest X-Ray Pneumonia : images + label uniquement (pas de clinique).
Structure attendue : .../{train,val,test}/{NORMAL,PNEUMONIA}/*.jpeg"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .anonymize import pseudo_id
from .transform import assemble, stage_images

SOURCE = "kaggle_pneumonia"
IMG_EXT = {".jpeg", ".jpg", ".png"}


def find_inputs(raw_dir: Path) -> dict:
    files = {}
    for p in raw_dir.rglob("*"):
        if p.suffix.lower() not in IMG_EXT or p.name.startswith(
                "._") or "__MACOSX" in p.parts:
            continue
        rel = p.relative_to(raw_dir)
        if rel.parts.count(
                "chest_xray") > 1:      # copie imbriquée chest_xray/chest_xray/
            continue
        files[rel.as_posix()] = p
    return files


def load_metadata(files: dict) -> pd.DataFrame:
    rows = []
    for rel in files:
        parts = Path(rel).parts
        stem = Path(rel).stem
        orig_split = next(
            (x for x in parts if x in (
                "train", "val", "test")), None)
        label = next(
            (x.upper() for x in parts if x.upper() in (
                "NORMAL", "PNEUMONIA")), None)

        m = re.match(r"person(\d+)_(bacteria|virus)", stem, re.I)
        if m:
            key, detail, inferred = f"person{
                m.group(1)}", m.group(2).lower(), True
        else:
            m = re.search(r"IM-(\d+)", stem)
            key, inferred = (f"IM-{m.group(1)}", True) if m else (stem, False)
            detail = "normal" if label == "NORMAL" else None
        rows.append({"image_name": rel, "label": label, "patient_key": key,
                     "label_detail": detail, "orig_split": orig_split, "key_inferred": inferred})
    return pd.DataFrame(rows)


def prepare(meta: pd.DataFrame, files: dict, staging: Path):
    secret = config.PSEUDO_SECRET
    meta = meta.drop_duplicates("image_name").copy()
    meta["patient_pseudo_id"] = meta["patient_key"].map(
        lambda k: pseudo_id(f"{SOURCE}:{k}", secret, "P"))
    meta["image_id"] = meta["image_name"].map(
        lambda n: pseudo_id(f"{SOURCE}:{n}", secret, "I"))
    # pas de données cliniques dans cette source : NULL explicite (jamais
    # imputé)
    meta["sex"] = None
    meta["age_years"] = np.nan
    meta["age_missing"] = 1
    meta["view_position"] = None
    meta["follow_up_number"] = None

    img, kept, n_dupes = stage_images(meta, files, staging)

    # seule la pathologie « Pneumonia » est connue ; les 13 autres = inconnues
    # (pas 0)
    labels = pd.DataFrame({
        "image_id": kept["image_id"].values,
        "pathology": "Pneumonia",
        "present": (kept["label"] == "PNEUMONIA").astype(int).values,
    })
    tables = assemble(kept, img, labels, SOURCE)
    return tables, {"duplicates_dropped": n_dupes,
                    "images_prepared": len(tables["images"])}
