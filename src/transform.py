"""Briques communes + préparation NIH ChestX-ray14."""
import hashlib
from pathlib import Path

import pandas as pd

from . import config
from .anonymize import pseudo_id, reencode_image

SOURCE = "nih_cxr14"
COLS = {
    "Image Index": "image_name",
    "Finding Labels": "finding_labels",
    "Follow-up #": "follow_up_number",
    "Patient ID": "patient_raw_id",
    "Patient Age": "age_raw",
    "Patient Gender": "sex",
    "View Position": "view_position",
}


def assign_split(pseudo: str) -> str:
    h = int(hashlib.sha256(pseudo.encode()).hexdigest(), 16) % 100
    if h < config.SPLIT_RATIOS[0]:
        return "train"
    return "val" if h < config.SPLIT_RATIOS[1] else "test"


def stage_images(meta: pd.DataFrame, files: dict, staging: Path):
    """Ré-encode les images (sans métadonnées) dans `staging`, écarte les doublons de pixels."""
    rows, seen = [], set()
    for r in meta.itertuples():
        out = staging / f"{r.image_id}.png"
        info = reencode_image(files[r.image_name], out)
        if info["sha256"] in seen:
            out.unlink(missing_ok=True)
            continue
        seen.add(info["sha256"])
        rows.append({"image_id": r.image_id, "object_key": f"images/{r.image_id}.png", **info})
    img = pd.DataFrame(rows, columns=["image_id", "object_key", "sha256", "width", "height"])
    kept = meta[meta["image_id"].isin(img["image_id"])]
    return img, kept, len(meta) - len(img)


def assemble(meta: pd.DataFrame, img: pd.DataFrame, labels: pd.DataFrame, source: str) -> dict:
    images = meta[["image_id", "patient_pseudo_id", "view_position", "follow_up_number", "label_detail"]].merge(img, on="image_id")
    images["source"] = source
    images = images[["image_id", "patient_pseudo_id", "source", "object_key", "sha256", "width", "height",
                     "view_position", "follow_up_number", "label_detail"]]

    patients = (meta.groupby("patient_pseudo_id")
                .agg(sex=("sex", "first"), age_years=("age_years", "min"), age_missing=("age_missing", "max"))
                .reset_index())
    patients["source"] = source
    patients = patients[["patient_pseudo_id", "source", "sex", "age_years", "age_missing"]]

    clinical = meta[["image_id", "patient_pseudo_id", "age_years", "sex", "view_position", "follow_up_number"]].copy()
    clinical["lab_json"] = None

    splits = pd.DataFrame({"patient_pseudo_id": patients["patient_pseudo_id"]})
    splits["split"] = splits["patient_pseudo_id"].map(assign_split)

    return {"patients": patients, "images": images, "clinical_features": clinical,
            "labels": labels[labels["image_id"].isin(images["image_id"])], "splits": splits}


# ---------- NIH ----------
def find_inputs(raw_dir: Path):
    candidates = [p for pat in ("Data_Entry_2017*.csv", "sample_labels.csv", "*labels*.csv")
                  for p in raw_dir.rglob(pat)]
    if not candidates:
        raise FileNotFoundError(f"Aucun CSV de métadonnées NIH trouvé dans {raw_dir}")
    files = {p.name: p for p in raw_dir.rglob("*.png")}
    return candidates[0], files


def load_metadata(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path).rename(columns=COLS)[list(COLS.values())].copy()
    df["sex"] = df["sex"].astype("string").str.strip().str.upper().where(lambda s: s.isin(["M", "F"]))
    df["view_position"] = df["view_position"].astype("string").str.strip().str.upper()
    return df


def parse_age(s: pd.Series) -> pd.Series:
    a = pd.to_numeric(s.astype(str).str.extract(r"(\d+)")[0], errors="coerce")  # gère "058Y"
    return a.where((a >= 0) & (a <= config.MAX_AGE))


def prepare(meta: pd.DataFrame, files: dict, staging: Path):
    secret = config.PSEUDO_SECRET
    meta = meta[meta["image_name"].isin(files)].drop_duplicates("image_name").copy()
    meta["age_years"] = parse_age(meta["age_raw"])
    meta["age_missing"] = meta["age_years"].isna().astype(int)
    meta["patient_pseudo_id"] = meta["patient_raw_id"].map(lambda x: pseudo_id(f"{SOURCE}:{x}", secret, "P"))
    meta["image_id"] = meta["image_name"].map(lambda n: pseudo_id(f"{SOURCE}:{n}", secret, "I"))
    meta["label_detail"] = None

    img, kept, n_dupes = stage_images(meta, files, staging)

    lab = kept[["image_id", "finding_labels"]].copy()
    for p in config.PATHOLOGIES:
        lab[p] = lab["finding_labels"].fillna("").str.split("|").map(lambda L, p=p: int(p in L))
    labels = lab.melt(id_vars="image_id", value_vars=config.PATHOLOGIES, var_name="pathology", value_name="present")

    tables = assemble(kept, img, labels, SOURCE)
    return tables, {"duplicates_dropped": n_dupes, "images_prepared": len(tables["images"])}
