"""Validation Pandera + contrôles d'alignement, pour chaque source."""
import pandas as pd
import pandera.pandas as pa

from . import config


def _schema_check(schema, df, report, label):
    try:
        schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as e:
        report["critical"].append(f"{label} : {len(e.failure_cases)} violations de schéma")
        report.setdefault("schema_failures", []).extend(e.failure_cases.head(20).to_dict("records"))


def _finish(report, strict):
    if strict and report["critical"]:
        raise ValueError(f"Validation échouée : {report['critical']}")
    return report


# ---------- NIH ----------
nih_schema = pa.DataFrameSchema(
    {
        "image_name": pa.Column(str, unique=True),
        "patient_raw_id": pa.Column(int, coerce=True),
        "follow_up_number": pa.Column(int, pa.Check.ge(0), coerce=True),
        "sex": pa.Column(checks=pa.Check.isin(["M", "F"]), nullable=True),
        "view_position": pa.Column(checks=pa.Check.isin(["PA", "AP"]), nullable=True),
    },
    strict=False,
)


def validate_nih(meta: pd.DataFrame, image_names: set, strict: bool = True) -> dict:
    report = {"n_rows": int(len(meta)), "n_files": len(image_names), "critical": [], "warnings": []}
    _schema_check(nih_schema, meta, report, "NIH")

    csv_names = set(meta["image_name"])
    rows_wo_img = csv_names - image_names
    imgs_wo_row = image_names - csv_names
    report["rows_without_image"], report["images_without_row"] = len(rows_wo_img), len(imgs_wo_row)
    if rows_wo_img:
        report["warnings"].append(f"{len(rows_wo_img)} lignes CSV sans image (exclues)")
    if imgs_wo_row:
        report["warnings"].append(f"{len(imgs_wo_row)} images sans ligne CSV (exclues)")

    conflicts = meta.dropna(subset=["sex"]).groupby("patient_raw_id")["sex"].nunique()
    n_conf = int((conflicts > 1).sum())
    report["patients_with_sex_conflict"] = n_conf
    if n_conf:
        report["critical"].append(f"{n_conf} patients avec sexe contradictoire")

    age = pd.to_numeric(meta["age_raw"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    report["age_missing_pct"] = round(float(age.isna().mean() * 100), 2)
    report["age_out_of_range"] = int(((age < 0) | (age > config.MAX_AGE)).sum())
    if report["age_out_of_range"]:
        report["warnings"].append(f"{report['age_out_of_range']} âges hors [0,{config.MAX_AGE}] -> NULL")
    return _finish(report, strict)


# ---------- Kaggle Pneumonia ----------
pneumo_schema = pa.DataFrameSchema(
    {
        "image_name": pa.Column(str, unique=True),
        "label": pa.Column(str, pa.Check.isin(["NORMAL", "PNEUMONIA"])),
        "patient_key": pa.Column(str),
    },
    strict=False,
)


def validate_pneumonia(meta: pd.DataFrame, strict: bool = True) -> dict:
    report = {"n_images": int(len(meta)), "critical": [], "warnings": []}
    if meta.empty:
        report["critical"].append("aucune image trouvée (structure du dataset inattendue ?)")
        return _finish(report, strict)
    _schema_check(pneumo_schema, meta, report, "Pneumonia")

    n_fallback = int((~meta["key_inferred"]).sum())
    report["images_without_patient_id_in_filename"] = n_fallback
    if n_fallback:
        report["warnings"].append(f"{n_fallback} images sans ID patient déductible du nom (1 image = 1 patient présumé)")

    leak = meta.groupby("patient_key")["orig_split"].nunique()
    n_leak = int((leak > 1).sum())
    report["patients_in_multiple_original_splits"] = n_leak
    if n_leak:
        report["warnings"].append(f"{n_leak} patients dans plusieurs splits Kaggle d'origine (fuite) : split recalculé par patient")

    report["class_balance"] = meta["label"].value_counts().to_dict()
    return _finish(report, strict)


# ---------- MIMIC-IV demo (EHR) ----------
ehr_patients_schema = pa.DataFrameSchema(
    {
        "subject_id": pa.Column(int, unique=True, coerce=True),
        "gender": pa.Column(checks=pa.Check.isin(["M", "F"]), nullable=True),
        "anchor_age": pa.Column(int, pa.Check.in_range(0, 120), coerce=True),
    },
    strict=False,
)
ehr_labs_schema = pa.DataFrameSchema(
    {
        "labevent_id": pa.Column(int, unique=True, coerce=True),
        "itemid": pa.Column(int, coerce=True),
    },
    strict=False,
)


def validate_ehr(patients: pd.DataFrame, labs: pd.DataFrame, strict: bool = True) -> dict:
    report = {"n_patients": int(len(patients)), "n_labevents": int(len(labs)), "critical": [], "warnings": []}
    _schema_check(ehr_patients_schema, patients, report, "EHR patients")
    _schema_check(ehr_labs_schema, labs, report, "EHR labevents")
    report["labs_valuenum_missing_pct"] = round(float(labs["valuenum"].isna().mean() * 100), 2)
    report["labs_without_hadm_pct"] = round(float(labs["hadm_id"].isna().mean() * 100), 2)
    orphans = int((~labs["subject_id"].isin(patients["subject_id"])).sum())
    if orphans:
        report["warnings"].append(f"{orphans} labos sans patient (exclus)")
    return _finish(report, strict)
