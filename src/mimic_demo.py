"""MIMIC-IV demo (PhysioNet, accès ouvert) -> tables ehr_* pseudonymisées.
Minimisation : race, insurance, language, marital_status et le texte libre des labos ne sont PAS ingérés."""
from pathlib import Path

import pandas as pd

from . import config
from .anonymize import pseudo_id

SOURCE = "mimic_iv_demo"


def _read(raw: Path, name: str, **kw) -> pd.DataFrame:
    return pd.read_csv(raw / "hosp" / f"{name}.csv.gz", **kw)


def load_raw(raw_dir: Path) -> dict:
    return {
        "patients": _read(raw_dir, "patients",
                          usecols=["subject_id", "gender", "anchor_age", "anchor_year_group", "dod"]),
        "admissions": _read(raw_dir, "admissions",
                            usecols=["subject_id", "hadm_id", "admittime", "dischtime",
                                     "admission_type", "hospital_expire_flag"],
                            parse_dates=["admittime", "dischtime"]),
        "diagnoses": _read(raw_dir, "diagnoses_icd", dtype={"icd_code": str}),
        "d_icd": _read(raw_dir, "d_icd_diagnoses", dtype={"icd_code": str}),
        "labevents": _read(raw_dir, "labevents",
                           usecols=["labevent_id", "subject_id", "hadm_id", "itemid", "charttime", "value",
                                    "valuenum", "valueuom", "ref_range_lower", "ref_range_upper", "flag"],
                           dtype={"hadm_id": "Int64", "value": str}, parse_dates=["charttime"]),
        "d_labitems": _read(raw_dir, "d_labitems"),
    }


def _pseudo(series: pd.Series, kind: str, prefix: str) -> pd.Series:
    s = series.astype("float64")
    mapping = {
        u: pseudo_id(
            f"{SOURCE}:{kind}:{
                int(u)}",
            config.PSEUDO_SECRET,
            prefix) for u in s.dropna().unique()}
    return s.map(mapping).astype(object).where(s.notna(), None)


def prepare(raw: dict):
    pat, adm = raw["patients"], raw["admissions"]

    ehr_patients = pd.DataFrame({
        "patient_pseudo_id": _pseudo(pat["subject_id"], "subject", "S"),
        "gender": pat["gender"].where(pat["gender"].isin(["M", "F"])),
        "anchor_age": pat["anchor_age"],
        "anchor_year_group": pat["anchor_year_group"],
        "deceased": pat["dod"].notna().astype(int),
    })

    adm = adm[adm["subject_id"].isin(pat["subject_id"])].copy()
    adm["hadm_pseudo_id"] = _pseudo(adm["hadm_id"], "hadm", "H")
    adm["patient_pseudo_id"] = _pseudo(adm["subject_id"], "subject", "S")
    ehr_adm = adm[["hadm_pseudo_id", "patient_pseudo_id", "admittime", "dischtime",
                   "admission_type", "hospital_expire_flag"]]

    dx = raw["diagnoses"].merge(raw["d_icd"][["icd_code", "icd_version", "long_title"]],
                                on=["icd_code", "icd_version"], how="left")
    dx = dx[dx["hadm_id"].isin(adm["hadm_id"])].copy()
    dx["hadm_pseudo_id"] = _pseudo(dx["hadm_id"], "hadm", "H")
    dx["patient_pseudo_id"] = _pseudo(dx["subject_id"], "subject", "S")
    ehr_dx = dx[["hadm_pseudo_id", "patient_pseudo_id",
                 "seq_num", "icd_code", "icd_version", "long_title"]]

    lab = raw["labevents"].merge(raw["d_labitems"][["itemid", "label", "fluid", "category"]],
                                 on="itemid", how="left")
    lab = lab[lab["subject_id"].isin(pat["subject_id"])].copy()
    lab["hadm_id"] = lab["hadm_id"].where(
        lab["hadm_id"].isin(set(adm["hadm_id"])))  # sinon NULL
    lab["patient_pseudo_id"] = _pseudo(lab["subject_id"], "subject", "S")
    lab["hadm_pseudo_id"] = _pseudo(lab["hadm_id"], "hadm", "H")
    lab = lab.rename(columns={"value": "value_text"})
    ehr_labs = lab[["labevent_id", "patient_pseudo_id", "hadm_pseudo_id", "itemid", "label", "fluid",
                    "category", "charttime", "value_text", "valuenum", "valueuom",
                    "ref_range_lower", "ref_range_upper", "flag"]]

    tables = {"ehr_patients": ehr_patients, "ehr_admissions": ehr_adm,
              "ehr_diagnoses": ehr_dx, "ehr_labevents": ehr_labs}
    return tables, {k: len(v) for k, v in tables.items()}
