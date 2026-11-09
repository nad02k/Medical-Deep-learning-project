"""Chargement transactionnel PostgreSQL (par source, idempotent) + contrôles post-ingestion."""
import json
import re
from datetime import datetime, timezone

from sqlalchemy import create_engine, text

from . import config


def engine():
    return create_engine(config.DB_URL)


def _apply_schema(c):
    ddl = re.sub(
        r"--[^\n]*",
        "",
        config.SCHEMA_SQL.read_text(
            encoding="utf-8"))
    for stmt in ddl.split(";"):
        if stmt.strip():
            c.execute(text(stmt))


def _write_audit(c, run_id, ts, prefix, audit):
    for step, metrics in audit.items():
        for m, v in metrics.items():
            c.execute(text("INSERT INTO ingestion_audit VALUES (:r,:t,:s,:m,:v)"),
                      {"r": run_id, "t": ts, "s": f"{prefix}:{step}", "m": m, "v": json.dumps(v, default=str)})


def load(source: str, tables: dict, run_id: str, audit: dict) -> None:
    """Remplace uniquement les lignes de `source` (les autres sources ne sont pas touchées)."""
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    p = {"s": source}
    with engine().begin() as c:
        _apply_schema(c)
        c.execute(
            text("DELETE FROM image_blobs WHERE image_id IN (SELECT image_id FROM images WHERE source=:s)"),
            p)
        c.execute(
            text(
                "DELETE FROM labels WHERE image_id IN (SELECT image_id FROM images WHERE source=:s)"),
            p)
        c.execute(
            text("DELETE FROM clinical_features WHERE image_id IN (SELECT image_id FROM images WHERE source=:s)"),
            p)
        c.execute(text("DELETE FROM images WHERE source=:s"), p)
        c.execute(
            text("DELETE FROM splits WHERE patient_pseudo_id IN (SELECT patient_pseudo_id FROM patients WHERE source=:s)"),
            p)
        c.execute(text("DELETE FROM patients WHERE source=:s"), p)
        for name in ("patients", "images",
                     "clinical_features", "labels", "splits"):
            tables[name].to_sql(
                name,
                c,
                if_exists="append",
                index=False,
                method="multi",
                chunksize=2000)
        _write_audit(c, run_id, ts, source, audit)


EHR_ORDER = [
    "ehr_patients",
    "ehr_admissions",
    "ehr_diagnoses",
    "ehr_labevents"]


def load_ehr(tables: dict, run_id: str, audit: dict) -> None:
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with engine().begin() as c:
        _apply_schema(c)
        for t in reversed(EHR_ORDER):
            c.execute(text(f"DELETE FROM {t}"))
        for t in EHR_ORDER:
            tables[t].to_sql(
                t,
                c,
                if_exists="append",
                index=False,
                method="multi",
                chunksize=2000)
        _write_audit(c, run_id, ts, "mimic_iv_demo", audit)


def post_checks() -> dict:
    q = {
        "images_without_patient": "SELECT COUNT(*) FROM images i LEFT JOIN patients p USING(patient_pseudo_id) WHERE p.patient_pseudo_id IS NULL",
        "images_without_clinical": "SELECT COUNT(*) FROM images i LEFT JOIN clinical_features c USING(image_id) WHERE c.image_id IS NULL",
        "images_without_labels": "SELECT COUNT(*) FROM images i WHERE NOT EXISTS (SELECT 1 FROM labels l WHERE l.image_id=i.image_id)",
        "images_without_blob": "SELECT COUNT(*) FROM images i LEFT JOIN image_blobs b USING(image_id) WHERE b.image_id IS NULL",
        "patients_without_split": "SELECT COUNT(*) FROM patients p LEFT JOIN splits s USING(patient_pseudo_id) WHERE s.patient_pseudo_id IS NULL",
    }
    out = {}
    with engine().connect() as c:
        for k, sql in q.items():
            out[k] = int(c.execute(text(sql)).scalar() or 0)
        rows = c.execute(text(
            "SELECT i.source, COUNT(*) AS n_images, COUNT(DISTINCT i.patient_pseudo_id) AS n_patients, "
            "ROUND(100.0*SUM(CASE WHEN c.age_years IS NULL THEN 1 ELSE 0 END)/COUNT(*),2) AS age_null_pct "
            "FROM images i JOIN clinical_features c USING(image_id) GROUP BY i.source")).mappings().all()
        out["per_source"] = {
            r["source"]: {
                k: float(
                    r[k]) for k in (
                    "n_images",
                    "n_patients",
                    "age_null_pct")} for r in rows}
    bad = [k for k in q if out[k]]
    out["ok"], out["failed_checks"] = not bad, bad
    return out


def post_checks_ehr() -> dict:
    q = {
        "n_patients": "SELECT COUNT(*) FROM ehr_patients",
        "n_admissions": "SELECT COUNT(*) FROM ehr_admissions",
        "n_diagnoses": "SELECT COUNT(*) FROM ehr_diagnoses",
        "n_labevents": "SELECT COUNT(*) FROM ehr_labevents",
        "labs_without_patient": "SELECT COUNT(*) FROM ehr_labevents l LEFT JOIN ehr_patients p USING(patient_pseudo_id) WHERE p.patient_pseudo_id IS NULL",
        "labs_without_label": "SELECT COUNT(*) FROM ehr_labevents WHERE label IS NULL",
        "labs_valuenum_null_pct": "SELECT ROUND(100.0*SUM(CASE WHEN valuenum IS NULL THEN 1 ELSE 0 END)/COUNT(*),2) FROM ehr_labevents",
    }
    out = {}
    with engine().connect() as c:
        for k, sql in q.items():
            out[k] = float(c.execute(text(sql)).scalar() or 0)
    bad = []
    if out["n_patients"] == 0 or out["n_labevents"] == 0:
        bad.append("tables vides")
    if out["labs_without_patient"]:
        bad.append("labs_without_patient")
    out["ok"], out["failed_checks"] = not bad, bad
    return out
