"""Orchestration Prefect multi-sources (lancé via run.py)."""
import json
import shutil
import uuid
from pathlib import Path

from prefect import flow, task

from src import collect, config, db, mimic_demo, pneumonia, storage, transform, validate

t_collect_kaggle = task(collect.download, name="collect-kaggle", retries=2, retry_delay_seconds=30)
t_collect_mimic = task(collect.download_mimic_demo, name="collect-mimic-demo", retries=2, retry_delay_seconds=30)
t_validate_nih = task(validate.validate_nih, name="validate-nih")
t_validate_pneumo = task(validate.validate_pneumonia, name="validate-pneumonia")
t_validate_ehr = task(validate.validate_ehr, name="validate-ehr")
t_upload = task(storage.upload_images, name="upload-minio", retries=2, retry_delay_seconds=10)
t_load = task(db.load, name="load-postgres")
t_load_ehr = task(db.load_ehr, name="load-postgres-ehr")
t_post = task(db.post_checks, name="post-checks")
t_post_ehr = task(db.post_checks_ehr, name="post-checks-ehr")


def _fresh_staging() -> Path:
    shutil.rmtree(config.STAGING, ignore_errors=True)
    config.STAGING.mkdir(parents=True, exist_ok=True)
    return config.STAGING


def _report(name: str, run_id: str, payload: dict) -> None:
    config.REPORTS.mkdir(exist_ok=True)
    (config.REPORTS / f"{name}_{run_id}.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


@flow(name="ingest-nih", log_prints=True)
def ingest_nih(source: str = "nih_sample", raw_dir: str | None = None):
    run_id = uuid.uuid4().hex[:8]
    raw = Path(raw_dir) if raw_dir else Path(t_collect_kaggle(source))
    csv, files = transform.find_inputs(raw)
    meta = transform.load_metadata(csv)
    pre = t_validate_nih(meta, set(files))
    tables, stats = transform.prepare(meta, files, _fresh_staging())
    t_load(transform.SOURCE, tables, run_id, {"pre_ingest": pre, "ingest": stats})
    n_up = t_upload(config.STAGING, tables["images"]["object_key"].tolist())
    post = t_post()
    _report("nih", run_id, {"pre": pre, "stats": {**stats, "images_stored_in_db": n_up}, "post": post})
    if not post["ok"]:
        raise RuntimeError(f"Post-checks échoués : {post['failed_checks']}")
    return post


@flow(name="ingest-pneumonia", log_prints=True)
def ingest_pneumonia(raw_dir: str | None = None):
    run_id = uuid.uuid4().hex[:8]
    raw = Path(raw_dir) if raw_dir else Path(t_collect_kaggle("pneumonia"))
    files = pneumonia.find_inputs(raw)
    meta = pneumonia.load_metadata(files)
    pre = t_validate_pneumo(meta)
    tables, stats = pneumonia.prepare(meta, files, _fresh_staging())
    t_load(pneumonia.SOURCE, tables, run_id, {"pre_ingest": pre, "ingest": stats})
    n_up = t_upload(config.STAGING, tables["images"]["object_key"].tolist())
    post = t_post()
    _report("pneumonia", run_id, {"pre": pre, "stats": {**stats, "images_stored_in_db": n_up}, "post": post})
    if not post["ok"]:
        raise RuntimeError(f"Post-checks échoués : {post['failed_checks']}")
    return post


@flow(name="ingest-mimic-demo", log_prints=True)
def ingest_mimic_demo(raw_dir: str | None = None):
    run_id = uuid.uuid4().hex[:8]
    raw_path = Path(raw_dir) if raw_dir else Path(t_collect_mimic())
    raw = mimic_demo.load_raw(raw_path)
    pre = t_validate_ehr(raw["patients"], raw["labevents"])
    tables, stats = mimic_demo.prepare(raw)
    t_load_ehr(tables, run_id, {"pre_ingest": pre, "ingest": stats})
    post = t_post_ehr()
    _report("mimic_demo", run_id, {"pre": pre, "stats": stats, "post": post})
    if not post["ok"]:
        raise RuntimeError(f"Post-checks échoués : {post['failed_checks']}")
    return post
