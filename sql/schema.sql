CREATE TABLE IF NOT EXISTS patients (
    patient_pseudo_id TEXT PRIMARY KEY,
    source            TEXT NOT NULL,
    sex               TEXT,
    age_years         REAL,
    age_missing       INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS images (
    image_id          TEXT PRIMARY KEY,
    patient_pseudo_id TEXT NOT NULL REFERENCES patients(patient_pseudo_id),
    source            TEXT NOT NULL,
    object_key        TEXT NOT NULL,
    sha256            TEXT NOT NULL,
    width             INTEGER,
    height            INTEGER,
    view_position     TEXT,
    follow_up_number  INTEGER,
    label_detail      TEXT
);

CREATE TABLE IF NOT EXISTS image_blobs (
    image_id   TEXT PRIMARY KEY REFERENCES images(image_id),
    content    BYTEA NOT NULL,
    mime_type  TEXT NOT NULL,
    size_bytes INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS clinical_features (
    image_id          TEXT PRIMARY KEY REFERENCES images(image_id),
    patient_pseudo_id TEXT NOT NULL REFERENCES patients(patient_pseudo_id),
    age_years         REAL,
    sex               TEXT,
    view_position     TEXT,
    follow_up_number  INTEGER,
    lab_json          TEXT
);

CREATE TABLE IF NOT EXISTS labels (
    image_id  TEXT NOT NULL REFERENCES images(image_id),
    pathology TEXT NOT NULL,
    present   INTEGER NOT NULL CHECK (present IN (0,1)),
    PRIMARY KEY (image_id, pathology)
);

CREATE TABLE IF NOT EXISTS splits (
    patient_pseudo_id TEXT PRIMARY KEY REFERENCES patients(patient_pseudo_id),
    split             TEXT NOT NULL CHECK (split IN ('train','val','test'))
);

CREATE TABLE IF NOT EXISTS ingestion_audit (
    run_id TEXT NOT NULL,
    run_ts TEXT NOT NULL,
    step   TEXT NOT NULL,
    metric TEXT NOT NULL,
    value  TEXT
);

CREATE INDEX IF NOT EXISTS idx_images_patient ON images(patient_pseudo_id);
CREATE INDEX IF NOT EXISTS idx_images_source ON images(source);
CREATE INDEX IF NOT EXISTS idx_labels_path ON labels(pathology);

CREATE TABLE IF NOT EXISTS ehr_patients (
    patient_pseudo_id TEXT PRIMARY KEY,
    gender            TEXT,
    anchor_age        INTEGER,
    anchor_year_group TEXT,
    deceased          INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS ehr_admissions (
    hadm_pseudo_id       TEXT PRIMARY KEY,
    patient_pseudo_id    TEXT NOT NULL REFERENCES ehr_patients(patient_pseudo_id),
    admittime            TIMESTAMP,
    dischtime            TIMESTAMP,
    admission_type       TEXT,
    hospital_expire_flag INTEGER
);

CREATE TABLE IF NOT EXISTS ehr_diagnoses (
    hadm_pseudo_id    TEXT NOT NULL REFERENCES ehr_admissions(hadm_pseudo_id),
    patient_pseudo_id TEXT NOT NULL REFERENCES ehr_patients(patient_pseudo_id),
    seq_num           INTEGER NOT NULL,
    icd_code          TEXT NOT NULL,
    icd_version       INTEGER NOT NULL,
    long_title        TEXT,
    PRIMARY KEY (hadm_pseudo_id, seq_num)
);

CREATE TABLE IF NOT EXISTS ehr_labevents (
    labevent_id       BIGINT PRIMARY KEY,
    patient_pseudo_id TEXT NOT NULL REFERENCES ehr_patients(patient_pseudo_id),
    hadm_pseudo_id    TEXT REFERENCES ehr_admissions(hadm_pseudo_id),
    itemid            INTEGER NOT NULL,
    label             TEXT,
    fluid             TEXT,
    category          TEXT,
    charttime         TIMESTAMP,
    value_text        TEXT,
    valuenum          DOUBLE PRECISION,
    valueuom          TEXT,
    ref_range_lower   DOUBLE PRECISION,
    ref_range_upper   DOUBLE PRECISION,
    flag              TEXT
);

CREATE INDEX IF NOT EXISTS idx_ehr_labs_patient ON ehr_labevents(patient_pseudo_id, itemid);
