import os
from pathlib import Path
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# Charge la configuration externalisée YAML (params.yaml)
PARAMS_FILE = ROOT / "configs" / "params.yaml"
if PARAMS_FILE.exists():
    with open(PARAMS_FILE, "r", encoding="utf-8") as f:
        PARAMS = yaml.safe_load(f) or {}
else:
    PARAMS = {}

# Répertoires
RAW = ROOT / os.getenv("RAW_DIR", PARAMS.get("data",
                       {}).get("raw_dir", "data/raw"))
STAGING = ROOT / PARAMS.get("data", {}).get("staging_dir", "data/staging")
REPORTS = ROOT / PARAMS.get("data", {}).get("reports_dir", "reports")
SCHEMA_SQL = ROOT / "sql" / "schema.sql"

# Base de données PostgreSQL
DB_URL = (
    f"postgresql+psycopg2://{
        os.getenv('POSTGRES_USER')}:{
        os.getenv('POSTGRES_PASSWORD')}"
    f"@{os.getenv('POSTGRES_HOST',
                  'localhost')}:{os.getenv('POSTGRES_PORT',
                                           '5432')}/{os.getenv('POSTGRES_DB')}"
)

# MinIO / S3 config (pour compatibilité)
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "chestxray")

# Stockage et lots
STORAGE_BACKEND = os.getenv(
    "STORAGE_BACKEND",
    PARAMS.get(
        "storage",
        {}).get(
            "backend",
        "postgres"))
BATCH_SIZE = PARAMS.get("storage", {}).get("batch_size", 50)

# Pseudonymisation
PSEUDO_SECRET = os.getenv("PSEUDO_SECRET", "")

# Règles métier & validation (externalisées depuis params.yaml)
MAX_AGE = PARAMS.get("validation", {}).get("max_age", 110)
SPLIT_RATIOS = tuple(
    PARAMS.get(
        "validation", {}).get(
            "split_ratios", [
                70, 85]))

PATHOLOGIES = PARAMS.get("pathologies", [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass", "Nodule",
    "Pleural_Thickening", "Pneumonia", "Pneumothorax",
])

# Paramètres fusion
FUSION_CONFIG = PARAMS.get("fusion", {})
COLLECT_CONFIG = PARAMS.get("collect", {})
SOURCES_CONFIG = PARAMS.get("sources", {})
