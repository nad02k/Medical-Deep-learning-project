import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

RAW = ROOT / os.getenv("RAW_DIR", "data/raw")
STAGING = ROOT / "data" / "staging"
REPORTS = ROOT / "reports"
SCHEMA_SQL = ROOT / "sql" / "schema.sql"

DB_URL = (
    f"postgresql+psycopg2://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
    f"@{os.getenv('POSTGRES_HOST', 'localhost')}:{os.getenv('POSTGRES_PORT', '5432')}/{os.getenv('POSTGRES_DB')}"
)

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "chestxray")

PSEUDO_SECRET = os.getenv("PSEUDO_SECRET", "")

MAX_AGE = 110            # le NIH contient des âges aberrants (ex. 414)
SPLIT_RATIOS = (70, 85)  # <70 train, <85 val, sinon test

PATHOLOGIES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass", "Nodule",
    "Pleural_Thickening", "Pneumonia", "Pneumothorax",
]
