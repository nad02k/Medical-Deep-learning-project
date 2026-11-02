"""Stockage des images dans PostgreSQL (table image_blobs, colonne BYTEA) + lecture pour le modèle.

STORAGE_BACKEND=postgres (défaut) | local (copie dans data/object_store/)
"""
import io
import os
import shutil
from pathlib import Path

from PIL import Image
from sqlalchemy import text

from . import config, db

BACKEND = os.getenv("STORAGE_BACKEND", "postgres")
OBJECT_STORE = config.ROOT / "data" / "object_store"
BATCH = 50


def upload_images(staging: Path, object_keys: list) -> int:
    staging = Path(staging)
    if BACKEND == "local":
        for key in object_keys:
            dst = OBJECT_STORE / key
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(staging / Path(key).name, dst)
        return len(object_keys)

    eng = db.engine()
    sql = text("INSERT INTO image_blobs (image_id, content, mime_type, size_bytes) "
               "VALUES (:i, :c, 'image/png', :n)")
    for start in range(0, len(object_keys), BATCH):
        batch = []
        for key in object_keys[start:start + BATCH]:
            data = (staging / Path(key).name).read_bytes()
            batch.append({"i": Path(key).stem, "c": data, "n": len(data)})
        with eng.begin() as c:          # une transaction par lot : mémoire bornée
            c.execute(sql, batch)
    return len(object_keys)


# ---------- lecture (pour le Dataset PyTorch) ----------
def fetch_image_bytes(image_id: str, engine=None) -> bytes:
    eng = engine or db.engine()
    with eng.connect() as c:
        row = c.execute(text("SELECT content FROM image_blobs WHERE image_id=:i"), {"i": image_id}).first()
    if row is None:
        raise KeyError(f"Image introuvable : {image_id}")
    return bytes(row[0])


def fetch_image(image_id: str, engine=None) -> Image.Image:
    """Retourne une image PIL en niveaux de gris."""
    return Image.open(io.BytesIO(fetch_image_bytes(image_id, engine))).convert("L")
