"""Pseudonymisation (HMAC) et nettoyage des images (métadonnées supprimées)."""
import hashlib
import hmac
from pathlib import Path

from PIL import Image


def pseudo_id(raw_id, secret: str, prefix: str = "P") -> str:
    if not secret:
        raise ValueError("PSEUDO_SECRET manquant : refus d'ingérer sans pseudonymisation.")
    digest = hmac.new(secret.encode(), str(raw_id).encode(), hashlib.sha256).hexdigest()
    return f"{prefix}_{digest[:16]}"


def reencode_image(src: Path, dst: Path) -> dict:
    """Ré-encode en PNG niveaux de gris (EXIF / chunks texte supprimés) + hash des pixels."""
    with Image.open(src) as im:
        im = im.convert("L")
        pixel_hash = hashlib.sha256(im.tobytes()).hexdigest()
        dst.parent.mkdir(parents=True, exist_ok=True)
        im.save(dst, format="PNG")
        return {"sha256": pixel_hash, "width": im.width, "height": im.height}
