"""Collecte : Kaggle (NIH, Pneumonia) + PhysioNet MIMIC-IV demo (accès ouvert, licence ODbL)."""
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from . import config

KAGGLE = {
    "nih_sample": "nih-chest-xrays/sample",
    "nih_full": "nih-chest-xrays/data",
    # si échec : copier le slug exact depuis la page Kaggle du dataset
    "pneumonia": "paultimothymooney/chest-xray-pneumonia",
}

MIMIC_DEMO_BASE = "https://physionet.org/files/mimic-iv-demo/2.2/"
MIMIC_DEMO_FILES = [f"hosp/{n}.csv.gz" for n in
                    ("patients", "admissions", "diagnoses_icd", "d_icd_diagnoses", "labevents", "d_labitems")]


def sha256_file(p: Path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def write_manifest(folder: Path, source: str) -> Path:
    files = [p for p in sorted(folder.rglob("*"))
             if p.is_file() and p.name != "MANIFEST.json"]
    manifest = {
        "source": source,
        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_files": len(files),
        "files": {p.relative_to(folder).as_posix(): sha256_file(p) for p in files},
    }
    out = folder / "MANIFEST.json"
    out.write_text(json.dumps(manifest, indent=1))
    return out


def download(source: str) -> Path:
    """Kaggle : le cache kagglehub est placé dans data/raw pour que DVC le suive."""
    config.RAW.mkdir(parents=True, exist_ok=True)
    os.environ["KAGGLEHUB_CACHE"] = str(config.RAW)
    import kagglehub  # nécessite KAGGLE_USERNAME / KAGGLE_KEY ou kaggle.json

    path = Path(kagglehub.dataset_download(KAGGLE[source]))
    write_manifest(path, source)
    return path


def _parse_sums(txt: str) -> dict:
    sums = {}
    for line in txt.splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2:
            sums[parts[1].strip().lstrip("*").removeprefix("./")] = parts[0]
    return sums


def _get_text(url: str, attempts: int = 6) -> str:
    import requests
    for i in range(attempts):
        try:
            return requests.get(url, timeout=(15, 60), headers={
                                "Accept-Encoding": "identity"}).text
        except requests.RequestException as e:
            print(f"  réessai {i + 1}/{attempts} ({type(e).__name__})")
            time.sleep(min(2 ** i, 30))
    raise RuntimeError(f"Impossible de télécharger {url}")


def _download_file(url: str, out: Path, expected_sha: str,
                   attempts: int = 12) -> None:
    """Téléchargement avec REPRISE (en-tête Range) : une coupure ne fait pas repartir de zéro.
    Le fichier n'est publié que si son SHA-256 correspond à la liste officielle."""
    import requests

    if out.exists() and sha256_file(out) == expected_sha:
        print(f"  {out.name} déjà présent et vérifié")
        return
    tmp = out.with_name(out.name + ".part")
    for i in range(attempts):
        pos = tmp.stat().st_size if tmp.exists() else 0
        headers = {"Accept-Encoding": "identity"}
        if pos:
            headers["Range"] = f"bytes={pos}-"
        try:
            with requests.get(url, headers=headers, stream=True, timeout=(15, 60)) as r:
                if r.status_code != 416:           # 416 = tout est déjà là
                    r.raise_for_status()
                    mode = "ab" if (pos and r.status_code == 206) else "wb"
                    with open(tmp, mode) as f:
                        for chunk in r.iter_content(64 * 1024):
                            f.write(chunk)
        except requests.RequestException as e:
            print(f"  {out.name}: coupure ({type(e).__name__}), reprise à {tmp.stat().st_size if tmp.exists() else 0} octets "
                  f"[{i + 1}/{attempts}]")
            time.sleep(min(2 ** i, 20))
            continue
        if tmp.exists() and sha256_file(tmp) == expected_sha:
            tmp.replace(out)
            print(f"  {out.name} OK")
            return
        # contenu invalide : on repart de zéro
        tmp.unlink(missing_ok=True)
        print(
            f"  {
                out.name}: checksum invalide, nouvelle tentative [{
                i + 1}/{attempts}]")
    raise RuntimeError(
        f"Échec du téléchargement de {url} après {attempts} tentatives")


def download_mimic_demo() -> Path:
    """MIMIC-IV demo (100 patients) : 6 fichiers utiles, reprise sur coupure + vérification SHA-256."""
    dest = config.RAW / "mimic_iv_demo"
    sums = _parse_sums(_get_text(MIMIC_DEMO_BASE + "SHA256SUMS.txt"))
    for rel in MIMIC_DEMO_FILES:
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        expected = sums.get(rel)
        if expected is None:
            raise RuntimeError(
                f"{rel} absent de SHA256SUMS.txt : intégrité non vérifiable")
        _download_file(MIMIC_DEMO_BASE + rel, out, expected)
    write_manifest(dest, "mimic_iv_demo")
    return dest
